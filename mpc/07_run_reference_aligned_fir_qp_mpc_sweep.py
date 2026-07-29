#!/usr/bin/env python3
"""
C1-QP2 — reference-aligned empirical FIR-QP-MPC common tuning sweep.

Uses:
  - C0 MuJoCo replay baseline as the fixed-phase pitch reference;
  - C1-QP1 FIR response h from alpha to root-pitch deviation;
  - a small AR model of the uncontrolled disturbance decay;
  - a 20 ms constrained dynamic-matrix-control QP.

It tests identical controller candidates for forward and backward. C1 is only
accepted if the same candidate improves both cases and remains safe.
"""

from __future__ import annotations
from collections import deque
from pathlib import Path
import argparse, math, sys, time
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "model" / "biped_planar.xml"
TRAJ = ROOT / "transfer" / "trajectories"
LOG = ROOT / "results" / "logs"
FIG = ROOT / "results" / "figures"

JOINTS = [
    "left_hip_pitch_joint", "left_knee_pitch_joint", "left_ankle_pitch_joint",
    "right_hip_pitch_joint", "right_knee_pitch_joint", "right_ankle_pitch_joint",
]
ROOT_PITCH = "root_pitch_x"
TORSO = "hip_plate_link"

# Existing actuator-aware execution model
DELAY_S = 0.020
ACTUATOR_TAU_S = 0.030
MAX_SPEED_RAD_S = 8.0

# Online DMC / QP
TS = 0.020
N = 12                         # 0.24 s horizon
AR_ORDER = 4
QP_ITERS = 120
ALPHA_LIMIT_DEG = 5.0
ALPHA_SLEW_DEG = 1.0
G = np.array([0.27, 0.062, -0.228, 0.27, 0.062, -0.228], dtype=float)

DIST_T = 0.400
DIST_PITCH_DEG = 2.0
DIST_RATE = 0.25

CANDIDATES = [
    {"name": "conservative", "q": 800.0, "r": 50.0, "slew": 80.0},
    {"name": "balanced", "q": 1500.0, "r": 20.0, "slew": 25.0},
    {"name": "responsive", "q": 2500.0, "r": 10.0, "slew": 12.0},
    {"name": "aggressive", "q": 4000.0, "r": 5.0, "slew": 6.0},
]

# Common acceptance criteria
MIN_CONTACTS = 4
MIN_HEIGHT_RATIO = 0.65
MAX_PEAK_ERROR_DEG = 15.0
MAX_JOINT_CORR_DEG = 1.50
MIN_IAE_IMPROVEMENT = 0.02
MAX_PEAK_INCREASE = 0.05
MAX_FINAL_INCREASE_DEG = 0.05
SETTLING_BAND_DEG = 0.50


class ActuatorProxy:
    def __init__(self, n, dt):
        self.n = int(n); self.dt = float(dt)
        self.delay = int(round(DELAY_S / self.dt))
        self.hist = deque(maxlen=self.delay + 1)
        self.out = np.zeros(self.n)

    def reset(self, q):
        q = np.asarray(q, dtype=float).reshape(self.n)
        self.hist.clear()
        for _ in range(self.delay + 1):
            self.hist.append(q.copy())
        self.out = q.copy()

    def update(self, q):
        q = np.asarray(q, dtype=float).reshape(self.n)
        self.hist.append(q.copy())
        delayed = self.hist[0]
        gain = min(1.0, self.dt / ACTUATOR_TAU_S)
        delta = gain * (delayed - self.out)
        lim = MAX_SPEED_RAD_S * self.dt
        self.out += np.clip(delta, -lim, lim)
        return self.out.copy()


def require_id(model, kind, name):
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise RuntimeError(f"Required MuJoCo name not found: {name}")
    return int(value)


def map_actuators(model, joint_ids):
    result = []
    for jid in joint_ids:
        candidates = [i for i in range(model.nu) if int(model.actuator_trnid[i, 0]) == jid]
        if len(candidates) != 1:
            raise RuntimeError(f"Joint {jid} actuator mapping failed: {candidates}")
        result.append(candidates[0])
    return result


def interp(t, tx, values):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return float(np.interp(t, tx, values))
    return np.asarray([np.interp(t, tx, values[:, j]) for j in range(values.shape[1])])


def load_fddp(direction):
    path = TRAJ / f"com_shift_{direction}_ref.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as z:
        return path, np.asarray(z["t_ref"], dtype=float), np.asarray(z["q_ref"], dtype=float)


def load_baseline(direction):
    path = LOG / f"c1_qp0_baseline_{direction}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path}; run C1-QP0 calibration first.")
    df = pd.read_csv(path)
    return path, df["time_s"].to_numpy(), df["root_pitch_rad"].to_numpy(), df["root_pitch_rate_rad_s"].to_numpy()


def load_fir(direction):
    path = LOG / "c1_qp1_fir_identification.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{path}; run C1-QP1 identification first.")
    report = yaml.safe_load(path.read_text(encoding="utf-8"))
    h = np.asarray(report["directions"][direction]["fir_coefficients_rad_per_rad"], dtype=float)
    if len(h) < 3:
        raise RuntimeError("FIR kernel is too short.")
    return h


def fit_ar(direction, base_t, base_pitch):
    """Fit AR(d) to C0's measured disturbance relative to MuJoCo baseline."""
    path = LOG / f"c0_reference_only_{direction}_pitch_disturbance.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path}; run original C1-QP script first.")
    df = pd.read_csv(path)
    if "measured_pitch_rad" not in df:
        raise RuntimeError("C0 disturbance log lacks measured_pitch_rad.")

    end = min(float(df["time_s"].max()), float(base_t[-1]))
    t = np.arange(DIST_T, end + 1e-12, TS)
    measured = np.interp(t, df["time_s"], df["measured_pitch_rad"])
    ref = np.interp(t, base_t, base_pitch)
    e = measured - ref
    if len(e) <= AR_ORDER + 3:
        raise RuntimeError("Not enough samples to fit AR.")

    X = np.asarray([e[k-AR_ORDER:k][::-1] for k in range(AR_ORDER, len(e))])
    y = e[AR_ORDER:]
    a = np.linalg.solve(X.T @ X + 1e-7*np.eye(AR_ORDER), X.T @ y)

    comp = np.zeros((AR_ORDER, AR_ORDER))
    comp[0] = a
    comp[1:, :-1] = np.eye(AR_ORDER-1)
    radius = float(np.max(np.abs(np.linalg.eigvals(comp))))
    stabilized = False
    if radius >= 0.985:
        a *= 0.95 / max(radius, 1e-9)
        comp[0] = a
        radius = float(np.max(np.abs(np.linalg.eigvals(comp))))
        stabilized = True

    pred = np.zeros_like(e)
    pred[:AR_ORDER] = e[:AR_ORDER]
    for k in range(AR_ORDER, len(e)):
        pred[k] = float(np.dot(a, pred[k-AR_ORDER:k][::-1]))
    rmse_deg = float(np.rad2deg(np.sqrt(np.mean((pred-e)**2))))
    return a, radius, stabilized, rmse_deg, str(path)


def projected_box_qp(H, f, lo, hi, step=None):
    if step is None:
        maximum_eigenvalue = float(np.max(np.linalg.eigvalsh(H)))
        step = 1.0 / max(maximum_eigenvalue, 1e-10)
    u = np.zeros_like(f)
    for _ in range(QP_ITERS):
        u = np.clip(u - step * (H @ u + f), lo, hi)
    return u


class FIRQP:
    """
    Dynamic matrix control:
      future error = natural AR forecast + delayed past commands + S*future commands.
    """
    def __init__(self, h, ar, candidate):
        self.h = np.asarray(h); self.M = len(h)
        self.ar = np.asarray(ar); self.p = len(ar)
        self.q = float(candidate["q"]); self.r = float(candidate["r"]); self.slew = float(candidate["slew"])
        self.prev = 0.0
        self.u_hist = deque([0.0]*(self.M-1), maxlen=self.M-1)  # u[k-1], u[k-2], ...
        self.d_hist = deque([0.0]*self.p, maxlen=self.p)
        self.S = self._make_S()
        self.D = np.eye(N)
        for i in range(1, N):
            self.D[i, i-1] = -1.0
        self.H = self.q*(self.S.T@self.S) + self.r*np.eye(N) + self.slew*(self.D.T@self.D) + 1e-10*np.eye(N)
        eigmax = float(np.max(np.linalg.eigvalsh(self.H)))
        self.qp_step = 1.0 / max(eigmax, 1e-10)

    def _make_S(self):
        S = np.zeros((N, N))
        # output rows k+1,...k+N vs controls u[k],...,u[k+N-1]
        for row in range(N):
            future = row + 1
            for col in range(N):
                lag = future - col
                if 0 <= lag < self.M:
                    S[row, col] = self.h[lag]
        return S

    def _natural_future(self):
        state = list(self.d_hist)
        out = []
        for _ in range(N):
            value = float(np.dot(self.ar, np.asarray(state[-self.p:][::-1])))
            state.append(value); out.append(value)
        return np.asarray(out)

    def _past_control_future(self):
        out = []
        old = list(self.u_hist)
        for row in range(N):
            future = row + 1
            val = 0.0
            for index, uold in enumerate(old, start=1):
                lag = future + index
                if lag < self.M:
                    val += self.h[lag]*uold
            out.append(val)
        return np.asarray(out)

    def update(self, error):
        # observed error minus delayed known command response -> natural residual
        known = 0.0
        for index, uold in enumerate(self.u_hist, start=1):
            if index < self.M:
                known += self.h[index]*uold
        self.d_hist.append(float(error-known))

        free = self._natural_future() + self._past_control_future()
        dprev = np.zeros(N); dprev[0] = self.prev
        f = self.q*(self.S.T@free) - self.slew*(self.D.T@dprev)
        amax = math.radians(ALPHA_LIMIT_DEG)
        sequence = projected_box_qp(
            self.H,
            f,
            np.full(N, -amax),
            np.full(N, amax),
            step=self.qp_step,
        )

        slew = math.radians(ALPHA_SLEW_DEG)
        alpha = float(np.clip(sequence[0], self.prev-slew, self.prev+slew))
        alpha = float(np.clip(alpha, -amax, amax))
        self.prev = alpha
        self.u_hist.appendleft(alpha)
        return alpha


def settling_time(t, err_deg):
    for i in np.where(t >= DIST_T)[0]:
        if np.all(np.abs(err_deg[i:]) <= SETTLING_BAND_DEG):
            return float(t[i] - DIST_T)
    return None


def simulate(direction, controller, candidate_name):
    fddp_path, t_ref, q_ref = load_fddp(direction)
    base_path, bt, bp, br = load_baseline(direction)

    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)

    joint_ids = [require_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS]
    qadr = [int(model.jnt_qposadr[i]) for i in joint_ids]
    act_ids = map_actuators(model, joint_ids)
    pitch_joint = require_id(model, mujoco.mjtObj.mjOBJ_JOINT, ROOT_PITCH)
    pitch_qadr = int(model.jnt_qposadr[pitch_joint]); pitch_dadr = int(model.jnt_dofadr[pitch_joint])
    torso = require_id(model, mujoco.mjtObj.mjOBJ_BODY, TORSO)

    mujoco.mj_resetData(model, data)
    for a, q in zip(qadr, q_ref[0]):
        data.qpos[a] = q
    mujoco.mj_forward(model, data)
    h0 = float(data.xpos[torso][2])

    proxy = ActuatorProxy(6, dt); proxy.reset(q_ref[0])
    update_steps = max(1, int(round(TS/dt)))
    delta = np.zeros(6); alpha = 0.0; was_disturbed = False
    duration = float(t_ref[-1]); nsteps = int(round(duration/dt))+1
    rows = []

    for k in range(nsteps):
        t = min(float(data.time), duration)
        if not was_disturbed and t >= DIST_T:
            data.qpos[pitch_qadr] += math.radians(DIST_PITCH_DEG)
            data.qvel[pitch_dadr] += DIST_RATE
            mujoco.mj_forward(model, data)
            was_disturbed = True

        qnom = interp(t, t_ref, q_ref)
        pref = interp(t, bt, bp)
        pmeas = float(data.qpos[pitch_qadr])
        err = pmeas - pref
        qptime = 0.0
        if controller is not None and k % update_steps == 0:
            start = time.perf_counter_ns()
            alpha = controller.update(err)
            qptime = (time.perf_counter_ns()-start)/1e3
            delta = G*alpha
        elif controller is None:
            alpha = 0.0; delta[:] = 0.0

        cmd = qnom + delta
        proxied = proxy.update(cmd)
        data.ctrl[:] = 0.0
        for aid, val in zip(act_ids, proxied):
            data.ctrl[aid] = val
        mujoco.mj_step(model, data)

        tn = min(float(data.time), duration); refn = interp(tn, bt, bp)
        pafter = float(data.qpos[pitch_qadr])
        row = {
            "time_s": float(data.time),
            "pitch_ref_rad": float(refn),
            "pitch_measured_rad": pafter,
            "pitch_error_deg": math.degrees(pafter-refn),
            "pitch_rate_ref_rad_s": float(interp(tn, bt, br)),
            "pitch_rate_measured_rad_s": float(data.qvel[pitch_dadr]),
            "alpha_deg": math.degrees(alpha),
            "qp_solve_time_us": qptime,
            "torso_height_m": float(data.xpos[torso][2]),
            "ncon": int(data.ncon),
        }
        for j, name in enumerate(JOINTS):
            row[f"{name}_delta_rad"] = float(delta[j])
        rows.append(row)

    df = pd.DataFrame(rows)
    times = df.time_s.to_numpy(); err = df.pitch_error_deg.to_numpy(); mask = times >= DIST_T
    qp = df.loc[df.qp_solve_time_us>0, "qp_solve_time_us"]
    corr_cols = [f"{n}_delta_rad" for n in JOINTS]
    metrics = {
        "candidate": candidate_name, "direction": direction,
        "fddp_reference": str(fddp_path), "mujoco_baseline_reference": str(base_path),
        "pitch": {
            "iae_deg_s": float(np.trapezoid(np.abs(err[mask]), times[mask])),
            "peak_abs_error_deg": float(np.max(np.abs(err[mask]))),
            "final_abs_error_deg": float(abs(err[-1])),
            "settling_time_to_0p5deg_s": settling_time(times,err),
        },
        "stability": {
            "initial_torso_height_m": h0,
            "minimum_torso_height_m": float(df.torso_height_m.min()),
            "min_contacts": int(df.ncon.min()),
        },
        "command": {
            "max_abs_joint_correction_deg": float(np.rad2deg(np.abs(df[corr_cols].to_numpy()).max())),
            "median_qp_solve_time_us": None if len(qp)==0 else float(np.median(qp)),
            "max_qp_solve_time_us": None if len(qp)==0 else float(np.max(qp)),
        }
    }
    return df, metrics


def compare(c0, c1):
    m0, m1 = c0["pitch"], c1["pitch"]
    iae_imp = (m0["iae_deg_s"]-m1["iae_deg_s"])/max(m0["iae_deg_s"],1e-12)
    peak_inc = (m1["peak_abs_error_deg"]-m0["peak_abs_error_deg"])/max(m0["peak_abs_error_deg"],1e-12)
    final_inc = m1["final_abs_error_deg"]-m0["final_abs_error_deg"]
    stable = (
        c1["stability"]["min_contacts"] >= MIN_CONTACTS and
        c1["stability"]["minimum_torso_height_m"] >= MIN_HEIGHT_RATIO*c1["stability"]["initial_torso_height_m"] and
        m1["peak_abs_error_deg"] <= MAX_PEAK_ERROR_DEG and
        c1["command"]["max_abs_joint_correction_deg"] <= MAX_JOINT_CORR_DEG
    )
    perf = iae_imp >= MIN_IAE_IMPROVEMENT and peak_inc <= MAX_PEAK_INCREASE and final_inc <= MAX_FINAL_INCREASE_DEG
    return {
        "iae_improvement_ratio": float(iae_imp),
        "peak_error_increase_ratio": float(peak_inc),
        "final_abs_error_increase_deg": float(final_inc),
        "stability_pass": bool(stable),
        "performance_pass": bool(perf),
        "integrity_passed": bool(stable and perf),
    }


def plot_pair(direction, c0, c1, label):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(c0.time_s, c0.pitch_error_deg, label="C0 FDDP reference only")
    ax.plot(c1.time_s, c1.pitch_error_deg, label=f"C1 FIR-QP ({label})")
    ax.axvline(DIST_T, linestyle="--", label="disturbance")
    ax.axhline(SETTLING_BAND_DEG, linestyle=":"); ax.axhline(-SETTLING_BAND_DEG, linestyle=":")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Pitch error vs MuJoCo C0 baseline (deg)")
    ax.set_title(f"C0 vs C1 reference-aligned FIR-QP-MPC — {direction}")
    ax.grid(True); ax.legend(); fig.tight_layout()
    p1 = FIG/f"c1_qp2_{direction}_pitch_comparison.png"; fig.savefig(p1,dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(c1.time_s,c1.alpha_deg,label="alpha")
    for n in JOINTS:
        ax.plot(c1.time_s,np.rad2deg(c1[f"{n}_delta_rad"]),label=f"{n} correction")
    ax.axvline(DIST_T, linestyle="--", label="disturbance")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Command correction (deg)")
    ax.set_title(f"C1 FIR-QP-MPC correction profile — {direction}")
    ax.grid(True); ax.legend(ncol=2,fontsize=8); fig.tight_layout()
    p2 = FIG/f"c1_qp2_{direction}_corrections.png"; fig.savefig(p2,dpi=180); plt.close(fig)
    return str(p1), str(p2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=("forward","backward","both"), default="both")
    args = parser.parse_args()
    dirs = ("forward","backward") if args.direction=="both" else (args.direction,)

    print("===== C1-QP2: REFERENCE-ALIGNED FIR-QP-MPC TUNING SWEEP =====")
    print("MuJoCo baseline reference + FIR response + AR disturbance forecast.")
    print(f"Update {TS:.3f} s; horizon {N*TS:.3f} s; common candidates only.")

    h, ar, ar_meta, c0 = {}, {}, {}, {}
    for d in dirs:
        _, bt, bp, _ = load_baseline(d)
        a, radius, stabilized, rmse, source = fit_ar(d, bt, bp)
        h[d] = load_fir(d); ar[d] = a
        ar_meta[d] = {"coefficients": [float(x) for x in a], "spectral_radius":radius, "stabilized":stabilized, "fit_rmse_deg":rmse, "source":source}
        print(f"{d}: FIR M={len(h[d])}; AR radius={radius:.4f}; AR RMSE={rmse:.4f} deg")
        c0[d] = simulate(d,None,"C0")

    all_results, raw = [], {}
    for cand in CANDIDATES:
        name = cand["name"]; raw[name] = {}; item={"candidate":cand,"directions":{}}
        print(f"\n----- {name} -----")
        for d in dirs:
            control = FIRQP(h[d],ar[d],cand)
            df, met = simulate(d,control,name)
            comp = compare(c0[d][1],met)
            item["directions"][d]={"C0":c0[d][1],"C1":met,"comparison":comp}
            raw[name][d]=df
            print(f"{d}: IAE {100*comp['iae_improvement_ratio']:+.2f}% | peak {100*comp['peak_error_increase_ratio']:+.2f}% | final {comp['final_abs_error_increase_deg']:+.4f} deg | pass={comp['integrity_passed']}")
        item["common_parameter_pass"] = all(item["directions"][d]["comparison"]["integrity_passed"] for d in dirs)
        item["mean_iae_improvement_ratio"] = float(np.mean([item["directions"][d]["comparison"]["iae_improvement_ratio"] for d in dirs]))
        all_results.append(item)

    good=[r for r in all_results if r["common_parameter_pass"]]
    selected=max(good if good else all_results,key=lambda x:x["mean_iae_improvement_ratio"])
    status="valid_common_candidate_found" if good else "no_candidate_met_all_acceptance_conditions"
    selname=selected["candidate"]["name"]

    LOG.mkdir(parents=True,exist_ok=True)
    output={"stage":"C1_QP2_reference_aligned_empirical_FIR_QP_MPC_tuning_sweep",
            "purpose":"Evaluate common FIR-QP-MPC candidates after MuJoCo reference alignment and pulse-response identification.",
            "scope_note":"Fixed-time phase C1 only; C2 event-phase synchronisation is inactive.",
            "model":{"sample_time_s":TS,"horizon_s":N*TS,"alpha_limit_deg":ALPHA_LIMIT_DEG,"alpha_slew_deg_per_update":ALPHA_SLEW_DEG,"G_rad_per_rad":[float(x) for x in G],"AR_order":AR_ORDER,
                     "directions":{d:{"FIR_coefficients_rad_per_rad":[float(x) for x in h[d]],"AR":ar_meta[d]} for d in dirs}},
            "disturbance":{"time_s":DIST_T,"root_pitch_deg":DIST_PITCH_DEG,"root_pitch_rate_rad_s":DIST_RATE},
            "acceptance":{"min_contacts":MIN_CONTACTS,"min_IAE_improvement_ratio":MIN_IAE_IMPROVEMENT,"max_peak_increase_ratio":MAX_PEAK_INCREASE,"max_final_increase_deg":MAX_FINAL_INCREASE_DEG,"max_joint_correction_deg":MAX_JOINT_CORR_DEG},
            "candidate_results":all_results,
            "selection":{"status":status,"selected_candidate":selname,"common_parameter_pass":selected["common_parameter_pass"],"mean_iae_improvement_ratio":selected["mean_iae_improvement_ratio"]},
           }
    paths={}
    for d in dirs:
        c0path=LOG/f"c1_qp2_c0_{d}.csv"; c1path=LOG/f"c1_qp2_{selname}_{d}.csv"
        c0[d][0].to_csv(c0path,index=False); raw[selname][d].to_csv(c1path,index=False)
        p1,p2=plot_pair(d,c0[d][0],raw[selname][d],selname)
        paths[d]={"c0_csv":str(c0path),"selected_c1_csv":str(c1path),"pitch_plot":p1,"correction_plot":p2}
    output["selected_outputs"]=paths
    summary=LOG/"c1_qp2_fir_mpc_tuning_summary.yaml"
    summary.write_text(yaml.safe_dump(output,sort_keys=False),encoding="utf-8")

    table=[]
    for r in all_results:
        for d in dirs:
            comp=r["directions"][d]["comparison"]; met=r["directions"][d]["C1"]
            table.append({"candidate":r["candidate"]["name"],"direction":d,"common_pass":r["common_parameter_pass"],
                          "iae_improvement_ratio":comp["iae_improvement_ratio"],"peak_error_increase_ratio":comp["peak_error_increase_ratio"],
                          "final_abs_error_increase_deg":comp["final_abs_error_increase_deg"],"integrity_passed":comp["integrity_passed"],
                          "median_qp_solve_us":met["command"]["median_qp_solve_time_us"],"max_joint_correction_deg":met["command"]["max_abs_joint_correction_deg"]})
    pd.DataFrame(table).to_csv(LOG/"c1_qp2_fir_mpc_tuning_table.csv",index=False)

    print(f"\nSelected candidate: {selname} ({status})")
    print("C1-QP2 FIR-MPC VALIDATION PASSED" if selected["common_parameter_pass"] else "C1-QP2 TUNING COMPLETE — NO COMMON PASSING CANDIDATE")
    print(f"Saved summary:\n  {summary}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nC1-QP2 FIR-MPC FAILED: {exc}",file=sys.stderr)
        sys.exit(1)
