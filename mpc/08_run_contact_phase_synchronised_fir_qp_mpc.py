#!/usr/bin/env python3
"""
C2 - contact-event phase-synchronised FIR-QP-MPC validation.

This stage preserves the successful C1 fixed-time FIR-QP-MPC controller and
adds a bounded contact-event phase synchronisation layer around the nominal
reference clock. Toe/heel touch sensors are monitored online. When a future
reference contains left/right contact transitions, observed touchdown/liftoff
events can correct the reference-time offset. For the current double-support
CoM-shift references, no contact transition is scheduled, so the C2 layer is
expected to remain inactive while still logging contact signals and phase state.

Scope note:
  - This is simulation-only.
  - FDDP remains the offline nominal trajectory generator.
  - The hardware-facing command concept remains q_cmd = q_ref + delta_q.
  - tau_ref is not used as a servo command.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
C1_PATH = ROOT / "mpc" / "07_run_reference_aligned_fir_qp_mpc_sweep.py"
XML = ROOT / "model" / "biped_planar.xml"
TRAJ = ROOT / "transfer" / "trajectories"
LOG = ROOT / "results" / "logs"
FIG = ROOT / "results" / "figures"

TOUCH_SENSOR_NAMES = {
    "left_toe": "left_toe_touch",
    "left_heel": "left_heel_touch",
    "right_toe": "right_toe_touch",
    "right_heel": "right_heel_touch",
}
TOUCH_THRESHOLD_N = 0.25
DEBOUNCE_S = 0.010
MATCH_WINDOW_S = 0.120
MAX_PHASE_OFFSET_S = 0.060
MAX_PHASE_STEP_S = 0.020
PHASE_CORRECTION_GAIN = 0.75


def load_c1_module():
    if not C1_PATH.exists():
        raise FileNotFoundError(C1_PATH)
    spec = importlib.util.spec_from_file_location("c1_fir_qp", C1_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


C1 = load_c1_module()


def require_sensor(model, name):
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        available = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(model.nsensor)]
        raise RuntimeError(f"Required C2 touch sensor not found: {name}; available={available}")
    return int(sensor_id)


def sensor_scalar(model, data, sensor_id):
    adr = int(model.sensor_adr[sensor_id])
    dim = int(model.sensor_dim[sensor_id])
    values = np.asarray(data.sensordata[adr:adr + dim], dtype=float)
    return float(values[0]) if values.size else float("nan")


def bool_array(values):
    arr = np.asarray(values)
    if arr.dtype.kind in ("U", "S", "O"):
        return np.asarray([str(x).lower() in ("1", "true", "yes", "active") for x in arr], dtype=bool)
    return arr.astype(float) > 0.5


def load_fddp_full(direction):
    path = TRAJ / f"com_shift_{direction}_ref.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as z:
        t_ref = np.asarray(z["t_ref"], dtype=float)
        q_ref = np.asarray(z["q_ref"], dtype=float)
        phase = np.asarray(z["phase"]).astype(str) if "phase" in z.files else np.asarray(["unknown"] * len(t_ref))
        left = bool_array(z["left_contact_active"]) if "left_contact_active" in z.files else np.ones_like(t_ref, dtype=bool)
        right = bool_array(z["right_contact_active"]) if "right_contact_active" in z.files else np.ones_like(t_ref, dtype=bool)
        contact_mode = np.asarray(z["contact_mode"]).astype(str) if "contact_mode" in z.files else phase.copy()
    if q_ref.shape != (len(t_ref), 6):
        raise RuntimeError(f"{path}: q_ref shape {q_ref.shape}; expected ({len(t_ref)}, 6)")
    return {
        "path": str(path),
        "t_ref": t_ref,
        "q_ref": q_ref,
        "phase": phase,
        "left_contact_active": left,
        "right_contact_active": right,
        "contact_mode": contact_mode,
    }


def build_reference_contact_events(reference):
    t_ref = reference["t_ref"]
    events = []
    for leg, key in (("left", "left_contact_active"), ("right", "right_contact_active")):
        active = np.asarray(reference[key], dtype=bool)
        for idx in range(1, len(active)):
            if active[idx] == active[idx - 1]:
                continue
            kind = f"{leg}_{'touchdown' if active[idx] else 'liftoff'}"
            events.append({"time_s": float(t_ref[idx]), "kind": kind, "source": key, "index": int(idx)})
    phase = np.asarray(reference["phase"]).astype(str)
    for idx in range(1, len(phase)):
        if phase[idx] != phase[idx - 1]:
            events.append({
                "time_s": float(t_ref[idx]),
                "kind": "phase_change",
                "source": "phase",
                "from": str(phase[idx - 1]),
                "to": str(phase[idx]),
                "index": int(idx),
            })
    return sorted(events, key=lambda item: item["time_s"])


class ContactObserver:
    def __init__(self, model, data, dt):
        self.sensor_ids = {logical: require_sensor(model, sensor) for logical, sensor in TOUCH_SENSOR_NAMES.items()}
        self.debounce_steps = max(1, int(round(DEBOUNCE_S / dt)))
        raw = self.sample(model, data)
        self.site_active = {name: raw[f"{name}_force_N"] > TOUCH_THRESHOLD_N for name in TOUCH_SENSOR_NAMES}
        self.leg_active = self._leg_state(self.site_active)
        self.pending_counts = {"left": 0, "right": 0}

    def sample(self, model, data):
        return {f"{logical}_force_N": sensor_scalar(model, data, sid) for logical, sid in self.sensor_ids.items()}

    @staticmethod
    def _leg_state(site_active):
        return {
            "left": bool(site_active["left_toe"] or site_active["left_heel"]),
            "right": bool(site_active["right_toe"] or site_active["right_heel"]),
        }

    def update(self, model, data, t):
        values = self.sample(model, data)
        raw_site = {name: values[f"{name}_force_N"] > TOUCH_THRESHOLD_N for name in TOUCH_SENSOR_NAMES}
        raw_leg = self._leg_state(raw_site)
        events = []
        for leg in ("left", "right"):
            if raw_leg[leg] == self.leg_active[leg]:
                self.pending_counts[leg] = 0
                continue
            self.pending_counts[leg] += 1
            if self.pending_counts[leg] >= self.debounce_steps:
                self.leg_active[leg] = raw_leg[leg]
                self.pending_counts[leg] = 0
                events.append({"time_s": float(t), "kind": f"{leg}_{'touchdown' if raw_leg[leg] else 'liftoff'}"})
        self.site_active = raw_site
        return values, raw_site, self.leg_active.copy(), events


class ContactPhaseSynchronizer:
    def __init__(
        self, reference_events, predictive=False,
        max_offset_s=MAX_PHASE_OFFSET_S, max_step_s=MAX_PHASE_STEP_S,
        match_window_s=MATCH_WINDOW_S,
        correction_gain=PHASE_CORRECTION_GAIN,
    ):
        self.reference_events = [event for event in reference_events if event["kind"] != "phase_change"]
        self.predictive = bool(predictive)
        self.max_offset_s = float(max_offset_s)
        self.max_step_s = float(max_step_s)
        self.correction_gain = float(correction_gain)
        self.match_window_s = float(match_window_s)
        if not np.isfinite(self.match_window_s) or self.match_window_s <= 0.0:
            raise ValueError("match_window_s must be finite and positive")
        self.event_offsets = {}
        self.next_index = 0
        self.offset_s = 0.0
        self.matched = []
        self.skipped = []

    @property
    def active(self):
        return len(self.reference_events) > 0

    def _offset_for_expected(self, expected):
        if self.predictive:
            return float(self.event_offsets.get(expected["kind"], 0.0))
        return self.offset_s

    def reference_time(self, wall_time, duration):
        if self.predictive and self.next_index < len(self.reference_events):
            self.offset_s = self._offset_for_expected(
                self.reference_events[self.next_index]
            )
        return float(np.clip(wall_time + self.offset_s, 0.0, duration))

    def update(self, wall_time, observed_events):
        records = []
        if not self.reference_events:
            for event in observed_events:
                records.append({"status": "unmatched_no_reference_events", **event})
            return records

        while self.next_index < len(self.reference_events):
            expected = self.reference_events[self.next_index]
            expected_wall_time = float(expected["time_s"])
            if not self.predictive:
                expected_wall_time -= self._offset_for_expected(expected)
            if wall_time <= expected_wall_time + self.match_window_s:
                break
            skipped = {"status": "skipped", "wall_time_s": float(wall_time), "reference_event": expected}
            self.skipped.append(skipped)
            records.append(skipped)
            self.next_index += 1

        for observed in observed_events:
            if self.next_index >= len(self.reference_events):
                records.append({"status": "unmatched_no_remaining_reference_events", **observed})
                continue
            expected = self.reference_events[self.next_index]
            expected_wall_time = float(expected["time_s"])
            if not self.predictive:
                expected_wall_time -= self._offset_for_expected(expected)
            timing_error = float(observed["time_s"] - expected_wall_time)
            if observed["kind"] == expected["kind"] and abs(timing_error) <= self.match_window_s:
                if self.predictive:
                    old_offset = self._offset_for_expected(expected)
                    target_offset = float(np.clip(
                        timing_error, -self.max_offset_s, self.max_offset_s
                    ))
                else:
                    old_offset = self.offset_s
                    target_offset = float(np.clip(expected["time_s"] - observed["time_s"], -self.max_offset_s, self.max_offset_s))
                desired = old_offset + self.correction_gain * (target_offset - old_offset)
                step = float(np.clip(desired - old_offset, -self.max_step_s, self.max_step_s))
                new_offset = float(np.clip(old_offset + step, -self.max_offset_s, self.max_offset_s))
                if self.predictive:
                    self.event_offsets[expected["kind"]] = new_offset
                else:
                    self.offset_s = new_offset
                matched = {
                    "status": "matched",
                    "observed_event": observed,
                    "reference_event": expected,
                    "timing_error_s": timing_error,
                    "old_offset_s": old_offset,
                    "new_offset_s": new_offset,
                }
                self.matched.append(matched)
                records.append(matched)
                self.next_index += 1
                if self.predictive:
                    if self.next_index < len(self.reference_events):
                        self.offset_s = self._offset_for_expected(
                            self.reference_events[self.next_index]
                        )
                    else:
                        self.offset_s = 0.0
            else:
                records.append({"status": "unmatched_kind_or_window", "observed_event": observed, "reference_event": expected, "timing_error_s": timing_error})
        return records


def selected_candidate():
    summary = LOG / "c1_qp2_fir_mpc_tuning_summary.yaml"
    if summary.exists():
        report = yaml.safe_load(summary.read_text(encoding="utf-8"))
        selected = report.get("selection", {}).get("selected_candidate", "balanced")
        for item in report.get("candidate_results", []):
            cand = item.get("candidate", {})
            if cand.get("name") == selected:
                return cand
    for cand in C1.CANDIDATES:
        if cand["name"] == "balanced":
            return cand
    return C1.CANDIDATES[0]


def simulate_c2(direction, controller, candidate_name, ar_meta):
    reference = load_fddp_full(direction)
    t_ref = reference["t_ref"]
    q_ref = reference["q_ref"]
    duration = float(t_ref[-1])
    base_path, bt, bp, br = C1.load_baseline(direction)
    reference_events = build_reference_contact_events(reference)

    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)

    joint_ids = [C1.require_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in C1.JOINTS]
    qadr = [int(model.jnt_qposadr[jid]) for jid in joint_ids]
    act_ids = C1.map_actuators(model, joint_ids)
    pitch_joint = C1.require_id(model, mujoco.mjtObj.mjOBJ_JOINT, C1.ROOT_PITCH)
    pitch_qadr = int(model.jnt_qposadr[pitch_joint])
    pitch_dadr = int(model.jnt_dofadr[pitch_joint])
    torso = C1.require_id(model, mujoco.mjtObj.mjOBJ_BODY, C1.TORSO)

    mujoco.mj_resetData(model, data)
    for addr, q in zip(qadr, q_ref[0]):
        data.qpos[addr] = q
    mujoco.mj_forward(model, data)

    h0 = float(data.xpos[torso][2])
    observer = ContactObserver(model, data, dt)
    phase_sync = ContactPhaseSynchronizer(reference_events)
    proxy = C1.ActuatorProxy(6, dt)
    proxy.reset(q_ref[0])

    update_steps = max(1, int(round(C1.TS / dt)))
    delta = np.zeros(6)
    alpha = 0.0
    was_disturbed = False
    nsteps = int(round(duration / dt)) + 1
    rows = []
    event_records = []

    for k in range(nsteps):
        wall_t = min(float(data.time), duration)
        if not was_disturbed and wall_t >= C1.DIST_T:
            data.qpos[pitch_qadr] += math.radians(C1.DIST_PITCH_DEG)
            data.qvel[pitch_dadr] += C1.DIST_RATE
            mujoco.mj_forward(model, data)
            was_disturbed = True

        contact_values, site_active, leg_active, observed_events = observer.update(model, data, wall_t)
        sync_records = phase_sync.update(wall_t, observed_events)
        event_records.extend(sync_records)
        phase_t = phase_sync.reference_time(wall_t, duration)

        qnom = C1.interp(phase_t, t_ref, q_ref)
        pref = C1.interp(phase_t, bt, bp)
        pmeas = float(data.qpos[pitch_qadr])
        err = pmeas - pref
        qptime = 0.0
        if k % update_steps == 0:
            start = time.perf_counter_ns()
            alpha = controller.update(err)
            qptime = (time.perf_counter_ns() - start) / 1e3
            delta = C1.G * alpha

        cmd = qnom + delta
        proxied = proxy.update(cmd)
        data.ctrl[:] = 0.0
        for aid, val in zip(act_ids, proxied):
            data.ctrl[aid] = val
        mujoco.mj_step(model, data)

        wall_next = min(float(data.time), duration)
        phase_next = phase_sync.reference_time(wall_next, duration)
        refn = C1.interp(phase_next, bt, bp)
        pafter = float(data.qpos[pitch_qadr])
        row = {
            "time_s": float(data.time),
            "wall_time_s": float(data.time),
            "reference_time_s": phase_next,
            "phase_offset_s": float(phase_sync.offset_s),
            "phase_sync_active": bool(phase_sync.active),
            "scheduled_contact_events": int(len(phase_sync.reference_events)),
            "matched_contact_events": int(len(phase_sync.matched)),
            "skipped_contact_events": int(len(phase_sync.skipped)),
            "pitch_ref_rad": float(refn),
            "pitch_measured_rad": pafter,
            "pitch_error_deg": math.degrees(pafter - refn),
            "pitch_rate_ref_rad_s": float(C1.interp(phase_next, bt, br)),
            "pitch_rate_measured_rad_s": float(data.qvel[pitch_dadr]),
            "alpha_deg": math.degrees(alpha),
            "qp_solve_time_us": qptime,
            "torso_height_m": float(data.xpos[torso][2]),
            "ncon": int(data.ncon),
            "left_contact_active": bool(leg_active["left"]),
            "right_contact_active": bool(leg_active["right"]),
        }
        for name, value in contact_values.items():
            row[name] = float(value)
        for name, active in site_active.items():
            row[f"{name}_active"] = bool(active)
        for j, name in enumerate(C1.JOINTS):
            row[f"{name}_delta_rad"] = float(delta[j])
        rows.append(row)

    df = pd.DataFrame(rows)
    times = df["wall_time_s"].to_numpy()
    err = df["pitch_error_deg"].to_numpy()
    mask = times >= C1.DIST_T
    qp = df.loc[df.qp_solve_time_us > 0, "qp_solve_time_us"]
    corr_cols = [f"{name}_delta_rad" for name in C1.JOINTS]
    metrics = {
        "candidate": candidate_name,
        "direction": direction,
        "fddp_reference": reference["path"],
        "mujoco_baseline_reference": str(base_path),
        "pitch": {
            "iae_deg_s": float(np.trapezoid(np.abs(err[mask]), times[mask])),
            "peak_abs_error_deg": float(np.max(np.abs(err[mask]))),
            "final_abs_error_deg": float(abs(err[-1])),
            "settling_time_to_0p5deg_s": C1.settling_time(times, err),
        },
        "stability": {
            "initial_torso_height_m": h0,
            "minimum_torso_height_m": float(df.torso_height_m.min()),
            "min_contacts": int(df.ncon.min()),
        },
        "command": {
            "max_abs_joint_correction_deg": float(np.rad2deg(np.abs(df[corr_cols].to_numpy()).max())),
            "median_qp_solve_time_us": None if len(qp) == 0 else float(np.median(qp)),
            "max_qp_solve_time_us": None if len(qp) == 0 else float(np.max(qp)),
        },
        "phase_sync": {
            "active": bool(phase_sync.active),
            "scheduled_contact_events": int(len(phase_sync.reference_events)),
            "matched_contact_events": int(len(phase_sync.matched)),
            "skipped_contact_events": int(len(phase_sync.skipped)),
            "max_abs_phase_offset_ms": float(1000.0 * np.max(np.abs(df.phase_offset_s.to_numpy()))),
            "reference_events": phase_sync.reference_events,
            "event_records": event_records,
            "configuration": {
                "touch_threshold_N": TOUCH_THRESHOLD_N,
                "debounce_s": DEBOUNCE_S,
                "match_window_s": MATCH_WINDOW_S,
                "max_phase_offset_s": MAX_PHASE_OFFSET_S,
                "max_phase_step_s": MAX_PHASE_STEP_S,
                "correction_gain": PHASE_CORRECTION_GAIN,
            },
        },
        "AR": ar_meta,
    }
    return df, metrics


def compare_c1_c2(c1_metrics, c2_metrics):
    p1 = c1_metrics["pitch"]
    p2 = c2_metrics["pitch"]
    return {
        "iae_change_ratio_vs_C1": float((p2["iae_deg_s"] - p1["iae_deg_s"]) / max(p1["iae_deg_s"], 1e-12)),
        "settling_change_s_vs_C1": None if p1["settling_time_to_0p5deg_s"] is None or p2["settling_time_to_0p5deg_s"] is None else float(p2["settling_time_to_0p5deg_s"] - p1["settling_time_to_0p5deg_s"]),
        "final_abs_error_change_deg_vs_C1": float(p2["final_abs_error_deg"] - p1["final_abs_error_deg"]),
        "max_joint_correction_change_deg_vs_C1": float(c2_metrics["command"]["max_abs_joint_correction_deg"] - c1_metrics["command"]["max_abs_joint_correction_deg"]),
    }


def plot_outputs(direction, c0_df, c1_df, c2_df, label):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(c0_df.time_s, c0_df.pitch_error_deg, label="C0 reference only")
    ax.plot(c1_df.time_s, c1_df.pitch_error_deg, label=f"C1 fixed phase ({label})")
    ax.plot(c2_df.wall_time_s, c2_df.pitch_error_deg, label="C2 contact-phase sync")
    ax.axvline(C1.DIST_T, linestyle="--", label="disturbance")
    ax.axhline(C1.SETTLING_BAND_DEG, linestyle=":")
    ax.axhline(-C1.SETTLING_BAND_DEG, linestyle=":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch error vs phase reference (deg)")
    ax.set_title(f"C0/C1/C2 pitch recovery comparison - {direction}")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    pitch_plot = FIG / f"c2_phase_sync_{direction}_pitch_comparison.png"
    fig.savefig(pitch_plot, dpi=180)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(c2_df.wall_time_s, 1000.0 * c2_df.phase_offset_s, label="phase offset", color="tab:purple")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Phase offset (ms)")
    ax1.grid(True)
    ax2 = ax1.twinx()
    for name in ("left_toe", "left_heel", "right_toe", "right_heel"):
        ax2.plot(c2_df.wall_time_s, c2_df[f"{name}_force_N"], label=f"{name} force", alpha=0.65)
    ax2.set_ylabel("Touch sensor force (N)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, ncol=2, fontsize=8, loc="upper right")
    ax1.set_title(f"C2 contact sensing and phase offset - {direction}")
    fig.tight_layout()
    phase_plot = FIG / f"c2_phase_sync_{direction}_contacts_phase.png"
    fig.savefig(phase_plot, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(c2_df.wall_time_s, c2_df.alpha_deg, label="alpha")
    for name in C1.JOINTS:
        ax.plot(c2_df.wall_time_s, np.rad2deg(c2_df[f"{name}_delta_rad"]), label=f"{name} correction")
    ax.axvline(C1.DIST_T, linestyle="--", label="disturbance")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Command correction (deg)")
    ax.set_title(f"C2 command correction profile - {direction}")
    ax.grid(True)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    correction_plot = FIG / f"c2_phase_sync_{direction}_corrections.png"
    fig.savefig(correction_plot, dpi=180)
    plt.close(fig)
    return str(pitch_plot), str(phase_plot), str(correction_plot)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=("forward", "backward", "both"), default="both")
    args = parser.parse_args()
    dirs = ("forward", "backward") if args.direction == "both" else (args.direction,)
    cand = selected_candidate()
    name = cand["name"]

    print("===== C2: CONTACT-EVENT PHASE-SYNCHRONISED FIR-QP-MPC =====")
    print(f"Controller candidate: {name}")
    print("Toe/heel touch sensors are monitored; phase correction is event-triggered and bounded.")

    LOG.mkdir(parents=True, exist_ok=True)
    outputs = {}
    direction_results = {}
    for direction in dirs:
        print(f"\n----- {direction} -----")
        _, bt, bp, _ = C1.load_baseline(direction)
        ar, radius, stabilized, rmse, source = C1.fit_ar(direction, bt, bp)
        h = C1.load_fir(direction)
        ar_meta = {"coefficients": [float(x) for x in ar], "spectral_radius": float(radius), "stabilized": bool(stabilized), "fit_rmse_deg": float(rmse), "source": source}

        c0_df, c0_metrics = C1.simulate(direction, None, "C0")
        c1_df, c1_metrics = C1.simulate(direction, C1.FIRQP(h, ar, cand), name)
        c2_df, c2_metrics = simulate_c2(direction, C1.FIRQP(h, ar, cand), name, ar_meta)

        c0_vs_c2 = C1.compare(c0_metrics, c2_metrics)
        c1_vs_c2 = compare_c1_c2(c1_metrics, c2_metrics)
        c2_csv = LOG / f"c2_phase_sync_{direction}.csv"
        c2_df.to_csv(c2_csv, index=False)
        pitch_plot, phase_plot, correction_plot = plot_outputs(direction, c0_df, c1_df, c2_df, name)
        outputs[direction] = {"c2_csv": str(c2_csv), "pitch_plot": pitch_plot, "phase_plot": phase_plot, "correction_plot": correction_plot}
        direction_results[direction] = {"C0": c0_metrics, "C1": c1_metrics, "C2": c2_metrics, "C0_vs_C2": c0_vs_c2, "C1_vs_C2": c1_vs_c2}
        print(
            f"scheduled={c2_metrics['phase_sync']['scheduled_contact_events']} "
            f"matched={c2_metrics['phase_sync']['matched_contact_events']} "
            f"max_offset={c2_metrics['phase_sync']['max_abs_phase_offset_ms']:.3f} ms "
            f"IAE(C2)={c2_metrics['pitch']['iae_deg_s']:.4f} deg*s "
            f"C0 improvement={100.0*c0_vs_c2['iae_improvement_ratio']:+.2f}%"
        )

    summary = {
        "stage": "C2_contact_event_phase_synchronised_FIR_QP_MPC",
        "scope_note": "C2 is implemented as an event-capable phase synchronisation layer around the existing C1 FIR-QP-MPC. The current references are double_support only, so no contact transition is scheduled and the layer is expected to remain inactive in this validation.",
        "controller_candidate": cand,
        "disturbance": {"time_s": C1.DIST_T, "root_pitch_deg": C1.DIST_PITCH_DEG, "root_pitch_rate_rad_s": C1.DIST_RATE},
        "phase_sync_configuration": {
            "touch_sensor_names": TOUCH_SENSOR_NAMES,
            "touch_threshold_N": TOUCH_THRESHOLD_N,
            "debounce_s": DEBOUNCE_S,
            "match_window_s": MATCH_WINDOW_S,
            "max_phase_offset_s": MAX_PHASE_OFFSET_S,
            "max_phase_step_s": MAX_PHASE_STEP_S,
            "correction_gain": PHASE_CORRECTION_GAIN,
        },
        "direction_results": direction_results,
        "outputs": outputs,
        "interpretation_guardrails": [
            "Do not claim walking-gait phase synchronisation from this double-support validation.",
            "The C2 layer is ready to use scheduled contact transitions when a walking/contact-switching FDDP reference is added.",
            "For the present CoM-shift references, C2 mainly validates sensor monitoring and compatibility with C1, not a new performance gain.",
        ],
    }
    summary_path = LOG / "c2_contact_phase_sync_summary.yaml"
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")

    table = []
    for direction, result in direction_results.items():
        c2 = result["C2"]
        table.append({
            "direction": direction,
            "candidate": name,
            "scheduled_contact_events": c2["phase_sync"]["scheduled_contact_events"],
            "matched_contact_events": c2["phase_sync"]["matched_contact_events"],
            "max_abs_phase_offset_ms": c2["phase_sync"]["max_abs_phase_offset_ms"],
            "c2_iae_deg_s": c2["pitch"]["iae_deg_s"],
            "c2_settling_time_s": c2["pitch"]["settling_time_to_0p5deg_s"],
            "c0_to_c2_iae_improvement_ratio": result["C0_vs_C2"]["iae_improvement_ratio"],
            "c2_vs_c1_iae_change_ratio": result["C1_vs_C2"]["iae_change_ratio_vs_C1"],
            "max_joint_correction_deg": c2["command"]["max_abs_joint_correction_deg"],
            "median_qp_solve_time_us": c2["command"]["median_qp_solve_time_us"],
            "min_contacts": c2["stability"]["min_contacts"],
        })
    pd.DataFrame(table).to_csv(LOG / "c2_contact_phase_sync_table.csv", index=False)

    print(f"\nSaved summary:\n  {summary_path}")
    print("C2 CONTACT-PHASE SYNCHRONISATION VALIDATION COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nC2 CONTACT-PHASE SYNCHRONISATION FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

