#!/usr/bin/env python3
"""
C1-A v2 — Fixed-phase FDDP-MPC pitch-recovery validation.

The first C1-A run converged but was evaluated incorrectly: it required a
2-degree pitch disturbance with positive pitch rate to disappear within 10 ms.
This script evaluates recovery at the end of a 0.30 s prediction horizon and
assigns explicit state-cost weights to root pitch and root pitch rate.

This remains an offline one-shot MPC formulation validation:
  disturbed state + fixed-time C0 reference window -> short FDDP replan
  -> q_cmd_next = q_plan[1]

It does not yet use online MuJoCo state feedback or C2 contact-event phase sync.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import crocoddyl
import yaml


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "model" / "biped_v1.urdf"
TRAJ_DIR = ROOT / "transfer" / "trajectories"
LOG_DIR = ROOT / "results" / "logs"
FIG_DIR = ROOT / "results" / "figures"

LEFT_FOOT = "left_foot_contact_fixed"
RIGHT_FOOT = "right_foot_contact_fixed"

DT = 0.010
HORIZON = 30              # 0.30 s
PHASE_INDEX = 40          # t=0.40 s in C0
MAX_ITER = 250
DAMPING = 1.0e-8
CONTACT_GAINS = np.array([0.0, 0.0])

INJECTED_PITCH_DEG = 2.0
INJECTED_PITCH_RATE = 0.25  # rad/s

# State tangent vector: [root xyz(3), root rot(3), joints(6),
#                         root linear vel(3), root angular vel(3), joint vel(6)]
# Root pitch about +X = index 3; pitch rate about +X = index 15.
RUN_W = np.array([
    5, 5, 5,                 # root translation
    2500, 20, 20,            # root rotation: pitch is strongly weighted
    10, 10, 10, 10, 10, 10,  # joint positions
    2, 2, 2,                 # root linear velocity
    1200, 10, 10,            # root angular velocity: pitch rate strongly weighted
    1, 1, 1, 1, 1, 1,        # joint velocities
], dtype=float)
TERM_W = np.array([
    30, 30, 30,
    30000, 100, 100,
    100, 100, 100, 100, 100, 100,
    20, 20, 20,
    8000, 50, 50,
    5, 5, 5, 5, 5, 5,
], dtype=float)

R_WEIGHT = 2e-3
COM_W = np.array([2e2, 5e4, 2e2], dtype=float)

# Horizon-level recovery gates.
MAX_GAP = 1e-7
MAX_TERMINAL_PITCH_ERROR_DEG = 0.5
MIN_REDUCTION = 0.70
MAX_TERMINAL_RATE_ERROR = 0.10
MAX_NEXT_Q_CORRECTION_DEG = 5.0
MIN_LIMIT_MARGIN = 1e-2


def dump_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def com_series(model: pin.Model, xs: np.ndarray) -> np.ndarray:
    data = model.createData()
    result = np.zeros((len(xs), 3))
    for k, x in enumerate(xs):
        result[k] = np.asarray(
            pin.centerOfMass(model, data, x[: model.nq]), dtype=float
        ).reshape(3)
    return result


def support_poses(model: pin.Model, x: np.ndarray, lf: int, rf: int):
    data = model.createData()
    pin.forwardKinematics(model, data, x[: model.nq])
    pin.updateFramePlacements(model, data)
    return data.oMf[lf].copy(), data.oMf[rf].copy()


def pitch(model: pin.Model, x: np.ndarray) -> float:
    q = x[: model.nq]  # Pinocchio free-flyer quaternion config is [x,y,z,w]
    quat = pin.Quaternion(q[6], q[3], q[4], q[5])
    R = quat.matrix()
    return float(np.arctan2(R[2, 1], R[2, 2]))


def pitch_rate(model: pin.Model, x: np.ndarray) -> float:
    return float(x[model.nq + 3])


def make_action(
    state,
    actuation,
    x_ref: np.ndarray,
    u_ref: np.ndarray,
    com_ref: np.ndarray,
    w_state: np.ndarray,
    r_weight: float,
    lf: int,
    rf: int,
    left_pose,
    right_pose,
    dt: float,
):
    nu = actuation.nu
    contacts = crocoddyl.ContactModelMultiple(state, nu)
    contacts.addContact(
        LEFT_FOOT,
        crocoddyl.ContactModel6D(
            state, lf, left_pose, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            nu, CONTACT_GAINS,
        ),
    )
    contacts.addContact(
        RIGHT_FOOT,
        crocoddyl.ContactModel6D(
            state, rf, right_pose, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            nu, CONTACT_GAINS,
        ),
    )
    costs = crocoddyl.CostModelSum(state, nu)

    x_res = crocoddyl.ResidualModelState(state, x_ref, nu)
    x_act = crocoddyl.ActivationModelWeightedQuad(w_state)
    costs.addCost(
        "weighted_state_tracking",
        crocoddyl.CostModelResidual(state, x_act, x_res),
        1.0,
    )

    if r_weight > 0:
        u_res = crocoddyl.ResidualModelControl(state, u_ref)
        costs.addCost(
            "nominal_control_tracking",
            crocoddyl.CostModelResidual(state, u_res),
            r_weight,
        )

    c_res = crocoddyl.ResidualModelCoMPosition(state, com_ref, nu)
    c_act = crocoddyl.ActivationModelWeightedQuad(COM_W)
    costs.addCost(
        "com_tracking",
        crocoddyl.CostModelResidual(state, c_act, c_res),
        1.0,
    )

    dam = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contacts, costs, DAMPING, False
    )
    return crocoddyl.IntegratedActionModelEuler(dam, dt)


def total_cost(problem, xs, us) -> float:
    problem.calc(xs, us)
    return float(sum(float(d.cost) for d in problem.runningDatas) + float(problem.terminalData.cost))


def gap_stats(problem, state, xs, us) -> dict:
    problem.calc(xs, us)
    g = []
    for k, d in enumerate(problem.runningDatas):
        gap = np.asarray(state.diff(xs[k + 1], d.xnext), dtype=float).reshape(-1)
        g.append(float(np.linalg.norm(gap)))
    g = np.asarray(g)
    return {
        "max_gap_norm": float(g.max()),
        "rms_gap_norm": float(np.sqrt(np.mean(g * g))),
        "first_gap_norm": float(g[0]),
        "last_gap_norm": float(g[-1]),
    }


def joint_limits(model: pin.Model, xs: np.ndarray) -> dict:
    q = xs[:, 7:model.nq]
    low = np.asarray(model.lowerPositionLimit[7:model.nq], dtype=float)
    up = np.asarray(model.upperPositionLimit[7:model.nq], dtype=float)
    low_m = q - low
    up_m = up - q
    margin = float(min(np.min(low_m), np.min(up_m)))
    return {
        "minimum_lower_margin_rad": float(np.min(low_m)),
        "minimum_upper_margin_rad": float(np.min(up_m)),
        "minimum_margin_rad": margin,
        "within_position_limits": bool(np.all(low_m >= -1e-10) and np.all(up_m >= -1e-10)),
        "has_safety_margin": bool(margin >= MIN_LIMIT_MARGIN),
    }


def disturbed_state(state, x_nom: np.ndarray) -> np.ndarray:
    dx = np.zeros(state.ndx)
    dx[3] = np.deg2rad(INJECTED_PITCH_DEG)
    dx[15] = INJECTED_PITCH_RATE
    return np.asarray(state.integrate(x_nom, dx), dtype=float)


def plot_recovery(direction: str, t: np.ndarray, nominal: np.ndarray, planned: np.ndarray, model: pin.Model):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p_nom = np.rad2deg([pitch(model, x) for x in nominal])
    p_plan = np.rad2deg([pitch(model, x) for x in planned])
    r_nom = [pitch_rate(model, x) for x in nominal]
    r_plan = [pitch_rate(model, x) for x in planned]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(t, p_nom, label="C0 nominal root pitch", linewidth=1.8)
    ax.plot(t, p_plan, label="C1 MPC predicted root pitch", linewidth=1.8)
    ax.scatter([t[0]], [p_plan[0]], marker="x", s=70, label="disturbed current state")
    ax.set_xlabel("Prediction time from current phase (s)")
    ax.set_ylabel("Root pitch about X (deg)")
    ax.set_title(f"C1-A v2 fixed-phase MPC pitch recovery — {direction}")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"c1_fixed_phase_mpc_{direction}_pitch_recovery.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(t, r_nom, label="C0 nominal pitch rate", linewidth=1.8)
    ax.plot(t, r_plan, label="C1 MPC predicted pitch rate", linewidth=1.8)
    ax.scatter([t[0]], [r_plan[0]], marker="x", s=70, label="disturbed current rate")
    ax.set_xlabel("Prediction time from current phase (s)")
    ax.set_ylabel("Root pitch rate about X (rad/s)")
    ax.set_title(f"C1-A v2 fixed-phase MPC pitch-rate recovery — {direction}")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"c1_fixed_phase_mpc_{direction}_pitch_rate_recovery.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=("forward", "backward"), default="forward")
    direction = parser.parse_args().direction

    print("===== C1-A v2: FIXED-PHASE FDDP-MPC PITCH-RECOVERY VALIDATION =====")
    print("Recovery is assessed at the 0.30-s horizon endpoint, not at 10 ms.")
    print(f"Direction: {direction}")
    print(f"Horizon: {HORIZON} x {DT:.3f}s = {HORIZON * DT:.3f}s")
    print(f"Injected disturbance: pitch={INJECTED_PITCH_DEG:+.2f} deg, pitch-rate={INJECTED_PITCH_RATE:+.3f} rad/s")
    print(f"JMinvJt damping: {DAMPING:.0e}")

    ref_path = TRAJ_DIR / f"com_shift_{direction}_ref.npz"
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing C0 reference: {ref_path}")

    model = pin.buildModelFromUrdf(str(URDF), pin.JointModelFreeFlyer())
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    if (model.nq, model.nv, state.ndx, actuation.nu) != (13, 12, 24, 6):
        raise RuntimeError(f"Unexpected dimensions nq={model.nq}, nv={model.nv}, ndx={state.ndx}, nu={actuation.nu}")

    with np.load(ref_path) as z:
        xs_all = np.asarray(z["state_ref"], dtype=float)
        us_all = np.asarray(z["tau_ref"], dtype=float)
        t_all = np.asarray(z["t_ref"], dtype=float)

    if PHASE_INDEX + HORIZON >= len(xs_all):
        raise RuntimeError("Reference window exceeds trajectory.")
    nominal_xs = xs_all[PHASE_INDEX:PHASE_INDEX + HORIZON + 1].copy()
    nominal_us = us_all[PHASE_INDEX:PHASE_INDEX + HORIZON].copy()
    t = t_all[PHASE_INDEX:PHASE_INDEX + HORIZON + 1] - t_all[PHASE_INDEX]
    nominal_com = com_series(model, nominal_xs)

    lf = model.getFrameId(LEFT_FOOT)
    rf = model.getFrameId(RIGHT_FOOT)
    left_pose, right_pose = support_poses(model, nominal_xs[0], lf, rf)

    running = [
        make_action(
            state, actuation, nominal_xs[k], nominal_us[k], nominal_com[k],
            RUN_W, R_WEIGHT, lf, rf, left_pose, right_pose, DT
        )
        for k in range(HORIZON)
    ]
    terminal = make_action(
        state, actuation, nominal_xs[-1], np.zeros(6), nominal_com[-1],
        TERM_W, 0.0, lf, rf, left_pose, right_pose, 0.0
    )

    x_meas = disturbed_state(state, nominal_xs[0])
    problem = crocoddyl.ShootingProblem(x_meas, running, terminal)
    xs_seed = [x.copy() for x in nominal_xs]
    xs_seed[0] = x_meas.copy()
    us_seed = [u.copy() for u in nominal_us]

    initial_cost = total_cost(problem, xs_seed, us_seed)
    initial_gaps = gap_stats(problem, state, xs_seed, us_seed)

    solver = crocoddyl.SolverFDDP(problem)
    solver.th_stop = 1e-7
    started = time.perf_counter()
    return_success = bool(solver.solve(xs_seed, us_seed, MAX_ITER, False))
    solve_ms = (time.perf_counter() - started) * 1000.0
    stop_passed = bool(float(solver.stop) <= float(solver.th_stop))
    accepted = bool(return_success or stop_passed)

    xs = np.asarray(solver.xs, dtype=float)
    us = np.asarray(solver.us, dtype=float)
    final_cost = total_cost(problem, [x.copy() for x in xs], [u.copy() for u in us])
    final_gaps = gap_stats(problem, state, [x.copy() for x in xs], [u.copy() for u in us])

    p_nom_0 = pitch(model, nominal_xs[0])
    p_dist_0 = pitch(model, x_meas)
    p_nom_T = pitch(model, nominal_xs[-1])
    p_mpc_T = pitch(model, xs[-1])
    rate_nom_T = pitch_rate(model, nominal_xs[-1])
    rate_mpc_T = pitch_rate(model, xs[-1])

    initial_pitch_error_deg = float(np.rad2deg(p_dist_0 - p_nom_0))
    terminal_pitch_error_deg = float(np.rad2deg(p_mpc_T - p_nom_T))
    reduction = float(1.0 - abs(terminal_pitch_error_deg) / max(abs(initial_pitch_error_deg), 1e-12))
    terminal_rate_error = float(rate_mpc_T - rate_nom_T)

    q_nom_next = nominal_xs[1, 7:model.nq]
    q_cmd_next = xs[1, 7:model.nq]
    delta_q = q_cmd_next - q_nom_next
    max_delta_q_deg = float(np.max(np.abs(np.rad2deg(delta_q))))
    limits = joint_limits(model, xs)

    print("\n----- C1-A v2 MPC PITCH-RECOVERY RESULT -----")
    print(f"Solver return value: {return_success}")
    print(f"Solver stop criterion: {solver.stop:.3e}")
    print(f"Solver accepted: {accepted}")
    print(f"Wall-clock solve time: {solve_ms:.2f} ms")
    print(f"Initial / final cost: {initial_cost:.6e} / {final_cost:.6e}")
    print(f"Initial / final max shooting gap: {initial_gaps['max_gap_norm']:.3e} / {final_gaps['max_gap_norm']:.3e}")
    print(f"Pitch error current -> terminal (deg): {initial_pitch_error_deg:+.4f} -> {terminal_pitch_error_deg:+.4f}")
    print(f"Pitch-error reduction over horizon: {100.0 * reduction:.2f}%")
    print(f"Terminal pitch-rate error (rad/s): {terminal_rate_error:+.6f}")
    print(f"Maximum first-command correction |delta q_cmd| (deg): {max_delta_q_deg:.4f}")
    print(f"Minimum joint-limit margin (rad): {limits['minimum_margin_rad']:.6f}")

    failures = []
    finite = bool(np.all(np.isfinite(xs)) and np.all(np.isfinite(us)) and np.isfinite(final_cost))
    if not accepted:
        failures.append("FDDP MPC solve did not meet convergence criterion")
    if not finite:
        failures.append("non-finite state/control/cost")
    if final_gaps["max_gap_norm"] > MAX_GAP:
        failures.append("final shooting gap exceeds threshold")
    if abs(terminal_pitch_error_deg) > MAX_TERMINAL_PITCH_ERROR_DEG:
        failures.append("terminal root-pitch error exceeds threshold")
    if reduction < MIN_REDUCTION:
        failures.append("root-pitch error reduction over horizon is insufficient")
    if abs(terminal_rate_error) > MAX_TERMINAL_RATE_ERROR:
        failures.append("terminal root pitch-rate error exceeds threshold")
    if max_delta_q_deg > MAX_NEXT_Q_CORRECTION_DEG:
        failures.append("first joint-position correction is too large")
    if not limits["within_position_limits"]:
        failures.append("joint position limit violated")
    elif not limits["has_safety_margin"]:
        failures.append("joint-limit safety margin below threshold")

    output = {
        "stage": "C1A_v2_fixed_time_phase_fddp_mpc_pitch_recovery",
        "purpose": "Validate horizon-level root-pitch recovery using fixed-time-phase nonlinear FDDP-MPC.",
        "scope_note": "One-shot planner validation only; not yet repeated online MuJoCo MPC or C2 event phase synchronisation.",
        "inputs": {
            "urdf": str(URDF),
            "c0_reference": str(ref_path),
            "direction": direction,
            "phase_index": PHASE_INDEX,
            "phase_time_s": float(t_all[PHASE_INDEX]),
            "prediction_horizon_nodes": HORIZON,
            "prediction_horizon_s": HORIZON * DT,
        },
        "contact_dynamics": {
            "type": "two 6D flat-foot contacts",
            "JMinvJt_damping": DAMPING,
            "baumgarte_gains_Kp_Kd": [float(v) for v in CONTACT_GAINS],
        },
        "injected_disturbance": {
            "root_pitch_deg": INJECTED_PITCH_DEG,
            "root_pitch_rate_rad_s": INJECTED_PITCH_RATE,
        },
        "state_activation_weights": {
            "running": [float(v) for v in RUN_W],
            "terminal": [float(v) for v in TERM_W],
            "root_pitch_tangent_index": 3,
            "root_pitch_rate_tangent_index": 15,
        },
        "solver": {
            "name": "SolverFDDP",
            "max_iterations": MAX_ITER,
            "return_success": return_success,
            "stop_criterion": float(solver.stop),
            "threshold": float(solver.th_stop),
            "accepted_convergence": accepted,
            "wall_clock_solve_time_ms": solve_ms,
            "initial_cost": initial_cost,
            "final_cost": final_cost,
            "initial_shooting_gaps": initial_gaps,
            "final_shooting_gaps": final_gaps,
        },
        "recovery_metrics": {
            "nominal_current_root_pitch_deg": float(np.rad2deg(p_nom_0)),
            "disturbed_current_root_pitch_deg": float(np.rad2deg(p_dist_0)),
            "initial_pitch_error_deg": initial_pitch_error_deg,
            "nominal_terminal_root_pitch_deg": float(np.rad2deg(p_nom_T)),
            "mpc_terminal_root_pitch_deg": float(np.rad2deg(p_mpc_T)),
            "terminal_pitch_error_deg": terminal_pitch_error_deg,
            "pitch_error_reduction_ratio": reduction,
            "nominal_terminal_pitch_rate_rad_s": float(rate_nom_T),
            "mpc_terminal_pitch_rate_rad_s": float(rate_mpc_T),
            "terminal_pitch_rate_error_rad_s": terminal_rate_error,
        },
        "position_command_candidate": {
            "definition": "q_cmd_next = q_plan[1], the next desired position command for the actuator layer.",
            "q_nominal_next_rad": [float(v) for v in q_nom_next],
            "q_cmd_next_rad": [float(v) for v in q_cmd_next],
            "delta_q_cmd_rad": [float(v) for v in delta_q],
            "max_abs_delta_q_cmd_deg": max_delta_q_deg,
        },
        "joint_limits": limits,
        "integrity_passed": bool(not failures),
        "failures": failures,
        "next_step": (
            "C1-B: implement repeated MuJoCo closed-loop MPC with a MuJoCo-to-Pinocchio state mapper."
            if not failures else
            "Inspect terminal recovery metrics and retune before C1-B."
        ),
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    dump_yaml(LOG_DIR / f"c1_fixed_phase_mpc_{direction}_pitch_recovery.yaml", output)
    plot_recovery(direction, t, nominal_xs, xs, model)

    print(f"C1-A v2 MPC integrity: {'PASSED' if not failures else 'FAILED'}")
    for failure in failures:
        print(f"  - {failure}")
    print(f"Saved:\n  {LOG_DIR / f'c1_fixed_phase_mpc_{direction}_pitch_recovery.yaml'}")
    print(f"Plots:\n  {FIG_DIR / f'c1_fixed_phase_mpc_{direction}_pitch_recovery.png'}\n  {FIG_DIR / f'c1_fixed_phase_mpc_{direction}_pitch_rate_recovery.png'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nC1-A v2 PITCH-RECOVERY VALIDATION FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
