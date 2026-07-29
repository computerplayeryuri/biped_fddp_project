#!/usr/bin/env python3
"""
C1-A — Fixed-time-phase FDDP-MPC one-shot validation.

This is the first C1 validation after the C0 baseline.

What it proves
--------------
At a fixed phase on the exported C0 FDDP trajectory, create a short-horizon
receding-horizon FDDP problem, inject a controlled state disturbance, and
replan a local trajectory under the same 6D flat-foot contact model.

This is a genuine nonlinear MPC subproblem:
  current disturbed state + reference window -> short FDDP replan
  -> next desired q command candidate.

What it does NOT yet prove
--------------------------
It does not read MuJoCo online state and it does not use foot-event phase
resynchronisation. Those are the next C1-B and C2 stages.

The control output used later for the position-controlled ST3215 architecture
is the replanned joint-position state q_plan, NOT tau_plan.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
import time

import numpy as np
import yaml
import matplotlib.pyplot as plt
import pinocchio as pin
import crocoddyl


# ------------------------------ C1-A configuration ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = PROJECT_ROOT / "model" / "biped_v1.urdf"
TRAJECTORY_DIR = PROJECT_ROOT / "transfer" / "trajectories"
LOG_DIR = PROJECT_ROOT / "results" / "logs"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

LEFT_FOOT_FRAME = "left_foot_contact_fixed"
RIGHT_FOOT_FRAME = "right_foot_contact_fixed"

DT = 0.010
HORIZON_NODES = 20                  # 0.20 s prediction horizon
FIXED_PHASE_INDEX = 40              # t = 0.40 s, within CoM shift phase
MAX_ITER = 150

# Established by C0 derivative audit.
JM_INV_JT_DAMPING = 1.0e-8
CONTACT_GAINS_KP_KD = np.array([0.0, 0.0], dtype=float)

# Controlled synthetic state perturbation for this isolated MPC validation.
# Root pitch about +X and root pitch rate about +X.
INJECTED_PITCH_DEG = 2.0
INJECTED_PITCH_RATE_RAD_S = 0.25

# Cost design: locally track C0 nominal state/control and CoM reference.
RUNNING_STATE_WEIGHT = 3.0
RUNNING_CONTROL_WEIGHT = 2.0e-3
TERMINAL_STATE_WEIGHT = 15.0
COM_TRACKING_WEIGHT_Y = 5.0e4
COM_TRACKING_WEIGHT_XZ = 2.0e2

# Validation gates.
MAX_FINAL_GAP_NORM = 1.0e-7
MAX_ROOT_PITCH_ERROR_DEG = 0.5
MAX_JOINT_POSITION_OFFSET_DEG = 5.0
MIN_JOINT_LIMIT_MARGIN_RAD = 1.0e-2


def write_yaml(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")


def compute_com(model: pin.Model, xs: np.ndarray) -> np.ndarray:
    data = model.createData()
    values = np.zeros((xs.shape[0], 3), dtype=float)
    for k, x in enumerate(xs):
        values[k] = np.asarray(
            pin.centerOfMass(model, data, x[:model.nq]),
            dtype=float,
        ).reshape(3)
    return values


def support_references(
    model: pin.Model,
    x_reference: np.ndarray,
    left_frame_id: int,
    right_frame_id: int,
):
    data = model.createData()
    pin.forwardKinematics(model, data, x_reference[:model.nq])
    pin.updateFramePlacements(model, data)
    return (
        data.oMf[left_frame_id].copy(),
        data.oMf[right_frame_id].copy(),
    )


def build_action(
    state,
    actuation,
    x_reference: np.ndarray,
    u_reference: np.ndarray,
    com_target: np.ndarray,
    left_frame_id: int,
    right_frame_id: int,
    left_contact_reference,
    right_contact_reference,
    state_weight: float,
    control_weight: float,
    dt: float,
):
    nu = actuation.nu

    contacts = crocoddyl.ContactModelMultiple(state, nu)
    contacts.addContact(
        LEFT_FOOT_FRAME,
        crocoddyl.ContactModel6D(
            state,
            left_frame_id,
            left_contact_reference,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            nu,
            CONTACT_GAINS_KP_KD,
        ),
    )
    contacts.addContact(
        RIGHT_FOOT_FRAME,
        crocoddyl.ContactModel6D(
            state,
            right_frame_id,
            right_contact_reference,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            nu,
            CONTACT_GAINS_KP_KD,
        ),
    )

    costs = crocoddyl.CostModelSum(state, nu)

    state_residual = crocoddyl.ResidualModelState(state, x_reference, nu)
    costs.addCost(
        "nominal_state_tracking",
        crocoddyl.CostModelResidual(state, state_residual),
        state_weight,
    )

    control_residual = crocoddyl.ResidualModelControl(state, u_reference)
    costs.addCost(
        "nominal_control_tracking",
        crocoddyl.CostModelResidual(state, control_residual),
        control_weight,
    )

    com_residual = crocoddyl.ResidualModelCoMPosition(state, com_target, nu)
    com_activation = crocoddyl.ActivationModelWeightedQuad(
        np.array(
            [
                COM_TRACKING_WEIGHT_XZ,
                COM_TRACKING_WEIGHT_Y,
                COM_TRACKING_WEIGHT_XZ,
            ],
            dtype=float,
        )
    )
    costs.addCost(
        "com_tracking",
        crocoddyl.CostModelResidual(state, com_activation, com_residual),
        1.0,
    )

    differential = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state,
        actuation,
        contacts,
        costs,
        JM_INV_JT_DAMPING,
        False,
    )
    return crocoddyl.IntegratedActionModelEuler(differential, dt)


def total_cost(problem, xs, us) -> float:
    problem.calc(xs, us)
    return float(
        sum(float(data.cost) for data in problem.runningDatas)
        + float(problem.terminalData.cost)
    )


def gap_summary(problem, state, xs, us) -> dict:
    problem.calc(xs, us)
    values = []
    for k, data in enumerate(problem.runningDatas):
        gap = np.asarray(
            state.diff(xs[k + 1], data.xnext),
            dtype=float,
        ).reshape(-1)
        values.append(float(np.linalg.norm(gap)))

    values = np.asarray(values, dtype=float)
    return {
        "max_gap_norm": float(np.max(values)),
        "rms_gap_norm": float(np.sqrt(np.mean(values**2))),
        "first_gap_norm": float(values[0]),
        "last_gap_norm": float(values[-1]),
    }


def joint_limit_summary(model: pin.Model, xs: np.ndarray) -> dict:
    joints = xs[:, 7:model.nq]
    lower = np.asarray(model.lowerPositionLimit[7:model.nq], dtype=float)
    upper = np.asarray(model.upperPositionLimit[7:model.nq], dtype=float)

    lower_margin = joints - lower[None, :]
    upper_margin = upper[None, :] - joints
    min_lower = float(np.min(lower_margin))
    min_upper = float(np.min(upper_margin))
    minimum = float(min(min_lower, min_upper))

    return {
        "minimum_lower_margin_rad": min_lower,
        "minimum_upper_margin_rad": min_upper,
        "minimum_margin_rad": minimum,
        "within_position_limits": bool(
            np.all(lower_margin >= -1e-10)
            and np.all(upper_margin >= -1e-10)
        ),
        "has_safety_margin": bool(minimum >= MIN_JOINT_LIMIT_MARGIN_RAD),
    }


def root_pitch_from_state(model: pin.Model, state_vector: np.ndarray) -> float:
    """Return root pitch about +X from a free-flyer quaternion state."""
    q = state_vector[:model.nq]
    quaternion = pin.Quaternion(q[6], q[3], q[4], q[5])
    rotation = quaternion.matrix()
    return float(np.arctan2(rotation[2, 1], rotation[2, 2]))


def make_disturbed_state(state, x_nominal: np.ndarray) -> np.ndarray:
    """
    Add known local perturbations in StateMultibody tangent coordinates:
    tangent configuration indices:
      0..2 root translation
      3..5 root rotation
      6..11 six joint positions
    tangent velocity indices are offset by nv=12.
    """
    dx = np.zeros(state.ndx, dtype=float)
    dx[3] = np.deg2rad(INJECTED_PITCH_DEG)
    dx[12 + 3] = INJECTED_PITCH_RATE_RAD_S
    return np.asarray(state.integrate(x_nominal, dx), dtype=float)


def plot_result(
    direction: str,
    nominal_xs: np.ndarray,
    mpc_xs: np.ndarray,
    t_window: np.ndarray,
    model: pin.Model,
):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    nominal_joints = np.rad2deg(nominal_xs[:, 7:model.nq])
    mpc_joints = np.rad2deg(mpc_xs[:, 7:model.nq])

    joint_names = [
        "L hip", "L knee", "L ankle",
        "R hip", "R knee", "R ankle",
    ]

    figure, axis = plt.subplots(figsize=(12, 7))
    for index, label in enumerate(joint_names):
        axis.plot(
            t_window,
            nominal_joints[:, index],
            linewidth=1.5,
            label=f"{label} nominal",
        )
        axis.plot(
            t_window,
            mpc_joints[:, index],
            linewidth=1.1,
            linestyle="--",
            label=f"{label} MPC",
        )
    axis.set_xlabel("Prediction time from current phase (s)")
    axis.set_ylabel("Joint angle (deg)")
    axis.set_title(f"C1-A fixed-phase FDDP-MPC replanning — {direction}")
    axis.grid(True)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / f"c1_fixed_phase_mpc_{direction}_joint_plan.png",
        dpi=180,
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C1-A fixed-time-phase FDDP-MPC one-shot validation."
    )
    parser.add_argument(
        "--direction",
        choices=("forward", "backward"),
        default="forward",
        help="C0 reference direction to validate.",
    )
    args = parser.parse_args()
    direction = args.direction

    print("===== C1-A: FIXED-TIME-PHASE FDDP-MPC ONE-SHOT VALIDATION =====")
    print("This is not yet MuJoCo online control and not C2 phase resynchronisation.")
    print(f"Direction: {direction}")
    print(
        f"Prediction horizon: {HORIZON_NODES} x {DT:.3f}s = "
        f"{HORIZON_NODES * DT:.3f}s"
    )
    print(
        "Injected disturbance: "
        f"pitch={INJECTED_PITCH_DEG:+.2f} deg, "
        f"pitch-rate={INJECTED_PITCH_RATE_RAD_S:+.3f} rad/s"
    )
    print(f"JMinvJt damping: {JM_INV_JT_DAMPING:.0e}")

    reference_path = TRAJECTORY_DIR / f"com_shift_{direction}_ref.npz"
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Missing C0 trajectory: {reference_path}\n"
            "Run Stage 7C full continuation first."
        )

    model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    if model.nq != 13 or model.nv != 12 or actuation.nu != 6:
        raise RuntimeError(
            f"Unexpected model dimensions nq={model.nq}, nv={model.nv}, nu={actuation.nu}."
        )

    with np.load(reference_path) as loaded:
        required = ("state_ref", "tau_ref", "t_ref")
        missing = [key for key in required if key not in loaded.files]
        if missing:
            raise RuntimeError(f"{reference_path} missing {missing}; keys: {loaded.files}")
        state_ref = np.asarray(loaded["state_ref"], dtype=float)
        tau_ref = np.asarray(loaded["tau_ref"], dtype=float)
        t_ref = np.asarray(loaded["t_ref"], dtype=float)

    if state_ref.shape != (121, 25):
        raise RuntimeError(f"Unexpected state_ref shape {state_ref.shape}.")
    if tau_ref.shape != (120, 6):
        raise RuntimeError(f"Unexpected tau_ref shape {tau_ref.shape}.")
    if t_ref.shape != (121,):
        raise RuntimeError(f"Unexpected t_ref shape {t_ref.shape}.")
    if FIXED_PHASE_INDEX + HORIZON_NODES >= state_ref.shape[0]:
        raise RuntimeError("Fixed phase plus horizon exceeds reference length.")

    nominal_xs = state_ref[
        FIXED_PHASE_INDEX : FIXED_PHASE_INDEX + HORIZON_NODES + 1
    ].copy()
    nominal_us = tau_ref[
        FIXED_PHASE_INDEX : FIXED_PHASE_INDEX + HORIZON_NODES
    ].copy()
    t_window = t_ref[
        FIXED_PHASE_INDEX : FIXED_PHASE_INDEX + HORIZON_NODES + 1
    ] - t_ref[FIXED_PHASE_INDEX]

    left_frame_id = model.getFrameId(LEFT_FOOT_FRAME)
    right_frame_id = model.getFrameId(RIGHT_FOOT_FRAME)
    left_reference, right_reference = support_references(
        model,
        nominal_xs[0],
        left_frame_id,
        right_frame_id,
    )

    nominal_com = compute_com(model, nominal_xs)

    running_models = []
    for k in range(HORIZON_NODES):
        running_models.append(
            build_action(
                state,
                actuation,
                nominal_xs[k],
                nominal_us[k],
                nominal_com[k],
                left_frame_id,
                right_frame_id,
                left_reference,
                right_reference,
                RUNNING_STATE_WEIGHT,
                RUNNING_CONTROL_WEIGHT,
                DT,
            )
        )

    terminal_model = build_action(
        state,
        actuation,
        nominal_xs[-1],
        np.zeros(actuation.nu, dtype=float),
        nominal_com[-1],
        left_frame_id,
        right_frame_id,
        left_reference,
        right_reference,
        TERMINAL_STATE_WEIGHT,
        0.0,
        0.0,
    )

    x_measured = make_disturbed_state(state, nominal_xs[0])
    problem = crocoddyl.ShootingProblem(x_measured, running_models, terminal_model)

    # Initial guess: retain the nominal window but place the first node at
    # the current measured state. FDDP will close the resulting shooting gaps.
    xs_seed = [x.copy() for x in nominal_xs]
    xs_seed[0] = x_measured.copy()
    us_seed = [u.copy() for u in nominal_us]

    initial_cost = total_cost(problem, xs_seed, us_seed)
    initial_gaps = gap_summary(problem, state, xs_seed, us_seed)

    solver = crocoddyl.SolverFDDP(problem)
    solver.th_stop = 1e-7
    solve_start = time.perf_counter()
    return_success = bool(solver.solve(xs_seed, us_seed, MAX_ITER, False))
    solve_elapsed_ms = (time.perf_counter() - solve_start) * 1000.0
    stop_passed = bool(float(solver.stop) <= float(solver.th_stop))
    accepted = bool(return_success or stop_passed)

    mpc_xs = np.asarray(solver.xs, dtype=float)
    mpc_us = np.asarray(solver.us, dtype=float)
    final_cost = total_cost(
        problem,
        [x.copy() for x in mpc_xs],
        [u.copy() for u in mpc_us],
    )
    final_gaps = gap_summary(
        problem,
        state,
        [x.copy() for x in mpc_xs],
        [u.copy() for u in mpc_us],
    )

    nominal_pitch = root_pitch_from_state(model, nominal_xs[0])
    disturbed_pitch = root_pitch_from_state(model, x_measured)
    predicted_next_pitch = root_pitch_from_state(model, mpc_xs[1])
    nominal_next_pitch = root_pitch_from_state(model, nominal_xs[1])
    next_pitch_error_deg = float(
        np.rad2deg(predicted_next_pitch - nominal_next_pitch)
    )

    nominal_q_next = nominal_xs[1, 7:model.nq]
    q_cmd_next = mpc_xs[1, 7:model.nq]
    delta_q_next = q_cmd_next - nominal_q_next
    max_delta_q_deg = float(np.max(np.abs(np.rad2deg(delta_q_next))))

    limits = joint_limit_summary(model, mpc_xs)
    finite = bool(
        np.all(np.isfinite(mpc_xs))
        and np.all(np.isfinite(mpc_us))
        and np.isfinite(final_cost)
    )

    print()
    print("----- C1-A MPC VALIDATION RESULT -----")
    print(f"Solver return value: {return_success}")
    print(f"Solver stop criterion: {solver.stop:.3e}")
    print(f"Solver accepted: {accepted}")
    print(f"Wall-clock solve time: {solve_elapsed_ms:.2f} ms")
    print(f"Initial / final cost: {initial_cost:.6e} / {final_cost:.6e}")
    print(
        "Initial gaps max/RMS: "
        f"{initial_gaps['max_gap_norm']:.3e} / {initial_gaps['rms_gap_norm']:.3e}"
    )
    print(
        "Final gaps max/RMS: "
        f"{final_gaps['max_gap_norm']:.3e} / {final_gaps['rms_gap_norm']:.3e}"
    )
    print(
        "Root pitch nominal / disturbed / MPC next (deg): "
        f"{np.rad2deg(nominal_pitch):+.3f} / "
        f"{np.rad2deg(disturbed_pitch):+.3f} / "
        f"{np.rad2deg(predicted_next_pitch):+.3f}"
    )
    print(
        "MPC next-step root-pitch deviation from nominal (deg): "
        f"{next_pitch_error_deg:+.4f}"
    )
    print(
        "Maximum next joint-position correction |delta q_cmd| (deg): "
        f"{max_delta_q_deg:.4f}"
    )
    print(
        "Minimum joint-limit margin (rad): "
        f"{limits['minimum_margin_rad']:.6f}"
    )

    failures = []
    if not accepted:
        failures.append("FDDP MPC solve did not meet convergence criterion")
    if not finite:
        failures.append("non-finite state/control/cost")
    if final_gaps["max_gap_norm"] > MAX_FINAL_GAP_NORM:
        failures.append("final shooting gap exceeds threshold")
    if abs(next_pitch_error_deg) > MAX_ROOT_PITCH_ERROR_DEG:
        failures.append("next-step root-pitch deviation from nominal is too large")
    if max_delta_q_deg > MAX_JOINT_POSITION_OFFSET_DEG:
        failures.append("next-step joint-position correction is too large")
    if not limits["within_position_limits"]:
        failures.append("joint position limit violated")
    elif not limits["has_safety_margin"]:
        failures.append("joint-limit safety margin below threshold")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = LOG_DIR / f"c1_fixed_phase_mpc_{direction}_validation.yaml"

    summary = {
        "stage": "C1A_fixed_time_phase_fddp_mpc_one_shot_validation",
        "purpose": (
            "Validate a short-horizon fixed-time-phase nonlinear FDDP-MPC "
            "replan from a controlled synthetic root-pitch disturbance."
        ),
        "scope_note": (
            "This is a solver/reference-window validation. It does not yet "
            "use online MuJoCo state estimation or contact-event phase resynchronisation."
        ),
        "inputs": {
            "urdf": str(URDF_PATH),
            "c0_reference": str(reference_path),
            "direction": direction,
            "phase_index": int(FIXED_PHASE_INDEX),
            "phase_time_s": float(t_ref[FIXED_PHASE_INDEX]),
            "prediction_horizon_nodes": int(HORIZON_NODES),
            "prediction_horizon_s": float(HORIZON_NODES * DT),
        },
        "contact_dynamics": {
            "type": "two 6D flat-foot contacts",
            "JMinvJt_damping": float(JM_INV_JT_DAMPING),
            "baumgarte_gains_Kp_Kd": [float(v) for v in CONTACT_GAINS_KP_KD],
        },
        "injected_disturbance": {
            "root_pitch_deg": float(INJECTED_PITCH_DEG),
            "root_pitch_rate_rad_s": float(INJECTED_PITCH_RATE_RAD_S),
        },
        "solver": {
            "name": "SolverFDDP",
            "max_iterations": int(MAX_ITER),
            "return_success": bool(return_success),
            "stop_criterion": float(solver.stop),
            "threshold": float(solver.th_stop),
            "accepted_convergence": bool(accepted),
            "wall_clock_solve_time_ms": float(solve_elapsed_ms),
            "initial_cost": float(initial_cost),
            "final_cost": float(final_cost),
            "initial_shooting_gaps": initial_gaps,
            "final_shooting_gaps": final_gaps,
        },
        "position_command_candidate": {
            "definition": (
                "q_cmd_next = q_plan[1]; this is the next desired joint-position "
                "target for the position-controlled actuator layer."
            ),
            "q_nominal_next_rad": [float(v) for v in nominal_q_next],
            "q_cmd_next_rad": [float(v) for v in q_cmd_next],
            "delta_q_cmd_rad": [float(v) for v in delta_q_next],
            "max_abs_delta_q_cmd_deg": float(max_delta_q_deg),
        },
        "root_pitch_response": {
            "nominal_current_deg": float(np.rad2deg(nominal_pitch)),
            "disturbed_current_deg": float(np.rad2deg(disturbed_pitch)),
            "mpc_predicted_next_deg": float(np.rad2deg(predicted_next_pitch)),
            "nominal_next_deg": float(np.rad2deg(nominal_next_pitch)),
            "next_deviation_from_nominal_deg": float(next_pitch_error_deg),
        },
        "joint_limits": limits,
        "integrity_passed": bool(not failures),
        "failures": failures,
        "next_step": (
            "C1-B: map MuJoCo measured state into the free-flyer FDDP state, "
            "run the same fixed-phase MPC repeatedly, and command q_cmd_next "
            "through the actuator proxy."
            if not failures
            else
            "Do not proceed to closed-loop MuJoCo MPC until this validation is fixed."
        ),
    }

    write_yaml(summary_path, summary)
    plot_result(direction, nominal_xs, mpc_xs, t_window, model)

    print(f"C1-A MPC integrity: {'PASSED' if not failures else 'FAILED'}")
    for failure in failures:
        print(f"  - {failure}")
    print(f"Saved:\n  {summary_path}")
    print(
        "Saved plot:\n  "
        f"{FIGURE_DIR / f'c1_fixed_phase_mpc_{direction}_joint_plan.png'}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nC1-A FIXED-PHASE MPC VALIDATION FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
