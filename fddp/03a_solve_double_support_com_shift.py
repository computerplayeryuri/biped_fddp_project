#!/usr/bin/env python3
"""
STAGE 7C — Non-static double-support FDDP CoM-shift trajectories.

This script validates that the free-flyer biped model can generate a smooth,
contact-consistent centre-of-mass (CoM) motion while both feet remain rigidly
fixed to the ground. It solves and exports two FDDP references:

  1) forward:  CoM_y moves +12 mm, holds, then returns to neutral
  2) backward: CoM_y moves -12 mm, holds, then returns to neutral

Coordinate convention:
  +X = right, +Y = forward, +Z = up
  Pitch axis = X
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import yaml
import pinocchio as pin
import crocoddyl


# ------------------------------ User-facing configuration -------------------
DT = 0.010
HORIZON_NODES = 120                    # 1.20 s
MAX_ITER = 100
COM_SHIFT_M = 0.012                    # 12 mm; deliberately conservative
LEFT_FOOT_FRAME = "left_foot_contact_fixed"
RIGHT_FOOT_FRAME = "right_foot_contact_fixed"

# Cost weights. These are planning weights, not hardware-identified quantities.
RUNNING_STATE_WEIGHT = 0.05
RUNNING_CONTROL_WEIGHT = 1e-3
TERMINAL_STATE_WEIGHT = 5.0
COM_TRACKING_WEIGHT_Y = 2.0e4
COM_TRACKING_WEIGHT_XZ = 2.0e2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = PROJECT_ROOT / "model" / "biped_v1.urdf"
X0_PATH = PROJECT_ROOT / "transfer" / "initialisation" / "fddp_double_support_x0.npz"
OUTPUT_DIR = PROJECT_ROOT / "transfer" / "trajectories"


def evaluate_total_cost(problem, xs, us):
    """Evaluate total cost across running and terminal nodes."""
    problem.calc(xs, us)
    return (
        sum(float(data.cost) for data in problem.runningDatas)
        + float(problem.terminalData.cost)
    )


def load_x0(path: Path, expected_size: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing initial state: {path}\n"
            "Run: python fddp/01_prepare_double_support_initial_state.py"
        )

    with np.load(path) as loaded:
        for key in ("x0", "state", "initial_state"):
            if key in loaded.files:
                x0 = np.asarray(loaded[key], dtype=float).reshape(-1)
                break
        else:
            raise RuntimeError(
                f"Could not find x0 in {path}. Available keys: {loaded.files}"
            )

    if x0.size != expected_size:
        raise RuntimeError(
            f"Initial-state size mismatch: got {x0.size}, expected {expected_size}."
        )
    return x0


def make_contact_model(state, nu: int, frame_id: int):
    """Create a rigid 6D contact with the binding signature validated in Stage 7A."""
    return crocoddyl.ContactModel6D(
        state,
        frame_id,
        pin.SE3.Identity(),
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        nu,
        np.array([0.0, 0.0]),
    )


def make_integrated_model(
    state,
    actuation,
    x_ref: np.ndarray,
    com_target: np.ndarray,
    left_frame_id: int,
    right_frame_id: int,
    state_weight: float,
    control_weight: float,
    dt: float,
):
    """Build one contact-constrained Euler action model with a CoM target."""
    nu = actuation.nu

    contacts = crocoddyl.ContactModelMultiple(state, nu)
    contacts.addContact(
        LEFT_FOOT_FRAME,
        make_contact_model(state, nu, left_frame_id),
    )
    contacts.addContact(
        RIGHT_FOOT_FRAME,
        make_contact_model(state, nu, right_frame_id),
    )

    costs = crocoddyl.CostModelSum(state, nu)

    state_residual = crocoddyl.ResidualModelState(state, x_ref, nu)
    state_cost = crocoddyl.CostModelResidual(state, state_residual)
    costs.addCost("state_regularisation", state_cost, state_weight)

    control_residual = crocoddyl.ResidualModelControl(state, np.zeros(nu))
    control_cost = crocoddyl.CostModelResidual(state, control_residual)
    costs.addCost("control_regularisation", control_cost, control_weight)

    com_residual = crocoddyl.ResidualModelCoMPosition(state, com_target, nu)
    com_activation = crocoddyl.ActivationModelWeightedQuad(
        np.array([COM_TRACKING_WEIGHT_XZ, COM_TRACKING_WEIGHT_Y, COM_TRACKING_WEIGHT_XZ])
    )
    com_cost = crocoddyl.CostModelResidual(state, com_activation, com_residual)
    costs.addCost("com_tracking", com_cost, 1.0)

    differential = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state,
        actuation,
        contacts,
        costs,
        0.0,
        True,
    )
    return crocoddyl.IntegratedActionModelEuler(differential, dt)


def smoothstep(s: float) -> float:
    """Cubic zero-velocity endpoint interpolation for s in [0, 1]."""
    s = float(np.clip(s, 0.0, 1.0))
    return 3.0 * s * s - 2.0 * s * s * s


def make_com_y_schedule(initial_y: float, signed_shift_m: float) -> np.ndarray:
    """
    Build a neutral → shifted → neutral schedule over the 1.20 s horizon.

    0.00–0.20 s: neutral settle
    0.20–0.50 s: smooth ramp to peak shift
    0.50–0.80 s: hold
    0.80–1.10 s: smooth return
    1.10–1.20 s: neutral settle
    """
    t = np.arange(HORIZON_NODES + 1, dtype=float) * DT
    result = np.full(HORIZON_NODES + 1, initial_y, dtype=float)

    for i, ti in enumerate(t):
        if 0.20 <= ti < 0.50:
            alpha = smoothstep((ti - 0.20) / 0.30)
            result[i] = initial_y + signed_shift_m * alpha
        elif 0.50 <= ti < 0.80:
            result[i] = initial_y + signed_shift_m
        elif 0.80 <= ti < 1.10:
            alpha = smoothstep((ti - 0.80) / 0.30)
            result[i] = initial_y + signed_shift_m * (1.0 - alpha)

    return result


def compute_com_trajectory(model, xs: np.ndarray) -> np.ndarray:
    data = model.createData()
    com = np.zeros((xs.shape[0], 3), dtype=float)

    for k, x in enumerate(xs):
        q = x[: model.nq]
        com_k = np.asarray(pin.centerOfMass(model, data, q), dtype=float).reshape(3)
        com[k] = com_k

    return com


def compute_support_errors(
    model,
    xs: np.ndarray,
    left_frame_id: int,
    right_frame_id: int,
) -> tuple[float, float]:
    """Return maximum support-frame translation errors relative to node 0."""
    data = model.createData()
    q0 = xs[0, : model.nq]
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)
    left_ref = data.oMf[left_frame_id].translation.copy()
    right_ref = data.oMf[right_frame_id].translation.copy()

    left_max = 0.0
    right_max = 0.0
    for x in xs:
        q = x[: model.nq]
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        left_max = max(
            left_max,
            float(np.linalg.norm(data.oMf[left_frame_id].translation - left_ref)),
        )
        right_max = max(
            right_max,
            float(np.linalg.norm(data.oMf[right_frame_id].translation - right_ref)),
        )

    return left_max, right_max


def get_joint_limit_summary(model, xs: np.ndarray) -> dict:
    """
    Check only the six actuated joints. The first seven q entries belong to
    the free-flyer root and are not joint-angle limits.
    """
    q_joint = xs[:, 7 : model.nq]
    lower = np.asarray(model.lowerPositionLimit[7 : model.nq], dtype=float)
    upper = np.asarray(model.upperPositionLimit[7 : model.nq], dtype=float)

    lower_margin = q_joint - lower[None, :]
    upper_margin = upper[None, :] - q_joint

    return {
        "minimum_lower_margin_rad": float(np.min(lower_margin)),
        "minimum_upper_margin_rad": float(np.min(upper_margin)),
        "within_position_limits": bool(
            np.all(lower_margin >= -1e-8) and np.all(upper_margin >= -1e-8)
        ),
    }


def solve_one_direction(
    direction_name: str,
    signed_shift_m: float,
    model,
    x0: np.ndarray,
    left_frame_id: int,
    right_frame_id: int,
):
    """Solve, validate and export one non-static CoM-shift reference."""
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    nu = actuation.nu

    initial_com = compute_com_trajectory(model, x0[None, :])[0]
    com_y_target = make_com_y_schedule(initial_com[1], signed_shift_m)

    running_models = []
    for k in range(HORIZON_NODES):
        com_target = initial_com.copy()
        com_target[1] = com_y_target[k]
        running_models.append(
            make_integrated_model(
                state=state,
                actuation=actuation,
                x_ref=x0,
                com_target=com_target,
                left_frame_id=left_frame_id,
                right_frame_id=right_frame_id,
                state_weight=RUNNING_STATE_WEIGHT,
                control_weight=RUNNING_CONTROL_WEIGHT,
                dt=DT,
            )
        )

    terminal_target = initial_com.copy()
    terminal_target[1] = com_y_target[-1]
    terminal_model = make_integrated_model(
        state=state,
        actuation=actuation,
        x_ref=x0,
        com_target=terminal_target,
        left_frame_id=left_frame_id,
        right_frame_id=right_frame_id,
        state_weight=TERMINAL_STATE_WEIGHT,
        control_weight=0.0,
        dt=0.0,
    )

    problem = crocoddyl.ShootingProblem(x0, running_models, terminal_model)

    xs_init = [x0.copy() for _ in range(HORIZON_NODES + 1)]
    us_init = [np.zeros(nu, dtype=float) for _ in range(HORIZON_NODES)]
    initial_cost = evaluate_total_cost(problem, xs_init, us_init)

    solver = crocoddyl.SolverFDDP(problem)
    solver.th_stop = 1e-8
    success = bool(solver.solve(xs_init, us_init, MAX_ITER, False))

    xs = np.asarray(solver.xs, dtype=float)
    us = np.asarray(solver.us, dtype=float)

    expected_x_shape = (HORIZON_NODES + 1, model.nq + model.nv)
    expected_u_shape = (HORIZON_NODES, nu)
    if xs.shape != expected_x_shape:
        raise RuntimeError(f"Unexpected xs shape {xs.shape}; expected {expected_x_shape}.")
    if us.shape != expected_u_shape:
        raise RuntimeError(f"Unexpected us shape {us.shape}; expected {expected_u_shape}.")

    final_cost = evaluate_total_cost(
        problem,
        [x.copy() for x in xs],
        [u.copy() for u in us],
    )

    com_ref = compute_com_trajectory(model, xs)
    left_error, right_error = compute_support_errors(
        model, xs, left_frame_id, right_frame_id
    )
    joint_limit_summary = get_joint_limit_summary(model, xs)

    achieved_shift = float(
        np.max(com_ref[:, 1]) - initial_com[1]
        if signed_shift_m > 0.0
        else np.min(com_ref[:, 1]) - initial_com[1]
    )
    peak_tracking_error = float(np.max(np.abs(com_ref[:, 1] - com_y_target)))
    final_return_error = float(abs(com_ref[-1, 1] - initial_com[1]))

    # Export only the six actuated joint positions/velocities for downstream MPC.
    q_ref = xs[:, 7 : model.nq]
    dq_ref = xs[:, model.nq + 6 :]
    t_ref = np.arange(HORIZON_NODES + 1, dtype=float) * DT
    phase = np.full(HORIZON_NODES + 1, "double_support", dtype="U16")

    if q_ref.shape != (HORIZON_NODES + 1, 6):
        raise RuntimeError(f"q_ref shape error: got {q_ref.shape}; expected {(HORIZON_NODES + 1, 6)}.")
    if dq_ref.shape != (HORIZON_NODES + 1, 6):
        raise RuntimeError(f"dq_ref shape error: got {dq_ref.shape}; expected {(HORIZON_NODES + 1, 6)}.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"com_shift_{direction_name}_ref"
    npz_path = OUTPUT_DIR / f"{stem}.npz"
    yaml_path = OUTPUT_DIR / f"{stem}_summary.yaml"

    np.savez(
        npz_path,
        t_ref=t_ref,
        q_ref=q_ref,
        dq_ref=dq_ref,
        tau_ref=us,
        state_ref=xs,
        com_ref_world=com_ref,
        com_y_target=com_y_target,
        phase=phase,
        direction=np.asarray(direction_name),
        contact_mode=np.asarray("double_support"),
        left_contact_active=np.ones(HORIZON_NODES + 1, dtype=np.uint8),
        right_contact_active=np.ones(HORIZON_NODES + 1, dtype=np.uint8),
    )

    summary = {
        "stage": "7C_double_support_fddp_com_shift",
        "task": f"rigid_double_support_{direction_name}_com_shift_and_return",
        "purpose": (
            "Validate non-static FDDP trajectory generation under two rigid foot "
            "contacts before introducing swing-foot phases, MPC, or contact-event "
            "phase synchronisation."
        ),
        "coordinate_convention": {
            "forward_axis": "+Y",
            "pitch_axis": "+X",
        },
        "model": {
            "urdf": str(URDF_PATH),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "ndx": int(state.ndx),
            "nu": int(nu),
        },
        "solver": {
            "name": "SolverFDDP",
            "max_iterations": int(MAX_ITER),
            "success": bool(success),
            "stop_criterion": float(solver.stop),
            "initial_cost": float(initial_cost),
            "final_cost": float(final_cost),
        },
        "contacts": {
            "left_frame": LEFT_FOOT_FRAME,
            "right_frame": RIGHT_FOOT_FRAME,
            "max_left_translation_error_m": float(left_error),
            "max_right_translation_error_m": float(right_error),
        },
        "discretisation": {
            "dt_s": float(DT),
            "horizon_nodes": int(HORIZON_NODES),
            "horizon_s": float(HORIZON_NODES * DT),
        },
        "com_motion": {
            "initial_com_world_m": [float(v) for v in initial_com],
            "requested_peak_shift_y_m": float(signed_shift_m),
            "achieved_peak_shift_y_m": float(achieved_shift),
            "peak_abs_y_tracking_error_m": float(peak_tracking_error),
            "final_return_error_y_m": float(final_return_error),
        },
        "costs": {
            "running_state_weight": float(RUNNING_STATE_WEIGHT),
            "running_control_weight": float(RUNNING_CONTROL_WEIGHT),
            "terminal_state_weight": float(TERMINAL_STATE_WEIGHT),
            "com_tracking_weight_y": float(COM_TRACKING_WEIGHT_Y),
            "com_tracking_weight_xz": float(COM_TRACKING_WEIGHT_XZ),
        },
        "joint_limits": joint_limit_summary,
        "outputs": {
            "trajectory_npz": str(npz_path),
            "q_ref_shape": [int(v) for v in q_ref.shape],
            "dq_ref_shape": [int(v) for v in dq_ref.shape],
            "tau_ref_shape": [int(v) for v in us.shape],
            "com_ref_world_shape": [int(v) for v in com_ref.shape],
        },
    }

    yaml_path.write_text(
        yaml.safe_dump(summary, sort_keys=False),
        encoding="utf-8",
    )

    print()
    print(f"----- {direction_name.upper()} CoM SHIFT -----")
    print(f"Solver success: {success}")
    print(f"Solver stop criterion: {solver.stop:.3e}")
    print(f"Initial cost: {initial_cost:.10f}")
    print(f"Final cost:   {final_cost:.10f}")
    print(
        "Requested / achieved peak CoM_y shift: "
        f"{signed_shift_m * 1000.0:+.2f} / {achieved_shift * 1000.0:+.2f} mm"
    )
    print(f"Peak |CoM_y tracking error|: {peak_tracking_error * 1000.0:.3f} mm")
    print(f"Final CoM_y return error: {final_return_error * 1000.0:.3f} mm")
    print(
        "Maximum left/right support-frame translation error: "
        f"{left_error:.3e} / {right_error:.3e} m"
    )
    print(
        "Minimum lower/upper joint-limit margin: "
        f"{joint_limit_summary['minimum_lower_margin_rad']:.5f} / "
        f"{joint_limit_summary['minimum_upper_margin_rad']:.5f} rad"
    )
    print(f"Saved:\n  {npz_path}\n  {yaml_path}")

    # These are integrity gates, not intended to make a performance claim.
    if not success:
        raise RuntimeError(f"{direction_name}: SolverFDDP did not report success.")
    if not joint_limit_summary["within_position_limits"]:
        raise RuntimeError(f"{direction_name}: trajectory violates a joint position limit.")
    if abs(achieved_shift) < 0.004:
        raise RuntimeError(
            f"{direction_name}: achieved CoM shift is too small ({achieved_shift * 1000.0:.2f} mm). "
            "Do not proceed to MPC; inspect the exported summary first."
        )
    if final_return_error > 0.004:
        raise RuntimeError(
            f"{direction_name}: final CoM return error is too large "
            f"({final_return_error * 1000.0:.2f} mm)."
        )

    return summary


def main():
    print("===== STAGE 7C: DOUBLE-SUPPORT FDDP CoM-SHIFT SOLVE =====")
    print("Task: rigid double-support CoM shift and return")
    print("This is still NOT walking: both feet remain fixed in contact.")
    print(f"URDF: {URDF_PATH}")
    print(f"Horizon: {HORIZON_NODES} nodes x {DT:.3f} s = {HORIZON_NODES * DT:.3f} s")
    print(f"Requested CoM shift magnitude: {COM_SHIFT_M * 1000.0:.1f} mm")

    if not URDF_PATH.exists():
        raise FileNotFoundError(f"URDF not found: {URDF_PATH}")

    model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
    state = crocoddyl.StateMultibody(model)
    x0 = load_x0(X0_PATH, model.nq + model.nv)

    if model.nq != 13 or model.nv != 12:
        raise RuntimeError(
            f"Unexpected model dimensions nq={model.nq}, nv={model.nv}; "
            "this script expects the verified six-pitch free-flyer model."
        )

    try:
        left_frame_id = model.getFrameId(LEFT_FOOT_FRAME)
        right_frame_id = model.getFrameId(RIGHT_FOOT_FRAME)
    except Exception as exc:
        raise RuntimeError("Could not resolve support-foot frame IDs.") from exc

    if left_frame_id >= model.nframes or right_frame_id >= model.nframes:
        raise RuntimeError(
            f"Invalid support frame IDs: left={left_frame_id}, right={right_frame_id}."
        )

    solve_one_direction(
        "forward",
        +COM_SHIFT_M,
        model,
        x0,
        left_frame_id,
        right_frame_id,
    )
    solve_one_direction(
        "backward",
        -COM_SHIFT_M,
        model,
        x0,
        left_frame_id,
        right_frame_id,
    )

    print()
    print("DOUBLE-SUPPORT FDDP CoM-SHIFT SOLVE PASSED")
    print("Next stage: inspect the two references, then transfer a selected one to MuJoCo.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nSTAGE 7C FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
