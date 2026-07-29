#!/usr/bin/env python3
"""
STAGE 7C-v2 — Contact-stabilised double-support FDDP CoM-shift trajectories.

Why v2 exists:
The first Stage 7C run used zero-gain acceleration-level contacts and an
identity contact reference. Under non-static motion, this allowed foot-position
drift (especially at the right foot). This version:
  1) fixes each support foot to its actual x0 world placement; and
  2) adds moderate Baumgarte contact stabilisation gains.

It deliberately uses a conservative +/-6 mm forward/backward CoM shift because
the original -12 mm backward trajectory violated a joint-position limit.
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import yaml
import pinocchio as pin
import crocoddyl


# -------------------- Conservative Stage 7C-v2 configuration ----------------
DT = 0.010
HORIZON_NODES = 120                    # 1.20 s
MAX_ITER = 100
COM_SHIFT_M = 0.006                    # +/-6 mm: safe non-static validation motion

LEFT_FOOT_FRAME = "left_foot_contact_fixed"
RIGHT_FOOT_FRAME = "right_foot_contact_fixed"

# Crocoddyl contact gains are [natural frequency, damping ratio].
# This corresponds approximately to position feedback of 10^2 and velocity
# feedback of 2*10*1 in the stabilised contact acceleration constraint.
CONTACT_STABILISATION_GAINS = np.array([10.0, 1.0], dtype=float)

RUNNING_STATE_WEIGHT = 0.05
RUNNING_CONTROL_WEIGHT = 1e-3
TERMINAL_STATE_WEIGHT = 5.0
COM_TRACKING_WEIGHT_Y = 2.0e4
COM_TRACKING_WEIGHT_XZ = 2.0e2

# Integrity gates. A trajectory that fails any of these is deliberately not
# exported as an MPC reference.
MAX_SUPPORT_TRANSLATION_ERROR_M = 5.0e-4   # 0.5 mm
MIN_JOINT_LIMIT_MARGIN_RAD = 1.0e-2        # 0.57 deg
MIN_ACHIEVED_SHIFT_M = 0.003               # 3 mm
MAX_FINAL_RETURN_ERROR_M = 0.003           # 3 mm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = PROJECT_ROOT / "model" / "biped_v1.urdf"
X0_PATH = PROJECT_ROOT / "transfer" / "initialisation" / "fddp_double_support_x0.npz"
OUTPUT_DIR = PROJECT_ROOT / "transfer" / "trajectories"


def evaluate_total_cost(problem, xs, us):
    """Evaluate total cost across all running nodes plus the terminal node."""
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


def get_fixed_support_references(
    model: pin.Model,
    x0: np.ndarray,
    left_frame_id: int,
    right_frame_id: int,
) -> tuple[pin.SE3, pin.SE3]:
    """
    Compute the world placements of both support frames at x0.
    These are the desired positions/orientations that must remain fixed
    throughout the double-support trajectory.
    """
    data = model.createData()
    q0 = x0[: model.nq]
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)
    return data.oMf[left_frame_id].copy(), data.oMf[right_frame_id].copy()


def make_contact_model(
    state,
    nu: int,
    frame_id: int,
    reference_placement: pin.SE3,
):
    """
    A stabilised rigid 6D contact in LOCAL_WORLD_ALIGNED coordinates.
    The reference placement is the foot world pose at x0, not SE3.Identity().
    """
    return crocoddyl.ContactModel6D(
        state,
        frame_id,
        reference_placement,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        nu,
        CONTACT_STABILISATION_GAINS,
    )


def make_integrated_model(
    state,
    actuation,
    x_ref: np.ndarray,
    com_target: np.ndarray,
    left_frame_id: int,
    right_frame_id: int,
    left_reference: pin.SE3,
    right_reference: pin.SE3,
    state_weight: float,
    control_weight: float,
    dt: float,
):
    """Build one double-support contact-constrained Euler action model."""
    nu = actuation.nu

    contacts = crocoddyl.ContactModelMultiple(state, nu)
    contacts.addContact(
        LEFT_FOOT_FRAME,
        make_contact_model(state, nu, left_frame_id, left_reference),
    )
    contacts.addContact(
        RIGHT_FOOT_FRAME,
        make_contact_model(state, nu, right_frame_id, right_reference),
    )

    costs = crocoddyl.CostModelSum(state, nu)

    state_residual = crocoddyl.ResidualModelState(state, x_ref, nu)
    costs.addCost(
        "state_regularisation",
        crocoddyl.CostModelResidual(state, state_residual),
        state_weight,
    )

    control_residual = crocoddyl.ResidualModelControl(state, np.zeros(nu))
    costs.addCost(
        "control_regularisation",
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
        0.0,
        True,
    )
    return crocoddyl.IntegratedActionModelEuler(differential, dt)


def smoothstep(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return 3.0 * s * s - 2.0 * s * s * s


def make_com_y_schedule(initial_y: float, signed_shift_m: float) -> np.ndarray:
    """Neutral -> shifted -> neutral schedule across the 1.20 s trajectory."""
    t = np.arange(HORIZON_NODES + 1, dtype=float) * DT
    target = np.full(HORIZON_NODES + 1, initial_y, dtype=float)

    for k, tk in enumerate(t):
        if 0.20 <= tk < 0.50:
            target[k] = initial_y + signed_shift_m * smoothstep((tk - 0.20) / 0.30)
        elif 0.50 <= tk < 0.80:
            target[k] = initial_y + signed_shift_m
        elif 0.80 <= tk < 1.10:
            target[k] = initial_y + signed_shift_m * (
                1.0 - smoothstep((tk - 0.80) / 0.30)
            )
    return target


def compute_com_trajectory(model: pin.Model, xs: np.ndarray) -> np.ndarray:
    data = model.createData()
    com = np.zeros((xs.shape[0], 3), dtype=float)
    for k, x in enumerate(xs):
        com[k] = np.asarray(
            pin.centerOfMass(model, data, x[: model.nq]), dtype=float
        ).reshape(3)
    return com


def compute_support_errors(
    model: pin.Model,
    xs: np.ndarray,
    left_frame_id: int,
    right_frame_id: int,
    left_reference: pin.SE3,
    right_reference: pin.SE3,
) -> tuple[float, float]:
    """Maximum translational deviation from the intended fixed world foot poses."""
    data = model.createData()
    left_max = 0.0
    right_max = 0.0

    for x in xs:
        pin.forwardKinematics(model, data, x[: model.nq])
        pin.updateFramePlacements(model, data)
        left_max = max(
            left_max,
            float(np.linalg.norm(data.oMf[left_frame_id].translation - left_reference.translation)),
        )
        right_max = max(
            right_max,
            float(np.linalg.norm(data.oMf[right_frame_id].translation - right_reference.translation)),
        )
    return left_max, right_max


def get_joint_limit_summary(model: pin.Model, xs: np.ndarray) -> dict:
    q_joint = xs[:, 7 : model.nq]
    lower = np.asarray(model.lowerPositionLimit[7 : model.nq], dtype=float)
    upper = np.asarray(model.upperPositionLimit[7 : model.nq], dtype=float)

    lower_margin = q_joint - lower[None, :]
    upper_margin = upper[None, :] - q_joint
    min_lower = float(np.min(lower_margin))
    min_upper = float(np.min(upper_margin))

    return {
        "minimum_lower_margin_rad": min_lower,
        "minimum_upper_margin_rad": min_upper,
        "minimum_margin_rad": float(min(min_lower, min_upper)),
        "within_position_limits": bool(
            np.all(lower_margin >= -1e-8) and np.all(upper_margin >= -1e-8)
        ),
        "has_safety_margin": bool(
            min_lower >= MIN_JOINT_LIMIT_MARGIN_RAD
            and min_upper >= MIN_JOINT_LIMIT_MARGIN_RAD
        ),
    }


def check_integrity(
    direction_name: str,
    success: bool,
    achieved_shift: float,
    return_error: float,
    left_error: float,
    right_error: float,
    joint_limits: dict,
):
    """Reject invalid trajectories before they can become downstream references."""
    failures = []
    if not success:
        failures.append("SolverFDDP did not report success")
    if abs(achieved_shift) < MIN_ACHIEVED_SHIFT_M:
        failures.append(
            f"achieved CoM shift {achieved_shift * 1000.0:.2f} mm is below "
            f"{MIN_ACHIEVED_SHIFT_M * 1000.0:.1f} mm"
        )
    if return_error > MAX_FINAL_RETURN_ERROR_M:
        failures.append(
            f"final CoM return error {return_error * 1000.0:.2f} mm exceeds "
            f"{MAX_FINAL_RETURN_ERROR_M * 1000.0:.1f} mm"
        )
    if left_error > MAX_SUPPORT_TRANSLATION_ERROR_M:
        failures.append(
            f"left-foot drift {left_error * 1000.0:.3f} mm exceeds "
            f"{MAX_SUPPORT_TRANSLATION_ERROR_M * 1000.0:.1f} mm"
        )
    if right_error > MAX_SUPPORT_TRANSLATION_ERROR_M:
        failures.append(
            f"right-foot drift {right_error * 1000.0:.3f} mm exceeds "
            f"{MAX_SUPPORT_TRANSLATION_ERROR_M * 1000.0:.1f} mm"
        )
    if not joint_limits["within_position_limits"]:
        failures.append("trajectory violates a joint position limit")
    elif not joint_limits["has_safety_margin"]:
        failures.append(
            f"joint-limit safety margin is only "
            f"{joint_limits['minimum_margin_rad']:.5f} rad; required "
            f"{MIN_JOINT_LIMIT_MARGIN_RAD:.5f} rad"
        )

    if failures:
        raise RuntimeError(f"{direction_name}: " + "; ".join(failures))


def solve_one_direction(
    direction_name: str,
    signed_shift_m: float,
    model: pin.Model,
    x0: np.ndarray,
    left_frame_id: int,
    right_frame_id: int,
    left_reference: pin.SE3,
    right_reference: pin.SE3,
):
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    nu = actuation.nu

    initial_com = compute_com_trajectory(model, x0[None, :])[0]
    com_y_target = make_com_y_schedule(initial_com[1], signed_shift_m)

    running_models = []
    for k in range(HORIZON_NODES):
        target = initial_com.copy()
        target[1] = com_y_target[k]
        running_models.append(
            make_integrated_model(
                state,
                actuation,
                x0,
                target,
                left_frame_id,
                right_frame_id,
                left_reference,
                right_reference,
                RUNNING_STATE_WEIGHT,
                RUNNING_CONTROL_WEIGHT,
                DT,
            )
        )

    terminal_target = initial_com.copy()
    terminal_target[1] = com_y_target[-1]
    terminal_model = make_integrated_model(
        state,
        actuation,
        x0,
        terminal_target,
        left_frame_id,
        right_frame_id,
        left_reference,
        right_reference,
        TERMINAL_STATE_WEIGHT,
        0.0,
        0.0,
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

    expected_xs_shape = (HORIZON_NODES + 1, model.nq + model.nv)
    expected_us_shape = (HORIZON_NODES, nu)
    if xs.shape != expected_xs_shape:
        raise RuntimeError(f"Unexpected xs shape {xs.shape}; expected {expected_xs_shape}.")
    if us.shape != expected_us_shape:
        raise RuntimeError(f"Unexpected us shape {us.shape}; expected {expected_us_shape}.")

    final_cost = evaluate_total_cost(problem, [x.copy() for x in xs], [u.copy() for u in us])
    com_ref = compute_com_trajectory(model, xs)
    left_error, right_error = compute_support_errors(
        model,
        xs,
        left_frame_id,
        right_frame_id,
        left_reference,
        right_reference,
    )
    joint_limits = get_joint_limit_summary(model, xs)

    if signed_shift_m > 0.0:
        achieved_shift = float(np.max(com_ref[:, 1]) - initial_com[1])
    else:
        achieved_shift = float(np.min(com_ref[:, 1]) - initial_com[1])

    peak_tracking_error = float(np.max(np.abs(com_ref[:, 1] - com_y_target)))
    return_error = float(abs(com_ref[-1, 1] - initial_com[1]))

    print()
    print(f"----- {direction_name.upper()} CoM SHIFT (v2) -----")
    print(f"Solver success: {success}")
    print(f"Solver stop criterion: {solver.stop:.3e}")
    print(f"Initial cost: {initial_cost:.10f}")
    print(f"Final cost:   {final_cost:.10f}")
    print(
        "Requested / achieved peak CoM_y shift: "
        f"{signed_shift_m * 1000.0:+.2f} / {achieved_shift * 1000.0:+.2f} mm"
    )
    print(f"Peak |CoM_y tracking error|: {peak_tracking_error * 1000.0:.3f} mm")
    print(f"Final CoM_y return error: {return_error * 1000.0:.3f} mm")
    print(
        "Maximum left/right support-frame translation error: "
        f"{left_error * 1000.0:.3f} / {right_error * 1000.0:.3f} mm"
    )
    print(
        "Minimum lower/upper joint-limit margin: "
        f"{joint_limits['minimum_lower_margin_rad']:.5f} / "
        f"{joint_limits['minimum_upper_margin_rad']:.5f} rad"
    )

    # Do not write MPC reference files unless all physics/feasibility gates pass.
    check_integrity(
        direction_name,
        success,
        achieved_shift,
        return_error,
        left_error,
        right_error,
        joint_limits,
    )

    q_ref = xs[:, 7 : model.nq]
    dq_ref = xs[:, model.nq + 6 :]
    t_ref = np.arange(HORIZON_NODES + 1, dtype=float) * DT
    phase = np.full(HORIZON_NODES + 1, "double_support", dtype="U16")

    if q_ref.shape != (HORIZON_NODES + 1, 6):
        raise RuntimeError(f"q_ref shape {q_ref.shape}; expected {(HORIZON_NODES + 1, 6)}.")
    if dq_ref.shape != (HORIZON_NODES + 1, 6):
        raise RuntimeError(f"dq_ref shape {dq_ref.shape}; expected {(HORIZON_NODES + 1, 6)}.")

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
        "stage": "7C_v2_double_support_fddp_com_shift",
        "task": f"contact_stabilised_rigid_double_support_{direction_name}_com_shift_and_return",
        "purpose": (
            "Validate a non-static, contact-consistent FDDP trajectory before swing-foot "
            "phases, MPC, or contact-event phase synchronisation."
        ),
        "coordinate_convention": {"forward_axis": "+Y", "pitch_axis": "+X"},
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
            "reference_mode": "fixed_world_pose_at_x0",
            "stabilisation_gains": [float(v) for v in CONTACT_STABILISATION_GAINS],
            "max_left_translation_error_m": float(left_error),
            "max_right_translation_error_m": float(right_error),
            "required_max_translation_error_m": float(MAX_SUPPORT_TRANSLATION_ERROR_M),
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
            "final_return_error_y_m": float(return_error),
        },
        "joint_limits": joint_limits,
        "outputs": {
            "trajectory_npz": str(npz_path),
            "q_ref_shape": [int(v) for v in q_ref.shape],
            "dq_ref_shape": [int(v) for v in dq_ref.shape],
            "tau_ref_shape": [int(v) for v in us.shape],
            "com_ref_world_shape": [int(v) for v in com_ref.shape],
        },
    }
    yaml_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")

    print("Integrity gates: PASSED")
    print(f"Saved:\n  {npz_path}\n  {yaml_path}")


def main():
    print("===== STAGE 7C-v2: CONTACT-STABILISED DOUBLE-SUPPORT CoM SHIFT =====")
    print("Task: non-static, rigid double-support CoM shift and return")
    print("Both feet must remain fixed at their x0 world placements.")
    print(f"URDF: {URDF_PATH}")
    print(f"Horizon: {HORIZON_NODES} nodes x {DT:.3f} s = {HORIZON_NODES * DT:.3f} s")
    print(f"Requested CoM shift magnitude: {COM_SHIFT_M * 1000.0:.1f} mm")
    print(
        "Contact stabilisation gains [frequency, damping ratio]: "
        f"{CONTACT_STABILISATION_GAINS.tolist()}"
    )

    if not URDF_PATH.exists():
        raise FileNotFoundError(f"URDF not found: {URDF_PATH}")

    model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
    if model.nq != 13 or model.nv != 12:
        raise RuntimeError(
            f"Unexpected model dimensions nq={model.nq}, nv={model.nv}; "
            "expected six-pitch free-flyer model nq=13, nv=12."
        )

    x0 = load_x0(X0_PATH, model.nq + model.nv)
    state = crocoddyl.StateMultibody(model)

    left_frame_id = model.getFrameId(LEFT_FOOT_FRAME)
    right_frame_id = model.getFrameId(RIGHT_FOOT_FRAME)
    if left_frame_id >= model.nframes or right_frame_id >= model.nframes:
        raise RuntimeError(
            f"Invalid support-frame IDs: left={left_frame_id}, right={right_frame_id}."
        )

    left_reference, right_reference = get_fixed_support_references(
        model, x0, left_frame_id, right_frame_id
    )
    print(
        "Support references at x0 (m): "
        f"left={left_reference.translation.tolist()}, "
        f"right={right_reference.translation.tolist()}"
    )

    solve_one_direction(
        "forward",
        +COM_SHIFT_M,
        model,
        x0,
        left_frame_id,
        right_frame_id,
        left_reference,
        right_reference,
    )
    solve_one_direction(
        "backward",
        -COM_SHIFT_M,
        model,
        x0,
        left_frame_id,
        right_frame_id,
        left_reference,
        right_reference,
    )

    print()
    print("STAGE 7C-v2 PASSED")
    print("Next stage: visualise these valid references in MuJoCo before building MPC.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nSTAGE 7C-v2 FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
