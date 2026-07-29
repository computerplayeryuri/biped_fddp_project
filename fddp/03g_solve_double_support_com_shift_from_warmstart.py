#!/usr/bin/env python3
"""
STAGE 7C-FDDP — Solve non-static rigid double-support CoM shifts from
contact-consistent kinematic warm-start trajectories.

This script is the first dynamics solve that uses:
  xs_init, us_init from Stage 7C warm-start generation
rather than repeating static x0 for every node.

It solves two independent tasks:
  forward:  CoM_y +6 mm, hold, return
  backward: CoM_y -6 mm, hold, return

Both feet remain 6D rigid contacts referenced to their actual x0 world poses.

IMPORTANT:
- This is still NOT walking.
- tau_ref is a Crocoddyl virtual torque trajectory, not a command for ST3215.
- Later MPC/MuJoCo stages will consume q_ref and dq_ref only.
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import yaml
import pinocchio as pin
import crocoddyl


# ------------------------------- Configuration -------------------------------
DT = 0.010
HORIZON_NODES = 120
MAX_ITER = 300

LEFT_FOOT_FRAME = "left_foot_contact_fixed"
RIGHT_FOOT_FRAME = "right_foot_contact_fixed"

# Direct Crocoddyl Baumgarte gains [Kp, Kd].
# This is a moderate critically damped setting: sqrt(Kp)=20 rad/s, Kd=40.
# It is deliberately less stiff than v3 [2500,100] but stronger than v2 [10,1].
CONTACT_GAINS = np.array([400.0, 40.0], dtype=float)

RUNNING_STATE_WEIGHT = 0.05
RUNNING_CONTROL_WEIGHT = 1e-3
TERMINAL_STATE_WEIGHT = 5.0
COM_TRACKING_WEIGHT_Y = 2.0e4
COM_TRACKING_WEIGHT_XZ = 2.0e2

# Physical/trajectory integrity gates.
MAX_SUPPORT_TRANSLATION_ERROR_M = 5.0e-4     # 0.5 mm
MAX_SUPPORT_ROTATION_ERROR_RAD = np.deg2rad(0.5)
MIN_JOINT_LIMIT_MARGIN_RAD = 1.0e-2          # 0.57 deg
MIN_ACHIEVED_SHIFT_M = 0.003                 # 3 mm
MAX_COM_Y_TRACKING_ERROR_M = 1.5e-3          # 1.5 mm
MAX_FINAL_COM_Y_RETURN_ERROR_M = 1.0e-3      # 1.0 mm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = PROJECT_ROOT / "model" / "biped_v1.urdf"
X0_PATH = PROJECT_ROOT / "transfer" / "initialisation" / "fddp_double_support_x0.npz"
TRAJECTORY_DIR = PROJECT_ROOT / "transfer" / "trajectories"


def evaluate_total_cost(problem, xs, us):
    """Evaluate cost on the current trajectory for Crocoddyl builds without problem.cost."""
    problem.calc(xs, us)
    return (
        sum(float(data.cost) for data in problem.runningDatas)
        + float(problem.terminalData.cost)
    )


def load_npz_vector(path: Path, expected_size: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with np.load(path) as loaded:
        for key in ("x0", "state", "initial_state"):
            if key in loaded.files:
                vector = np.asarray(loaded[key], dtype=float).reshape(-1)
                break
        else:
            raise RuntimeError(
                f"Could not find x0 in {path}. Available keys: {loaded.files}"
            )

    if vector.size != expected_size:
        raise RuntimeError(
            f"x0 size mismatch: got {vector.size}, expected {expected_size}."
        )
    return vector


def load_warmstart(direction_name: str, expected_x_shape, expected_u_shape):
    path = TRAJECTORY_DIR / f"com_shift_{direction_name}_warmstart.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing warm-start: {path}\n"
            "Run: python fddp/03f_build_double_support_com_shift_warmstart.py"
        )

    with np.load(path) as loaded:
        required = ("xs_init", "us_init", "com_y_target", "t_ref")
        missing = [name for name in required if name not in loaded.files]
        if missing:
            raise RuntimeError(
                f"Warm-start {path} is missing keys {missing}; "
                f"available: {loaded.files}"
            )

        xs_init = np.asarray(loaded["xs_init"], dtype=float)
        us_init = np.asarray(loaded["us_init"], dtype=float)
        com_y_target = np.asarray(loaded["com_y_target"], dtype=float).reshape(-1)
        t_ref = np.asarray(loaded["t_ref"], dtype=float).reshape(-1)

    if xs_init.shape != expected_x_shape:
        raise RuntimeError(
            f"Warm-start xs_init shape {xs_init.shape}; expected {expected_x_shape}."
        )
    if us_init.shape != expected_u_shape:
        raise RuntimeError(
            f"Warm-start us_init shape {us_init.shape}; expected {expected_u_shape}."
        )
    if com_y_target.shape != (HORIZON_NODES + 1,):
        raise RuntimeError(
            f"com_y_target shape {com_y_target.shape}; "
            f"expected {(HORIZON_NODES + 1,)}."
        )
    if t_ref.shape != (HORIZON_NODES + 1,):
        raise RuntimeError(
            f"t_ref shape {t_ref.shape}; expected {(HORIZON_NODES + 1,)}."
        )

    return xs_init, us_init, com_y_target, t_ref, path


def get_support_references(
    model: pin.Model,
    x0: np.ndarray,
    left_frame_id: int,
    right_frame_id: int,
):
    data = model.createData()
    pin.forwardKinematics(model, data, x0[: model.nq])
    pin.updateFramePlacements(model, data)
    return data.oMf[left_frame_id].copy(), data.oMf[right_frame_id].copy()


def make_contact_model(
    state,
    nu: int,
    frame_id: int,
    reference: pin.SE3,
):
    return crocoddyl.ContactModel6D(
        state,
        frame_id,
        reference,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        nu,
        CONTACT_GAINS,
    )


def make_action_model(
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
            [COM_TRACKING_WEIGHT_XZ, COM_TRACKING_WEIGHT_Y, COM_TRACKING_WEIGHT_XZ],
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


def compute_com(model: pin.Model, xs: np.ndarray) -> np.ndarray:
    data = model.createData()
    output = np.zeros((xs.shape[0], 3), dtype=float)
    for k, x in enumerate(xs):
        output[k] = np.asarray(
            pin.centerOfMass(model, data, x[: model.nq]), dtype=float
        ).reshape(3)
    return output


def pose_errors(reference: pin.SE3, current: pin.SE3):
    translation_error = float(
        np.linalg.norm(current.translation - reference.translation)
    )
    rotation_error = float(
        np.linalg.norm(pin.log3(reference.rotation.T @ current.rotation))
    )
    return translation_error, rotation_error


def compute_support_errors(
    model: pin.Model,
    xs: np.ndarray,
    left_frame_id: int,
    right_frame_id: int,
    left_reference: pin.SE3,
    right_reference: pin.SE3,
):
    data = model.createData()
    left_translation_max = 0.0
    right_translation_max = 0.0
    left_rotation_max = 0.0
    right_rotation_max = 0.0

    for x in xs:
        pin.forwardKinematics(model, data, x[: model.nq])
        pin.updateFramePlacements(model, data)

        left_t, left_r = pose_errors(left_reference, data.oMf[left_frame_id])
        right_t, right_r = pose_errors(right_reference, data.oMf[right_frame_id])

        left_translation_max = max(left_translation_max, left_t)
        right_translation_max = max(right_translation_max, right_t)
        left_rotation_max = max(left_rotation_max, left_r)
        right_rotation_max = max(right_rotation_max, right_r)

    return {
        "max_left_translation_error_m": float(left_translation_max),
        "max_right_translation_error_m": float(right_translation_max),
        "max_left_rotation_error_rad": float(left_rotation_max),
        "max_right_rotation_error_rad": float(right_rotation_max),
    }


def joint_limit_summary(model: pin.Model, xs: np.ndarray):
    joint_positions = xs[:, 7:model.nq]
    lower = np.asarray(model.lowerPositionLimit[7:model.nq], dtype=float)
    upper = np.asarray(model.upperPositionLimit[7:model.nq], dtype=float)

    lower_margin = joint_positions - lower[None, :]
    upper_margin = upper[None, :] - joint_positions
    minimum_lower = float(np.min(lower_margin))
    minimum_upper = float(np.min(upper_margin))
    minimum_margin = float(min(minimum_lower, minimum_upper))

    return {
        "minimum_lower_margin_rad": minimum_lower,
        "minimum_upper_margin_rad": minimum_upper,
        "minimum_margin_rad": minimum_margin,
        "within_position_limits": bool(
            np.all(lower_margin >= -1e-10) and np.all(upper_margin >= -1e-10)
        ),
        "has_safety_margin": bool(minimum_margin >= MIN_JOINT_LIMIT_MARGIN_RAD),
    }


def assess_integrity(
    direction_name: str,
    accepted_convergence: bool,
    achieved_shift: float,
    peak_tracking_error: float,
    final_return_error: float,
    supports: dict,
    limits: dict,
):
    failures = []

    if not accepted_convergence:
        failures.append("SolverFDDP did not satisfy the convergence criterion")
    if abs(achieved_shift) < MIN_ACHIEVED_SHIFT_M:
        failures.append(
            f"achieved CoM shift {achieved_shift * 1000.0:.3f} mm is below "
            f"{MIN_ACHIEVED_SHIFT_M * 1000.0:.1f} mm"
        )
    if peak_tracking_error > MAX_COM_Y_TRACKING_ERROR_M:
        failures.append(
            f"peak CoM_y tracking error {peak_tracking_error * 1000.0:.3f} mm "
            f"exceeds {MAX_COM_Y_TRACKING_ERROR_M * 1000.0:.1f} mm"
        )
    if final_return_error > MAX_FINAL_COM_Y_RETURN_ERROR_M:
        failures.append(
            f"final CoM_y return error {final_return_error * 1000.0:.3f} mm "
            f"exceeds {MAX_FINAL_COM_Y_RETURN_ERROR_M * 1000.0:.1f} mm"
        )

    for side in ("left", "right"):
        translation = supports[f"max_{side}_translation_error_m"]
        rotation = supports[f"max_{side}_rotation_error_rad"]
        if translation > MAX_SUPPORT_TRANSLATION_ERROR_M:
            failures.append(
                f"{side}-foot translation error {translation * 1000.0:.3f} mm "
                f"exceeds {MAX_SUPPORT_TRANSLATION_ERROR_M * 1000.0:.1f} mm"
            )
        if rotation > MAX_SUPPORT_ROTATION_ERROR_RAD:
            failures.append(
                f"{side}-foot rotation error {np.rad2deg(rotation):.3f} deg "
                f"exceeds {np.rad2deg(MAX_SUPPORT_ROTATION_ERROR_RAD):.1f} deg"
            )

    if not limits["within_position_limits"]:
        failures.append("trajectory violates a joint position limit")
    elif not limits["has_safety_margin"]:
        failures.append(
            f"joint-limit margin {limits['minimum_margin_rad']:.5f} rad is below "
            f"{MIN_JOINT_LIMIT_MARGIN_RAD:.5f} rad"
        )

    if failures:
        print("FDDP integrity: FAILED")
        for item in failures:
            print(f"  - {item}")
        raise RuntimeError(f"{direction_name}: " + "; ".join(failures))


def solve_direction(
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

    expected_x_shape = (HORIZON_NODES + 1, model.nq + model.nv)
    expected_u_shape = (HORIZON_NODES, nu)
    xs_init, us_init, com_y_target, t_ref, warmstart_path = load_warmstart(
        direction_name, expected_x_shape, expected_u_shape
    )

    initial_com = compute_com(model, x0[None, :])[0]
    requested_peak_shift = float(
        np.max(com_y_target) - initial_com[1]
        if signed_shift_m > 0.0
        else np.min(com_y_target) - initial_com[1]
    )

    running_models = []
    for k in range(HORIZON_NODES):
        target = initial_com.copy()
        target[1] = com_y_target[k]
        running_models.append(
            make_action_model(
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
    terminal_model = make_action_model(
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

    xs_seed = [row.copy() for row in xs_init]
    us_seed = [row.copy() for row in us_init]
    initial_cost = evaluate_total_cost(problem, xs_seed, us_seed)

    solver = crocoddyl.SolverFDDP(problem)
    solver.th_stop = 1e-7
    solver_return_success = bool(solver.solve(xs_seed, us_seed, MAX_ITER, False))
    stopping_criterion_passed = bool(float(solver.stop) <= float(solver.th_stop))
    accepted_convergence = bool(
        solver_return_success or stopping_criterion_passed
    )

    xs = np.asarray(solver.xs, dtype=float)
    us = np.asarray(solver.us, dtype=float)
    if xs.shape != expected_x_shape:
        raise RuntimeError(f"Unexpected solver xs shape {xs.shape}; expected {expected_x_shape}.")
    if us.shape != expected_u_shape:
        raise RuntimeError(f"Unexpected solver us shape {us.shape}; expected {expected_u_shape}.")

    final_cost = evaluate_total_cost(
        problem, [row.copy() for row in xs], [row.copy() for row in us]
    )

    com_ref = compute_com(model, xs)
    if signed_shift_m > 0.0:
        achieved_shift = float(np.max(com_ref[:, 1]) - initial_com[1])
    else:
        achieved_shift = float(np.min(com_ref[:, 1]) - initial_com[1])
    peak_tracking_error = float(np.max(np.abs(com_ref[:, 1] - com_y_target)))
    final_return_error = float(abs(com_ref[-1, 1] - initial_com[1]))
    supports = compute_support_errors(
        model,
        xs,
        left_frame_id,
        right_frame_id,
        left_reference,
        right_reference,
    )
    limits = joint_limit_summary(model, xs)

    q_ref = xs[:, 7:model.nq]
    dq_ref = xs[:, model.nq + 6:]
    phase = np.full(HORIZON_NODES + 1, "double_support", dtype="U16")

    print()
    print(f"----- {direction_name.upper()} FDDP FROM WARM-START -----")
    print(f"Warm-start: {warmstart_path}")
    print(f"Solver return value: {solver_return_success}")
    print(f"Solver stop criterion: {solver.stop:.3e}")
    print(f"Stopping criterion passed: {stopping_criterion_passed}")
    print(f"Solver accepted: {accepted_convergence}")
    print(f"Initial / final cost: {initial_cost:.10f} / {final_cost:.10f}")
    print(
        "Requested / achieved peak CoM_y shift: "
        f"{requested_peak_shift * 1000.0:+.3f} / {achieved_shift * 1000.0:+.3f} mm"
    )
    print(f"Peak |CoM_y tracking error|: {peak_tracking_error * 1000.0:.4f} mm")
    print(f"Final CoM_y return error: {final_return_error * 1000.0:.4f} mm")
    print(
        "Max left/right foot translation error: "
        f"{supports['max_left_translation_error_m'] * 1000.0:.4f} / "
        f"{supports['max_right_translation_error_m'] * 1000.0:.4f} mm"
    )
    print(
        "Max left/right foot rotation error: "
        f"{np.rad2deg(supports['max_left_rotation_error_rad']):.5f} / "
        f"{np.rad2deg(supports['max_right_rotation_error_rad']):.5f} deg"
    )
    print(
        "Minimum lower/upper joint-limit margin: "
        f"{limits['minimum_lower_margin_rad']:.5f} / "
        f"{limits['minimum_upper_margin_rad']:.5f} rad"
    )

    # Do not export a reference unless it passes all gates.
    assess_integrity(
        direction_name,
        accepted_convergence,
        achieved_shift,
        peak_tracking_error,
        final_return_error,
        supports,
        limits,
    )

    if q_ref.shape != (HORIZON_NODES + 1, 6):
        raise RuntimeError(f"q_ref shape {q_ref.shape}; expected {(HORIZON_NODES + 1, 6)}.")
    if dq_ref.shape != (HORIZON_NODES + 1, 6):
        raise RuntimeError(f"dq_ref shape {dq_ref.shape}; expected {(HORIZON_NODES + 1, 6)}.")

    npz_path = TRAJECTORY_DIR / f"com_shift_{direction_name}_ref.npz"
    yaml_path = TRAJECTORY_DIR / f"com_shift_{direction_name}_ref_summary.yaml"

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
        warmstart_source=np.asarray(str(warmstart_path)),
    )

    summary = {
        "stage": "7C_double_support_fddp_com_shift_from_warmstart",
        "task": f"contact_stabilised_rigid_double_support_{direction_name}_com_shift_and_return",
        "purpose": (
            "Solve a non-static double-support FDDP trajectory from a "
            "contact-consistent kinematic warm-start."
        ),
        "coordinate_convention": {"forward_axis": "+Y", "pitch_axis": "+X"},
        "model": {
            "urdf": str(URDF_PATH),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "ndx": int(state.ndx),
            "nu": int(nu),
        },
        "warmstart": {
            "source": str(warmstart_path),
            "xs_init_shape": [int(v) for v in xs_init.shape],
            "us_init_shape": [int(v) for v in us_init.shape],
        },
        "solver": {
            "name": "SolverFDDP",
            "max_iterations": int(MAX_ITER),
            "return_success": bool(solver_return_success),
            "stopping_criterion_passed": bool(stopping_criterion_passed),
            "accepted_convergence": bool(accepted_convergence),
            "stop_criterion": float(solver.stop),
            "threshold": float(solver.th_stop),
            "initial_cost": float(initial_cost),
            "final_cost": float(final_cost),
        },
        "contacts": {
            "left_frame": LEFT_FOOT_FRAME,
            "right_frame": RIGHT_FOOT_FRAME,
            "reference_mode": "fixed_world_pose_at_x0",
            "baumgarte_gains_Kp_Kd": [float(value) for value in CONTACT_GAINS],
            **supports,
        },
        "discretisation": {
            "dt_s": float(DT),
            "horizon_nodes": int(HORIZON_NODES),
            "horizon_s": float(HORIZON_NODES * DT),
        },
        "com_motion": {
            "requested_peak_shift_y_m": float(requested_peak_shift),
            "achieved_peak_shift_y_m": float(achieved_shift),
            "peak_abs_y_tracking_error_m": float(peak_tracking_error),
            "final_return_error_y_m": float(final_return_error),
        },
        "joint_limits": limits,
        "outputs": {
            "trajectory_npz": str(npz_path),
            "q_ref_shape": [int(v) for v in q_ref.shape],
            "dq_ref_shape": [int(v) for v in dq_ref.shape],
            "tau_ref_shape": [int(v) for v in us.shape],
            "state_ref_shape": [int(v) for v in xs.shape],
        },
    }
    yaml_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")

    print("FDDP integrity: PASSED")
    print(f"Saved:\n  {npz_path}\n  {yaml_path}")


def main():
    print("===== STAGE 7C: FDDP CoM SHIFT FROM CONTACT-CONSISTENT WARM-START =====")
    print("Task: rigid double-support CoM shift and return")
    print("This is still NOT walking; both feet remain active 6D contacts.")
    print(f"URDF: {URDF_PATH}")
    print(f"Horizon: {HORIZON_NODES} nodes x {DT:.3f} s = {HORIZON_NODES * DT:.3f} s")
    print(f"Contact Baumgarte gains [Kp, Kd]: {CONTACT_GAINS.tolist()}")

    if not URDF_PATH.exists():
        raise FileNotFoundError(f"URDF not found: {URDF_PATH}")

    model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
    if model.nq != 13 or model.nv != 12:
        raise RuntimeError(
            f"Unexpected model dimensions nq={model.nq}, nv={model.nv}; "
            "expected six-pitch free-flyer model nq=13, nv=12."
        )

    x0 = load_npz_vector(X0_PATH, model.nq + model.nv)
    left_frame_id = model.getFrameId(LEFT_FOOT_FRAME)
    right_frame_id = model.getFrameId(RIGHT_FOOT_FRAME)
    if left_frame_id >= model.nframes or right_frame_id >= model.nframes:
        raise RuntimeError(
            f"Invalid support frame IDs: left={left_frame_id}, right={right_frame_id}."
        )

    left_reference, right_reference = get_support_references(
        model, x0, left_frame_id, right_frame_id
    )

    solve_direction(
        "forward",
        +0.006,
        model,
        x0,
        left_frame_id,
        right_frame_id,
        left_reference,
        right_reference,
    )
    solve_direction(
        "backward",
        -0.006,
        model,
        x0,
        left_frame_id,
        right_frame_id,
        left_reference,
        right_reference,
    )

    print()
    print("STAGE 7C FDDP FROM WARM-START PASSED")
    print("Next stage: inspect and replay the validated q_ref/dq_ref in MuJoCo.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nSTAGE 7C FDDP FROM WARM-START FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
