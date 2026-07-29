#!/usr/bin/env python3
"""
STAGE 7C-REVISED — 3D-contact FDDP continuation for double-support CoM shift.

Why this replaces the previous 6D/quasiStatic route:
  - The kinematic warm-start is valid but has large dynamics shooting gaps.
  - quasiStatic generates equilibrium inputs, not the acceleration-consistent
    input sequence required for a moving CoM transfer.
  - Two 6D bilateral contacts were rank-deficient (rank 8 from 12 rows) and
    numerically fragile for this planar six-pitch model.

This script uses:
  - two translational ContactModel3D support contacts;
  - standard FDDP continuation / homotopy: solve 0 mm first, then grow the
    target magnitude in small increments, using the previous FDDP solution as
    the next seed;
  - a posteriori checks for translation, foot-pitch rotation, CoM tracking,
    joint limits, and FDDP convergence.

This remains an OFFLINE nominal-trajectory planner. It is not walking and it
does not command ST3215 torque. Exported q_ref/dq_ref are for later MuJoCo/MPC
reference tracking.
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import yaml
import pinocchio as pin
import crocoddyl


# ---------------------- Geometry / continuation configuration ----------------
DT = 0.010
HORIZON_NODES = 120                 # 1.20 s
MAX_ITER_PER_STAGE = 100

LEFT_FOOT_FRAME = "left_foot_contact_fixed"
RIGHT_FOOT_FRAME = "right_foot_contact_fixed"

# FDDP continuation endpoints in metres. The final target is +/-6 mm.
HOMOTOPY_MAGNITUDES_M = [
    0.0000,
    0.0005,
    0.0010,
    0.0020,
    0.0030,
    0.0040,
    0.0050,
    0.0060,
]

# Crocoddyl uses direct Baumgarte gains. The official gait examples use a
# velocity stabilisation term in ContactModel3D; retain that conservative form.
CONTACT_GAINS_KP_KD = np.array([0.0, 50.0], dtype=float)

RUNNING_STATE_WEIGHT = 0.20
RUNNING_CONTROL_WEIGHT = 1e-3
TERMINAL_STATE_WEIGHT = 10.0
COM_TRACKING_WEIGHT_Y = 1.0e4
COM_TRACKING_WEIGHT_XZ = 2.0e2

# Final trajectory quality gates.
MAX_SUPPORT_TRANSLATION_ERROR_M = 1.0e-3      # 1.0 mm
MAX_SUPPORT_ROTATION_ERROR_RAD = np.deg2rad(2.0)
MIN_JOINT_LIMIT_MARGIN_RAD = 1.0e-2           # 0.57 deg
MIN_ACHIEVED_SHIFT_M = 0.003                   # 3 mm
MAX_COM_Y_TRACKING_ERROR_M = 2.0e-3           # 2 mm
MAX_FINAL_COM_Y_RETURN_ERROR_M = 1.0e-3        # 1 mm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = PROJECT_ROOT / "model" / "biped_v1.urdf"
X0_PATH = PROJECT_ROOT / "transfer" / "initialisation" / "fddp_double_support_x0.npz"
OUTPUT_DIR = PROJECT_ROOT / "transfer" / "trajectories"


def evaluate_total_cost(problem, xs, us):
    """Crocoddyl-version-compatible total cost calculation."""
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
                f"No x0-like key in {path}; available keys: {loaded.files}"
            )

    if x0.size != expected_size:
        raise RuntimeError(
            f"x0 size {x0.size}, but expected {expected_size}."
        )
    return x0


def smoothstep(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return 3.0 * s * s - 2.0 * s * s * s


def make_com_y_schedule(initial_y: float, signed_shift_m: float):
    """
    0.00–0.20 s: neutral
    0.20–0.50 s: smooth transfer
    0.50–0.80 s: hold
    0.80–1.10 s: smooth return
    1.10–1.20 s: neutral
    """
    t = np.arange(HORIZON_NODES + 1, dtype=float) * DT
    result = np.full(HORIZON_NODES + 1, initial_y, dtype=float)
    for k, tk in enumerate(t):
        if 0.20 <= tk < 0.50:
            result[k] = initial_y + signed_shift_m * smoothstep((tk - 0.20) / 0.30)
        elif 0.50 <= tk < 0.80:
            result[k] = initial_y + signed_shift_m
        elif 0.80 <= tk < 1.10:
            result[k] = initial_y + signed_shift_m * (
                1.0 - smoothstep((tk - 0.80) / 0.30)
            )
    return t, result


def get_support_references(model, x0, left_frame_id, right_frame_id):
    data = model.createData()
    pin.forwardKinematics(model, data, x0[:model.nq])
    pin.updateFramePlacements(model, data)
    return (
        data.oMf[left_frame_id].translation.copy(),
        data.oMf[right_frame_id].translation.copy(),
        data.oMf[left_frame_id].rotation.copy(),
        data.oMf[right_frame_id].rotation.copy(),
    )


def make_contact_model_3d(state, nu, frame_id, position_reference):
    """
    3D translational support contact in LOCAL_WORLD_ALIGNED coordinates.
    This hard-constrains world translation but not orientation.
    """
    return crocoddyl.ContactModel3D(
        state,
        frame_id,
        position_reference,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        nu,
        CONTACT_GAINS_KP_KD,
    )


def make_action_model(
    state,
    actuation,
    x_reference,
    com_target,
    left_frame_id,
    right_frame_id,
    left_position_reference,
    right_position_reference,
    state_weight,
    control_weight,
    dt,
):
    nu = actuation.nu

    contacts = crocoddyl.ContactModelMultiple(state, nu)
    contacts.addContact(
        LEFT_FOOT_FRAME,
        make_contact_model_3d(
            state, nu, left_frame_id, left_position_reference
        ),
    )
    contacts.addContact(
        RIGHT_FOOT_FRAME,
        make_contact_model_3d(
            state, nu, right_frame_id, right_position_reference
        ),
    )

    costs = crocoddyl.CostModelSum(state, nu)

    # This keeps the root orientation and joint postures near the stable stance,
    # indirectly regularising foot pitch while translational contacts hold feet.
    state_residual = crocoddyl.ResidualModelState(state, x_reference, nu)
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


def build_problem(
    model,
    x0,
    com_y_target,
    left_frame_id,
    right_frame_id,
    left_position_reference,
    right_position_reference,
):
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    initial_com = compute_com(model, x0[None, :])[0]

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
                left_position_reference,
                right_position_reference,
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
        left_position_reference,
        right_position_reference,
        TERMINAL_STATE_WEIGHT,
        0.0,
        0.0,
    )

    return crocoddyl.ShootingProblem(x0, running_models, terminal_model), state


def compute_com(model, xs):
    data = model.createData()
    output = np.zeros((xs.shape[0], 3), dtype=float)
    for k, x in enumerate(xs):
        output[k] = np.asarray(
            pin.centerOfMass(model, data, x[:model.nq]),
            dtype=float,
        ).reshape(3)
    return output


def pose_error(reference_rotation, reference_translation, current):
    translation_error = float(
        np.linalg.norm(current.translation - reference_translation)
    )
    rotation_error = float(
        np.linalg.norm(pin.log3(reference_rotation.T @ current.rotation))
    )
    return translation_error, rotation_error


def compute_support_errors(
    model,
    xs,
    left_frame_id,
    right_frame_id,
    left_position_reference,
    right_position_reference,
    left_rotation_reference,
    right_rotation_reference,
):
    data = model.createData()
    left_translation_max = 0.0
    right_translation_max = 0.0
    left_rotation_max = 0.0
    right_rotation_max = 0.0

    for x in xs:
        pin.forwardKinematics(model, data, x[:model.nq])
        pin.updateFramePlacements(model, data)

        left_t, left_r = pose_error(
            left_rotation_reference,
            left_position_reference,
            data.oMf[left_frame_id],
        )
        right_t, right_r = pose_error(
            right_rotation_reference,
            right_position_reference,
            data.oMf[right_frame_id],
        )
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


def joint_limit_summary(model, xs):
    joints = xs[:, 7:model.nq]
    lower = np.asarray(model.lowerPositionLimit[7:model.nq], dtype=float)
    upper = np.asarray(model.upperPositionLimit[7:model.nq], dtype=float)

    lower_margin = joints - lower[None, :]
    upper_margin = upper[None, :] - joints
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
        "has_safety_margin": bool(
            minimum_margin >= MIN_JOINT_LIMIT_MARGIN_RAD
        ),
    }


def solve_one_direction(
    direction,
    sign,
    model,
    x0,
    left_frame_id,
    right_frame_id,
    left_position_reference,
    right_position_reference,
    left_rotation_reference,
    right_rotation_reference,
):
    """
    Continuation: each solved amplitude becomes the next amplitude's seed.
    The first 0 mm task starts from the known static state and zero controls.
    """
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    nu = actuation.nu

    static_xs = [x0.copy() for _ in range(HORIZON_NODES + 1)]
    static_us = [np.zeros(nu, dtype=float) for _ in range(HORIZON_NODES)]
    xs_seed = static_xs
    us_seed = static_us
    stage_records = []

    initial_com_y = float(compute_com(model, x0[None, :])[0, 1])
    final_t_ref = None
    final_target = None
    final_problem = None
    final_solver = None

    for magnitude in HOMOTOPY_MAGNITUDES_M:
        signed_shift = float(sign * magnitude)
        t_ref, com_y_target = make_com_y_schedule(initial_com_y, signed_shift)

        problem, _ = build_problem(
            model,
            x0,
            com_y_target,
            left_frame_id,
            right_frame_id,
            left_position_reference,
            right_position_reference,
        )
        initial_cost = evaluate_total_cost(problem, xs_seed, us_seed)

        solver = crocoddyl.SolverFDDP(problem)
        solver.th_stop = 1e-7
        return_success = bool(
            solver.solve(xs_seed, us_seed, MAX_ITER_PER_STAGE, False)
        )
        stop_passed = bool(float(solver.stop) <= float(solver.th_stop))
        accepted = bool(return_success or stop_passed)

        xs_candidate = [np.asarray(x, dtype=float).copy() for x in solver.xs]
        us_candidate = [np.asarray(u, dtype=float).copy() for u in solver.us]
        final_cost = evaluate_total_cost(problem, xs_candidate, us_candidate)

        stage_record = {
            "requested_shift_m": signed_shift,
            "return_success": return_success,
            "stop_criterion": float(solver.stop),
            "threshold": float(solver.th_stop),
            "accepted_convergence": accepted,
            "initial_cost": float(initial_cost),
            "final_cost": float(final_cost),
        }
        stage_records.append(stage_record)

        print(
            f"  {direction:8s} target {signed_shift * 1000.0:+5.2f} mm | "
            f"accepted={accepted} | stop={solver.stop:.3e} | "
            f"cost={initial_cost:.4e}->{final_cost:.4e}"
        )

        if not accepted or not np.isfinite(final_cost):
            raise RuntimeError(
                f"{direction}: continuation failed at {signed_shift * 1000.0:+.2f} mm; "
                "no reference export was written."
            )

        xs_seed = xs_candidate
        us_seed = us_candidate
        final_t_ref = t_ref
        final_target = com_y_target
        final_problem = problem
        final_solver = solver

    xs = np.asarray(xs_seed, dtype=float)
    us = np.asarray(us_seed, dtype=float)
    expected_x_shape = (HORIZON_NODES + 1, model.nq + model.nv)
    expected_u_shape = (HORIZON_NODES, nu)
    if xs.shape != expected_x_shape:
        raise RuntimeError(f"{direction}: unexpected xs shape {xs.shape}.")
    if us.shape != expected_u_shape:
        raise RuntimeError(f"{direction}: unexpected us shape {us.shape}.")

    com_ref = compute_com(model, xs)
    if sign > 0:
        achieved_shift = float(np.max(com_ref[:, 1]) - initial_com_y)
    else:
        achieved_shift = float(np.min(com_ref[:, 1]) - initial_com_y)

    tracking_error = float(np.max(np.abs(com_ref[:, 1] - final_target)))
    return_error = float(abs(com_ref[-1, 1] - initial_com_y))
    supports = compute_support_errors(
        model,
        xs,
        left_frame_id,
        right_frame_id,
        left_position_reference,
        right_position_reference,
        left_rotation_reference,
        right_rotation_reference,
    )
    limits = joint_limit_summary(model, xs)

    q_ref = xs[:, 7:model.nq]
    dq_ref = xs[:, model.nq + 6:]
    phase = np.full(HORIZON_NODES + 1, "double_support", dtype="U16")

    print()
    print(f"----- {direction.upper()} FINAL 3D-CONTACT FDDP RESULT -----")
    print(
        "Requested / achieved CoM_y shift: "
        f"{sign * HOMOTOPY_MAGNITUDES_M[-1] * 1000.0:+.3f} / "
        f"{achieved_shift * 1000.0:+.3f} mm"
    )
    print(f"Peak |CoM_y tracking error|: {tracking_error * 1000.0:.4f} mm")
    print(f"Final CoM_y return error: {return_error * 1000.0:.4f} mm")
    print(
        "Max L/R foot translation error: "
        f"{supports['max_left_translation_error_m'] * 1000.0:.4f} / "
        f"{supports['max_right_translation_error_m'] * 1000.0:.4f} mm"
    )
    print(
        "Max L/R foot rotation error: "
        f"{np.rad2deg(supports['max_left_rotation_error_rad']):.4f} / "
        f"{np.rad2deg(supports['max_right_rotation_error_rad']):.4f} deg"
    )
    print(
        "Minimum lower/upper joint-limit margin: "
        f"{limits['minimum_lower_margin_rad']:.5f} / "
        f"{limits['minimum_upper_margin_rad']:.5f} rad"
    )

    failures = []
    if abs(achieved_shift) < MIN_ACHIEVED_SHIFT_M:
        failures.append("insufficient CoM shift")
    if tracking_error > MAX_COM_Y_TRACKING_ERROR_M:
        failures.append("CoM tracking error above threshold")
    if return_error > MAX_FINAL_COM_Y_RETURN_ERROR_M:
        failures.append("CoM final return error above threshold")
    if supports["max_left_translation_error_m"] > MAX_SUPPORT_TRANSLATION_ERROR_M:
        failures.append("left foot translation error above threshold")
    if supports["max_right_translation_error_m"] > MAX_SUPPORT_TRANSLATION_ERROR_M:
        failures.append("right foot translation error above threshold")
    if supports["max_left_rotation_error_rad"] > MAX_SUPPORT_ROTATION_ERROR_RAD:
        failures.append("left foot rotation error above threshold")
    if supports["max_right_rotation_error_rad"] > MAX_SUPPORT_ROTATION_ERROR_RAD:
        failures.append("right foot rotation error above threshold")
    if not limits["within_position_limits"]:
        failures.append("joint-limit violation")
    elif not limits["has_safety_margin"]:
        failures.append("joint-limit safety margin too small")

    if failures:
        print("3D-contact FDDP integrity: FAILED")
        for item in failures:
            print(f"  - {item}")
        raise RuntimeError(
            f"{direction}: final continuation trajectory rejected; "
            "no reference export was written."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"com_shift_{direction}_ref"
    npz_path = OUTPUT_DIR / f"{stem}.npz"
    yaml_path = OUTPUT_DIR / f"{stem}_summary.yaml"

    np.savez(
        npz_path,
        t_ref=final_t_ref,
        q_ref=q_ref,
        dq_ref=dq_ref,
        tau_ref=us,
        state_ref=xs,
        com_ref_world=com_ref,
        com_y_target=final_target,
        phase=phase,
        direction=np.asarray(direction),
        contact_mode=np.asarray("double_support_3d_translation"),
        left_contact_active=np.ones(HORIZON_NODES + 1, dtype=np.uint8),
        right_contact_active=np.ones(HORIZON_NODES + 1, dtype=np.uint8),
    )

    summary = {
        "stage": "7C_revised_3d_contact_fddp_homotopy",
        "task": f"double_support_3d_translation_{direction}_com_shift_and_return",
        "purpose": (
            "Generate a non-static nominal CoM-shift reference through FDDP "
            "continuation, avoiding the redundant bilateral 6D-contact formulation."
        ),
        "coordinate_convention": {"forward_axis": "+Y", "pitch_axis": "+X"},
        "model": {
            "urdf": str(URDF_PATH),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "ndx": int(state.ndx),
            "nu": int(nu),
        },
        "contacts": {
            "type": "two ContactModel3D translational supports",
            "left_frame": LEFT_FOOT_FRAME,
            "right_frame": RIGHT_FOOT_FRAME,
            "reference_mode": "actual x0 world translations",
            "baumgarte_gains_Kp_Kd": [
                float(v) for v in CONTACT_GAINS_KP_KD
            ],
            **supports,
        },
        "continuation": {
            "magnitudes_m": [float(v) for v in HOMOTOPY_MAGNITUDES_M],
            "max_iterations_per_stage": int(MAX_ITER_PER_STAGE),
            "stages": stage_records,
        },
        "com_motion": {
            "requested_peak_shift_y_m": float(
                sign * HOMOTOPY_MAGNITUDES_M[-1]
            ),
            "achieved_peak_shift_y_m": float(achieved_shift),
            "peak_abs_y_tracking_error_m": float(tracking_error),
            "final_return_error_y_m": float(return_error),
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
    yaml_path.write_text(
        yaml.safe_dump(summary, sort_keys=False),
        encoding="utf-8",
    )

    print("3D-contact FDDP integrity: PASSED")
    print(f"Saved:\n  {npz_path}\n  {yaml_path}")


def main():
    print("===== STAGE 7C-REVISED: 3D-CONTACT FDDP CONTINUATION =====")
    print("Task: double-support CoM shift with translational support contacts.")
    print("This is not walking; it is the first non-static FDDP validation task.")
    print(f"URDF: {URDF_PATH}")
    print(
        f"Horizon: {HORIZON_NODES} nodes x {DT:.3f} s = "
        f"{HORIZON_NODES * DT:.3f} s"
    )
    print(
        "Continuation magnitudes (mm): "
        + ", ".join(f"{m * 1000.0:.1f}" for m in HOMOTOPY_MAGNITUDES_M)
    )
    print(
        f"Contact gains [Kp, Kd]: {CONTACT_GAINS_KP_KD.tolist()}"
    )

    if not URDF_PATH.exists():
        raise FileNotFoundError(f"URDF not found: {URDF_PATH}")

    model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
    if model.nq != 13 or model.nv != 12:
        raise RuntimeError(
            f"Unexpected model dimensions nq={model.nq}, nv={model.nv}; "
            "expected nq=13, nv=12."
        )

    x0 = load_x0(X0_PATH, model.nq + model.nv)
    left_frame_id = model.getFrameId(LEFT_FOOT_FRAME)
    right_frame_id = model.getFrameId(RIGHT_FOOT_FRAME)

    if left_frame_id >= model.nframes or right_frame_id >= model.nframes:
        raise RuntimeError(
            f"Invalid support frame IDs: left={left_frame_id}, "
            f"right={right_frame_id}."
        )

    (
        left_position_reference,
        right_position_reference,
        left_rotation_reference,
        right_rotation_reference,
    ) = get_support_references(
        model, x0, left_frame_id, right_frame_id
    )

    print(
        "Support positions at x0 (m): "
        f"left={left_position_reference.tolist()}, "
        f"right={right_position_reference.tolist()}"
    )

    solve_one_direction(
        "forward",
        +1.0,
        model,
        x0,
        left_frame_id,
        right_frame_id,
        left_position_reference,
        right_position_reference,
        left_rotation_reference,
        right_rotation_reference,
    )
    solve_one_direction(
        "backward",
        -1.0,
        model,
        x0,
        left_frame_id,
        right_frame_id,
        left_position_reference,
        right_position_reference,
        left_rotation_reference,
        right_rotation_reference,
    )

    print()
    print("STAGE 7C-REVISED 3D-CONTACT FDDP CONTINUATION PASSED")
    print("Next stage: replay q_ref/dq_ref in MuJoCo with the actuator model.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nSTAGE 7C-REVISED FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
