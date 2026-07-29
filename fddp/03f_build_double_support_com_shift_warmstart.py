#!/usr/bin/env python3
"""
STAGE 7C-WARMSTART — Contact-consistent kinematic warm-start generation.

This script does NOT run FDDP. It builds a non-static initial state trajectory
for a later FDDP solve. The trajectory:
  - keeps both 6D support-foot velocities zero to first order;
  - follows a smooth +Y/-Y CoM schedule using the contact-nullspace;
  - stores full free-flyer state guesses xs_init and zero control guesses us_init;
  - validates CoM tracking, foot translation/orientation drift, and joint limits.

It generates:
  transfer/trajectories/com_shift_forward_warmstart.npz
  transfer/trajectories/com_shift_forward_warmstart_summary.yaml
  transfer/trajectories/com_shift_backward_warmstart.npz
  transfer/trajectories/com_shift_backward_warmstart_summary.yaml

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


# -------------------------- Conservative configuration -----------------------
DT = 0.010
HORIZON_NODES = 120                 # 1.20 s
COM_SHIFT_M = 0.006                 # +/- 6 mm
LEFT_FOOT_FRAME = "left_foot_contact_fixed"
RIGHT_FOOT_FRAME = "right_foot_contact_fixed"

# Validation gates for the *kinematic* warm-start. These are intentionally
# tighter than the old FDDP validation, because this is a seed trajectory.
MAX_COM_Y_TRACKING_ERROR_M = 1.0e-3         # 1.0 mm
MAX_FINAL_COM_Y_RETURN_ERROR_M = 2.0e-4     # 0.2 mm
MAX_FOOT_TRANSLATION_DRIFT_M = 2.0e-4       # 0.2 mm
MAX_FOOT_ROTATION_DRIFT_RAD = np.deg2rad(0.20)
MIN_JOINT_LIMIT_MARGIN_RAD = 1.0e-2         # 0.57 deg
NULLSPACE_AUTHORITY_EPS = 1.0e-8

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = PROJECT_ROOT / "model" / "biped_v1.urdf"
X0_PATH = PROJECT_ROOT / "transfer" / "initialisation" / "fddp_double_support_x0.npz"
OUTPUT_DIR = PROJECT_ROOT / "transfer" / "trajectories"


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


def smoothstep(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return 3.0 * s * s - 2.0 * s * s * s


def smoothstep_derivative(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return 6.0 * s * (1.0 - s)


def make_com_schedule(initial_y: float, signed_shift_m: float):
    """
    Construct neutral -> shifted -> neutral target and target velocity.

    0.00–0.20 s : hold
    0.20–0.50 s : smooth forward/backward transfer
    0.50–0.80 s : hold
    0.80–1.10 s : smooth return
    1.10–1.20 s : hold
    """
    t = np.arange(HORIZON_NODES + 1, dtype=float) * DT
    target_y = np.full(HORIZON_NODES + 1, initial_y, dtype=float)
    target_dy = np.zeros(HORIZON_NODES + 1, dtype=float)

    ramp_duration = 0.30
    for k, tk in enumerate(t):
        if 0.20 <= tk < 0.50:
            s = (tk - 0.20) / ramp_duration
            target_y[k] = initial_y + signed_shift_m * smoothstep(s)
            target_dy[k] = signed_shift_m * smoothstep_derivative(s) / ramp_duration
        elif 0.50 <= tk < 0.80:
            target_y[k] = initial_y + signed_shift_m
        elif 0.80 <= tk < 1.10:
            s = (tk - 0.80) / ramp_duration
            target_y[k] = initial_y + signed_shift_m * (1.0 - smoothstep(s))
            target_dy[k] = -signed_shift_m * smoothstep_derivative(s) / ramp_duration

    return t, target_y, target_dy


def contact_nullspace(contact_jacobian: np.ndarray):
    """Return rank, tolerance, singular values, and right nullspace basis."""
    _, singular_values, vh = np.linalg.svd(contact_jacobian, full_matrices=True)
    tolerance = (
        max(contact_jacobian.shape)
        * np.finfo(float).eps
        * float(singular_values[0])
    )
    rank = int(np.sum(singular_values > tolerance))
    basis = vh[rank:, :].T.copy()
    return rank, float(tolerance), singular_values, basis


def compute_kinematics(
    model: pin.Model,
    q: np.ndarray,
    left_frame_id: int,
    right_frame_id: int,
):
    """
    Return CoM, two 6D local-world-aligned contact Jacobians, and support poses.
    """
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    left_jacobian = np.asarray(
        pin.computeFrameJacobian(
            model, data, q, left_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        ),
        dtype=float,
    )
    right_jacobian = np.asarray(
        pin.computeFrameJacobian(
            model, data, q, right_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        ),
        dtype=float,
    )
    contact_jacobian = np.vstack([left_jacobian, right_jacobian])

    com = np.asarray(pin.centerOfMass(model, data, q), dtype=float).reshape(3)
    jcom_result = pin.jacobianCenterOfMass(model, data, q)
    if jcom_result is None:
        jcom_result = data.Jcom
    com_jacobian = np.asarray(jcom_result, dtype=float)

    return {
        "com": com,
        "contact_jacobian": contact_jacobian,
        "com_jacobian": com_jacobian,
        "left_pose": data.oMf[left_frame_id].copy(),
        "right_pose": data.oMf[right_frame_id].copy(),
    }


def contact_tangent_velocity(
    contact_jacobian: np.ndarray,
    com_y_jacobian: np.ndarray,
    desired_com_y_rate: float,
):
    """
    Compute the least-norm velocity in null(J_contact) that realizes the
    requested first-order CoM_y velocity.

    When desired rate is zero, return exactly zero. This preserves the final
    hold periods and avoids unnecessary numerical motion.
    """
    if abs(desired_com_y_rate) < 1e-14:
        return np.zeros(contact_jacobian.shape[1], dtype=float), 0, 0.0

    rank, _, _, nullspace = contact_nullspace(contact_jacobian)
    if nullspace.shape[1] == 0:
        raise RuntimeError("Two-foot contact Jacobian has no velocity nullspace.")

    authority_row = com_y_jacobian.reshape(1, -1) @ nullspace
    authority_norm = float(np.linalg.norm(authority_row))
    if authority_norm <= NULLSPACE_AUTHORITY_EPS:
        raise RuntimeError(
            "CoM_y has negligible authority in the two-foot contact nullspace."
        )

    a = authority_row.reshape(-1)
    z = a * (desired_com_y_rate / float(a @ a))
    velocity = nullspace @ z
    return velocity, rank, authority_norm


def pose_errors(reference: pin.SE3, current: pin.SE3):
    """Return translation norm [m] and rotation geodesic angle [rad]."""
    translation = float(np.linalg.norm(current.translation - reference.translation))
    rotation = float(np.linalg.norm(pin.log3(reference.rotation.T @ current.rotation)))
    return translation, rotation


def joint_limit_summary(model: pin.Model, q_sequence: np.ndarray):
    q_joint = q_sequence[:, 7:model.nq]
    lower = np.asarray(model.lowerPositionLimit[7:model.nq], dtype=float)
    upper = np.asarray(model.upperPositionLimit[7:model.nq], dtype=float)

    lower_margin = q_joint - lower[None, :]
    upper_margin = upper[None, :] - q_joint
    minimum_lower = float(np.min(lower_margin))
    minimum_upper = float(np.min(upper_margin))
    minimum = min(minimum_lower, minimum_upper)

    return {
        "minimum_lower_margin_rad": minimum_lower,
        "minimum_upper_margin_rad": minimum_upper,
        "minimum_margin_rad": float(minimum),
        "within_position_limits": bool(
            np.all(lower_margin >= -1e-10) and np.all(upper_margin >= -1e-10)
        ),
        "has_safety_margin": bool(minimum >= MIN_JOINT_LIMIT_MARGIN_RAD),
    }


def build_one_warmstart(
    direction_name: str,
    signed_shift_m: float,
    model: pin.Model,
    x0: np.ndarray,
    left_frame_id: int,
    right_frame_id: int,
):
    """
    Integrate a contact-nullspace velocity field along the desired CoM schedule.
    This creates a contact-consistent kinematic seed, not a dynamics solution.
    """
    q0 = x0[:model.nq].copy()
    initial = compute_kinematics(model, q0, left_frame_id, right_frame_id)
    initial_com = initial["com"].copy()
    left_reference = initial["left_pose"]
    right_reference = initial["right_pose"]

    t_ref, com_y_target, com_y_rate_target = make_com_schedule(
        initial_com[1], signed_shift_m
    )

    q_sequence = np.zeros((HORIZON_NODES + 1, model.nq), dtype=float)
    v_sequence = np.zeros((HORIZON_NODES + 1, model.nv), dtype=float)
    com_sequence = np.zeros((HORIZON_NODES + 1, 3), dtype=float)

    q_sequence[0] = q0
    com_sequence[0] = initial_com

    rank_history = []
    authority_history = []
    contact_velocity_residual_history = []

    q_current = q0.copy()
    for k in range(HORIZON_NODES):
        kin = compute_kinematics(model, q_current, left_frame_id, right_frame_id)
        v_current, rank, authority = contact_tangent_velocity(
            kin["contact_jacobian"],
            kin["com_jacobian"][1, :],
            com_y_rate_target[k],
        )

        residual = float(np.linalg.norm(kin["contact_jacobian"] @ v_current))
        q_next = pin.integrate(model, q_current, v_current * DT)

        q_sequence[k] = q_current
        v_sequence[k] = v_current
        com_sequence[k] = kin["com"]
        rank_history.append(int(rank))
        authority_history.append(float(authority))
        contact_velocity_residual_history.append(residual)

        q_current = q_next

    final_kin = compute_kinematics(model, q_current, left_frame_id, right_frame_id)
    q_sequence[-1] = q_current
    v_sequence[-1] = np.zeros(model.nv, dtype=float)
    com_sequence[-1] = final_kin["com"]

    left_translation_errors = []
    right_translation_errors = []
    left_rotation_errors = []
    right_rotation_errors = []
    for q in q_sequence:
        kin = compute_kinematics(model, q, left_frame_id, right_frame_id)
        left_t, left_r = pose_errors(left_reference, kin["left_pose"])
        right_t, right_r = pose_errors(right_reference, kin["right_pose"])
        left_translation_errors.append(left_t)
        right_translation_errors.append(right_t)
        left_rotation_errors.append(left_r)
        right_rotation_errors.append(right_r)

    max_left_translation = float(max(left_translation_errors))
    max_right_translation = float(max(right_translation_errors))
    max_left_rotation = float(max(left_rotation_errors))
    max_right_rotation = float(max(right_rotation_errors))

    if signed_shift_m > 0.0:
        achieved_shift = float(np.max(com_sequence[:, 1]) - initial_com[1])
    else:
        achieved_shift = float(np.min(com_sequence[:, 1]) - initial_com[1])

    peak_com_tracking_error = float(
        np.max(np.abs(com_sequence[:, 1] - com_y_target))
    )
    final_return_error = float(abs(com_sequence[-1, 1] - initial_com[1]))
    limits = joint_limit_summary(model, q_sequence)

    # Crocoddyl state layout is [q, v].
    xs_init = np.hstack([q_sequence, v_sequence])
    us_init = np.zeros((HORIZON_NODES, 6), dtype=float)
    q_ref = q_sequence[:, 7:model.nq]
    dq_ref = v_sequence[:, 6:]
    phase = np.full(HORIZON_NODES + 1, "double_support", dtype="U16")

    if xs_init.shape != (HORIZON_NODES + 1, model.nq + model.nv):
        raise RuntimeError(f"Unexpected xs_init shape: {xs_init.shape}.")
    if us_init.shape != (HORIZON_NODES, 6):
        raise RuntimeError(f"Unexpected us_init shape: {us_init.shape}.")
    if q_ref.shape != (HORIZON_NODES + 1, 6):
        raise RuntimeError(f"Unexpected q_ref shape: {q_ref.shape}.")
    if dq_ref.shape != (HORIZON_NODES + 1, 6):
        raise RuntimeError(f"Unexpected dq_ref shape: {dq_ref.shape}.")

    failures = []
    if abs(achieved_shift) < 0.003:
        failures.append(
            f"achieved CoM shift only {achieved_shift * 1000.0:.3f} mm"
        )
    if peak_com_tracking_error > MAX_COM_Y_TRACKING_ERROR_M:
        failures.append(
            f"peak CoM_y tracking error {peak_com_tracking_error * 1000.0:.3f} mm"
        )
    if final_return_error > MAX_FINAL_COM_Y_RETURN_ERROR_M:
        failures.append(
            f"final CoM_y return error {final_return_error * 1000.0:.3f} mm"
        )
    if max_left_translation > MAX_FOOT_TRANSLATION_DRIFT_M:
        failures.append(
            f"left foot translation drift {max_left_translation * 1000.0:.3f} mm"
        )
    if max_right_translation > MAX_FOOT_TRANSLATION_DRIFT_M:
        failures.append(
            f"right foot translation drift {max_right_translation * 1000.0:.3f} mm"
        )
    if max_left_rotation > MAX_FOOT_ROTATION_DRIFT_RAD:
        failures.append(
            f"left foot rotation drift {np.rad2deg(max_left_rotation):.4f} deg"
        )
    if max_right_rotation > MAX_FOOT_ROTATION_DRIFT_RAD:
        failures.append(
            f"right foot rotation drift {np.rad2deg(max_right_rotation):.4f} deg"
        )
    if not limits["within_position_limits"]:
        failures.append("joint position-limit violation")
    elif not limits["has_safety_margin"]:
        failures.append(
            f"joint-limit safety margin {limits['minimum_margin_rad']:.5f} rad"
        )

    print()
    print(f"----- {direction_name.upper()} KINEMATIC WARM-START -----")
    print(
        "Requested / achieved peak CoM_y shift: "
        f"{signed_shift_m * 1000.0:+.3f} / {achieved_shift * 1000.0:+.3f} mm"
    )
    print(f"Peak |CoM_y tracking error|: {peak_com_tracking_error * 1000.0:.4f} mm")
    print(f"Final CoM_y return error: {final_return_error * 1000.0:.4f} mm")
    print(
        "Max left/right foot translation drift: "
        f"{max_left_translation * 1000.0:.4f} / "
        f"{max_right_translation * 1000.0:.4f} mm"
    )
    print(
        "Max left/right foot rotation drift: "
        f"{np.rad2deg(max_left_rotation):.5f} / "
        f"{np.rad2deg(max_right_rotation):.5f} deg"
    )
    print(
        "Contact Jacobian rank range: "
        f"{min(rank_history)} to {max(rank_history)}"
    )
    print(
        "CoM_y nullspace authority range: "
        f"{min(authority_history):.6e} to {max(authority_history):.6e}"
    )
    print(
        "Max linearised contact-velocity residual: "
        f"{max(contact_velocity_residual_history):.3e}"
    )
    print(
        "Minimum lower/upper joint-limit margin: "
        f"{limits['minimum_lower_margin_rad']:.5f} / "
        f"{limits['minimum_upper_margin_rad']:.5f} rad"
    )

    if failures:
        print("Warm-start integrity: FAILED")
        for item in failures:
            print(f"  - {item}")
        raise RuntimeError(
            f"{direction_name} warm-start rejected; no files exported."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"com_shift_{direction_name}_warmstart"
    npz_path = OUTPUT_DIR / f"{stem}.npz"
    yaml_path = OUTPUT_DIR / f"{stem}_summary.yaml"

    np.savez(
        npz_path,
        t_ref=t_ref,
        q_ref=q_ref,
        dq_ref=dq_ref,
        xs_init=xs_init,
        us_init=us_init,
        q_full=q_sequence,
        v_full=v_sequence,
        com_ref_world=com_sequence,
        com_y_target=com_y_target,
        com_y_rate_target=com_y_rate_target,
        phase=phase,
        direction=np.asarray(direction_name),
        contact_mode=np.asarray("double_support"),
        left_contact_active=np.ones(HORIZON_NODES + 1, dtype=np.uint8),
        right_contact_active=np.ones(HORIZON_NODES + 1, dtype=np.uint8),
    )

    summary = {
        "stage": "7C_contact_consistent_kinematic_warmstart",
        "task": f"double_support_{direction_name}_com_shift_and_return_warmstart",
        "purpose": (
            "Provide SolverFDDP with a non-static, contact-nullspace-consistent "
            "initial state trajectory instead of repeating the static x0 state."
        ),
        "coordinate_convention": {"forward_axis": "+Y", "pitch_axis": "+X"},
        "model": {
            "urdf": str(URDF_PATH),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": 6,
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
            "peak_abs_y_tracking_error_m": float(peak_com_tracking_error),
            "final_return_error_y_m": float(final_return_error),
        },
        "contacts": {
            "left_frame": LEFT_FOOT_FRAME,
            "right_frame": RIGHT_FOOT_FRAME,
            "construction_method": (
                "integrate least-norm CoM_y velocity inside the instantaneous "
                "two-foot 6D contact-Jacobian nullspace"
            ),
            "contact_jacobian_rank_min": int(min(rank_history)),
            "contact_jacobian_rank_max": int(max(rank_history)),
            "com_y_authority_min": float(min(authority_history)),
            "com_y_authority_max": float(max(authority_history)),
            "max_linearised_contact_velocity_residual": float(
                max(contact_velocity_residual_history)
            ),
            "max_left_translation_drift_m": float(max_left_translation),
            "max_right_translation_drift_m": float(max_right_translation),
            "max_left_rotation_drift_rad": float(max_left_rotation),
            "max_right_rotation_drift_rad": float(max_right_rotation),
        },
        "joint_limits": limits,
        "outputs": {
            "warmstart_npz": str(npz_path),
            "xs_init_shape": [int(v) for v in xs_init.shape],
            "us_init_shape": [int(v) for v in us_init.shape],
            "q_ref_shape": [int(v) for v in q_ref.shape],
            "dq_ref_shape": [int(v) for v in dq_ref.shape],
        },
    }
    yaml_path.write_text(
        yaml.safe_dump(summary, sort_keys=False),
        encoding="utf-8",
    )

    print("Warm-start integrity: PASSED")
    print(f"Saved:\n  {npz_path}\n  {yaml_path}")


def main() -> None:
    print("===== STAGE 7C: CONTACT-CONSISTENT KINEMATIC WARM-START =====")
    print("This stage does not run FDDP and does not overwrite FDDP references.")
    print(f"URDF: {URDF_PATH}")
    print(f"Horizon: {HORIZON_NODES} nodes x {DT:.3f} s = {HORIZON_NODES * DT:.3f} s")
    print(f"Requested CoM shift magnitude: {COM_SHIFT_M * 1000.0:.1f} mm")

    if not URDF_PATH.exists():
        raise FileNotFoundError(f"URDF not found: {URDF_PATH}")

    model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
    if model.nq != 13 or model.nv != 12:
        raise RuntimeError(
            f"Unexpected model dimensions nq={model.nq}, nv={model.nv}; "
            "expected six-pitch free-flyer model nq=13, nv=12."
        )

    x0 = load_x0(X0_PATH, model.nq + model.nv)
    left_frame_id = model.getFrameId(LEFT_FOOT_FRAME)
    right_frame_id = model.getFrameId(RIGHT_FOOT_FRAME)

    if left_frame_id >= model.nframes or right_frame_id >= model.nframes:
        raise RuntimeError(
            f"Invalid support frame IDs: left={left_frame_id}, right={right_frame_id}."
        )

    build_one_warmstart(
        "forward", +COM_SHIFT_M, model, x0, left_frame_id, right_frame_id
    )
    build_one_warmstart(
        "backward", -COM_SHIFT_M, model, x0, left_frame_id, right_frame_id
    )

    print()
    print("CONTACT-CONSISTENT KINEMATIC WARM-START PASSED")
    print("Next stage: use these xs_init/us_init arrays as the FDDP initial guess.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nKINEMATIC WARM-START FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
