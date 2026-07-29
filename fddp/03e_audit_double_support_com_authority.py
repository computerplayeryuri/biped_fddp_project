#!/usr/bin/env python3
"""
STAGE 7C DIAGNOSTIC-2 — Double-support contact-manifold CoM authority audit.

This script does NOT solve FDDP and does NOT overwrite trajectories.

It answers a precise question:
    While both 6D foot contacts are held locally fixed, does the remaining
    contact-nullspace contain a direction that can move CoM_y (the project's
    forward axis)?

Outputs:
  - stacked two-foot 6D contact Jacobian rank/nullspace;
  - projection of the CoM_y Jacobian into that nullspace;
  - a least-norm tangent velocity that produces +10 mm/s CoM_y motion while
    satisfying the linearised two-foot contact constraints;
  - a small Pinocchio integration check at dt=0.01 s.

Coordinate convention:
  +X = right, +Y = forward, +Z = up
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import yaml
import pinocchio as pin


LEFT_FOOT_FRAME = "left_foot_contact_fixed"
RIGHT_FOOT_FRAME = "right_foot_contact_fixed"
TEST_COM_Y_RATE_MPS = 0.010  # +10 mm/s
TEST_DT_S = 0.010
AUTHORITY_EPS = 1e-8

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = PROJECT_ROOT / "model" / "biped_v1.urdf"
X0_PATH = PROJECT_ROOT / "transfer" / "initialisation" / "fddp_double_support_x0.npz"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "transfer"
    / "initialisation"
    / "double_support_com_authority_audit.yaml"
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


def svd_rank_nullspace(matrix: np.ndarray) -> tuple[int, float, np.ndarray, np.ndarray]:
    """Return numerical rank, tolerance, singular values, and right nullspace."""
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    if singular_values.size == 0:
        return 0, 0.0, singular_values, np.eye(matrix.shape[1])

    tolerance = (
        max(matrix.shape)
        * np.finfo(float).eps
        * float(singular_values[0])
    )
    rank = int(np.sum(singular_values > tolerance))
    nullspace_basis = vh[rank:, :].T.copy()
    return rank, float(tolerance), singular_values, nullspace_basis


def get_com_jacobian(model: pin.Model, data: pin.Data, q: np.ndarray) -> np.ndarray:
    """
    Retrieve the 3 x nv world CoM Jacobian across Pinocchio Python variants.
    """
    result = pin.jacobianCenterOfMass(model, data, q)
    if result is None:
        result = data.Jcom
    jacobian = np.asarray(result, dtype=float)
    if jacobian.shape != (3, model.nv):
        raise RuntimeError(
            f"Unexpected CoM Jacobian shape {jacobian.shape}; "
            f"expected {(3, model.nv)}."
        )
    return jacobian


def get_frame_positions(
    model: pin.Model,
    q: np.ndarray,
    left_frame_id: int,
    right_frame_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return (
        data.oMf[left_frame_id].translation.copy(),
        data.oMf[right_frame_id].translation.copy(),
    )


def to_float_list(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=float).reshape(-1)]


def main() -> None:
    print("===== STAGE 7C DIAGNOSTIC-2: DOUBLE-SUPPORT CoM AUTHORITY AUDIT =====")
    print("This script only analyses x0; it does not solve FDDP or overwrite trajectories.")
    print(f"URDF: {URDF_PATH}")

    if not URDF_PATH.exists():
        raise FileNotFoundError(f"URDF not found: {URDF_PATH}")

    model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
    if model.nq != 13 or model.nv != 12:
        raise RuntimeError(
            f"Unexpected model dimensions nq={model.nq}, nv={model.nv}; "
            "expected the validated six-pitch free-flyer model."
        )

    x0 = load_x0(X0_PATH, model.nq + model.nv)
    q0 = x0[: model.nq]

    left_frame_id = model.getFrameId(LEFT_FOOT_FRAME)
    right_frame_id = model.getFrameId(RIGHT_FOOT_FRAME)
    if left_frame_id >= model.nframes or right_frame_id >= model.nframes:
        raise RuntimeError(
            f"Could not resolve support frames: left={left_frame_id}, right={right_frame_id}."
        )

    data = model.createData()
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)

    left_jacobian = np.asarray(
        pin.computeFrameJacobian(
            model,
            data,
            q0,
            left_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        ),
        dtype=float,
    )
    right_jacobian = np.asarray(
        pin.computeFrameJacobian(
            model,
            data,
            q0,
            right_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        ),
        dtype=float,
    )
    contact_jacobian = np.vstack([left_jacobian, right_jacobian])
    rank, tolerance, singular_values, nullspace = svd_rank_nullspace(contact_jacobian)
    nullity = int(nullspace.shape[1])

    com0 = np.asarray(pin.centerOfMass(model, data, q0), dtype=float).reshape(3)
    com_jacobian = get_com_jacobian(model, data, q0)
    com_y_jacobian = com_jacobian[1:2, :]
    projected_com_y = com_y_jacobian @ nullspace
    com_y_authority = float(np.linalg.norm(projected_com_y))

    left0, right0 = get_frame_positions(
        model, q0, left_frame_id, right_frame_id
    )

    feasible = bool(nullity > 0 and com_y_authority > AUTHORITY_EPS)
    test_result = {
        "performed": False,
        "requested_com_y_rate_mps": float(TEST_COM_Y_RATE_MPS),
    }

    if feasible:
        # Least-Euclidean-norm nullspace coefficient z satisfying:
        # (J_com_y N) z = requested CoM_y rate.
        a = projected_com_y.reshape(-1)
        z = a * (TEST_COM_Y_RATE_MPS / float(a @ a))
        v_test = nullspace @ z

        contact_velocity_residual = float(np.linalg.norm(contact_jacobian @ v_test))
        predicted_com_y_rate = float(com_y_jacobian @ v_test)

        q1 = pin.integrate(model, q0, v_test * TEST_DT_S)
        data1 = model.createData()
        com1 = np.asarray(pin.centerOfMass(model, data1, q1), dtype=float).reshape(3)
        left1, right1 = get_frame_positions(
            model, q1, left_frame_id, right_frame_id
        )

        left_drift = float(np.linalg.norm(left1 - left0))
        right_drift = float(np.linalg.norm(right1 - right0))
        actual_com_y_delta = float(com1[1] - com0[1])
        expected_com_y_delta = float(TEST_COM_Y_RATE_MPS * TEST_DT_S)

        test_result = {
            "performed": True,
            "requested_com_y_rate_mps": float(TEST_COM_Y_RATE_MPS),
            "test_dt_s": float(TEST_DT_S),
            "least_norm_tangent_velocity": to_float_list(v_test),
            "tangent_velocity_norm": float(np.linalg.norm(v_test)),
            "linearised_contact_velocity_residual": contact_velocity_residual,
            "predicted_com_y_rate_mps": predicted_com_y_rate,
            "expected_com_y_delta_m": expected_com_y_delta,
            "integrated_com_y_delta_m": actual_com_y_delta,
            "left_foot_translation_drift_m": left_drift,
            "right_foot_translation_drift_m": right_drift,
            "joint_velocity_rad_per_s": to_float_list(v_test[6:]),
        }

    if not feasible:
        conclusion = (
            "CoM_y has negligible first-order authority inside the two-foot 6D "
            "contact nullspace at x0. A pure rigid-double-support forward/backward "
            "CoM-shift trajectory is not locally available in this posture."
        )
        recommendation = (
            "Do not continue tuning FDDP gains. Use a different planning task or "
            "a deliberately reduced contact model before attempting non-static motion."
        )
    else:
        conclusion = (
            "CoM_y has non-zero first-order authority inside the two-foot 6D contact "
            "nullspace at x0. The desired double-support shift is kinematically "
            "available; the FDDP failures are therefore most likely due to poor "
            "initialisation and/or numerical contact-dynamics conditioning."
        )
        recommendation = (
            "Build a contact-consistent kinematic warm-start trajectory, then use it "
            "as SolverFDDP's initial xs/us guess instead of repeating x0 at every node."
        )

    summary = {
        "stage": "7C_contact_manifold_com_authority_audit",
        "purpose": (
            "Determine whether the remaining local degrees of freedom under two "
            "6D support contacts can generate forward CoM_y motion."
        ),
        "coordinate_convention": {"forward_axis": "+Y", "pitch_axis": "+X"},
        "model": {"urdf": str(URDF_PATH), "nq": int(model.nq), "nv": int(model.nv)},
        "two_foot_6d_contact": {
            "jacobian_shape": [int(v) for v in contact_jacobian.shape],
            "rank": int(rank),
            "nullity": int(nullity),
            "svd_tolerance": float(tolerance),
            "singular_values": to_float_list(singular_values),
        },
        "com_y_authority": {
            "authority_norm": float(com_y_authority),
            "authority_threshold": float(AUTHORITY_EPS),
            "available": bool(feasible),
            "com_y_jacobian_world_row": to_float_list(com_y_jacobian),
        },
        "small_tangent_integration_test": test_result,
        "conclusion": conclusion,
        "recommended_next_step": recommendation,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")

    print()
    print(f"Two-foot 6D contact rank / nullity: {rank} / {nullity}")
    print(f"CoM_y nullspace authority norm: {com_y_authority:.9e}")
    print(f"CoM_y authority available: {feasible}")

    if feasible:
        print()
        print("Small tangent integration test:")
        print(
            "  requested / predicted CoM_y rate (mm/s): "
            f"{TEST_COM_Y_RATE_MPS * 1000.0:+.4f} / "
            f"{test_result['predicted_com_y_rate_mps'] * 1000.0:+.4f}"
        )
        print(
            "  expected / integrated CoM_y delta over 10 ms (mm): "
            f"{test_result['expected_com_y_delta_m'] * 1000.0:+.6f} / "
            f"{test_result['integrated_com_y_delta_m'] * 1000.0:+.6f}"
        )
        print(
            "  left / right foot translation drift over 10 ms (um): "
            f"{test_result['left_foot_translation_drift_m'] * 1e6:.3f} / "
            f"{test_result['right_foot_translation_drift_m'] * 1e6:.3f}"
        )
        print(
            "  linearised two-foot contact residual: "
            f"{test_result['linearised_contact_velocity_residual']:.3e}"
        )

    print()
    print("Conclusion:")
    print(f"  {conclusion}")
    print("Recommended next step:")
    print(f"  {recommendation}")
    print()
    print(f"Saved:\n  {OUTPUT_PATH}")
    print("DOUBLE-SUPPORT CoM AUTHORITY AUDIT PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nCoM AUTHORITY AUDIT FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
