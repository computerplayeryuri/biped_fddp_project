#!/usr/bin/env python3
"""
STAGE 7C-DIAGNOSTIC — Double-support contact-rank audit.

This script does NOT solve FDDP and does NOT overwrite trajectories.
It audits the instantaneous contact-constraint rank at the validated x0
configuration.

Why this matters:
- ContactModel6D on both feet produces 12 scalar constraints.
- The free-flyer six-pitch model has nv = 12 velocity variables.
- If the stacked 6D contact Jacobian is full rank, there is no local motion
  remaining under two welded foot contacts. A CoM-shift task is then
  structurally infeasible in that model.
- A planar/contact-reduced formulation must instead be used for the next
  non-static planning stage.

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = PROJECT_ROOT / "model" / "biped_v1.urdf"
X0_PATH = PROJECT_ROOT / "transfer" / "initialisation" / "fddp_double_support_x0.npz"
OUTPUT_PATH = PROJECT_ROOT / "transfer" / "initialisation" / "double_support_contact_rank_audit.yaml"


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


def numerical_rank_and_singular_values(matrix: np.ndarray) -> tuple[int, float, np.ndarray]:
    """Return numerical rank, tolerance and singular values."""
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0:
        return 0, 0.0, singular_values

    tolerance = (
        max(matrix.shape)
        * np.finfo(float).eps
        * float(singular_values[0])
    )
    rank = int(np.sum(singular_values > tolerance))
    return rank, float(tolerance), singular_values


def to_float_list(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=float).reshape(-1)]


def main() -> None:
    print("===== STAGE 7C DIAGNOSTIC: DOUBLE-SUPPORT CONTACT-RANK AUDIT =====")
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

    left_jacobian = pin.computeFrameJacobian(
        model,
        data,
        q0,
        left_frame_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )
    right_jacobian = pin.computeFrameJacobian(
        model,
        data,
        q0,
        right_frame_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )

    # Crocoddyl ContactModel3D constrains the top three Jacobian rows,
    # corresponding to frame translation in this convention.
    full_6d_stack = np.vstack([left_jacobian, right_jacobian])
    translation_3d_stack = np.vstack(
        [left_jacobian[:3, :], right_jacobian[:3, :]]
    )

    rank_6d, tol_6d, singular_6d = numerical_rank_and_singular_values(full_6d_stack)
    rank_3d, tol_3d, singular_3d = numerical_rank_and_singular_values(
        translation_3d_stack
    )

    nullity_6d = int(model.nv - rank_6d)
    nullity_3d = int(model.nv - rank_3d)

    left_position = data.oMf[left_frame_id].translation.copy()
    right_position = data.oMf[right_frame_id].translation.copy()

    print()
    print(f"Model velocity DoF nv: {model.nv}")
    print(
        "Left / right support positions at x0 (m): "
        f"{left_position.tolist()} / {right_position.tolist()}"
    )
    print()
    print("Two-foot 6D welded-contact Jacobian:")
    print(f"  stacked shape : {list(full_6d_stack.shape)}")
    print(f"  rank          : {rank_6d}")
    print(f"  nullity       : {nullity_6d}")
    print(f"  tolerance     : {tol_6d:.3e}")
    print(f"  singular vals : {np.array2string(singular_6d, precision=6)}")
    print()
    print("Two-foot 3D translational-contact Jacobian:")
    print(f"  stacked shape : {list(translation_3d_stack.shape)}")
    print(f"  rank          : {rank_3d}")
    print(f"  nullity       : {nullity_3d}")
    print(f"  tolerance     : {tol_3d:.3e}")
    print(f"  singular vals : {np.array2string(singular_3d, precision=6)}")
    print()

    if nullity_6d == 0:
        conclusion = (
            "The two 6D foot contacts remove all local velocity freedom at x0. "
            "A non-static CoM-shift task cannot be expected to remain contact-consistent "
            "with both feet modelled as welded 6D contacts."
        )
        recommended_next_model = (
            "Use a reduced planar-contact formulation for the non-static validation "
            "stage: translational contacts plus explicit foot-pitch/orientation "
            "regularisation, rather than two welded 6D contacts."
        )
    else:
        conclusion = (
            "The two 6D foot contacts leave local velocity freedom at x0. "
            "The previous drift must be investigated as a solver/contact-tuning issue."
        )
        recommended_next_model = (
            "Inspect contact reference placement and FDDP convergence before changing "
            "the contact dimensionality."
        )

    summary = {
        "stage": "7C_contact_rank_audit",
        "purpose": (
            "Determine whether two simultaneous 6D rigid foot contacts leave enough "
            "local kinematic freedom for a non-static CoM-shift task."
        ),
        "model": {
            "urdf": str(URDF_PATH),
            "nq": int(model.nq),
            "nv": int(model.nv),
        },
        "support_frames": {
            "left": LEFT_FOOT_FRAME,
            "right": RIGHT_FOOT_FRAME,
            "left_world_position_m": to_float_list(left_position),
            "right_world_position_m": to_float_list(right_position),
        },
        "two_foot_6d_contact": {
            "jacobian_shape": [int(v) for v in full_6d_stack.shape],
            "rank": rank_6d,
            "nullity": nullity_6d,
            "svd_tolerance": tol_6d,
            "singular_values": to_float_list(singular_6d),
        },
        "two_foot_3d_translation_contact": {
            "jacobian_shape": [int(v) for v in translation_3d_stack.shape],
            "rank": rank_3d,
            "nullity": nullity_3d,
            "svd_tolerance": tol_3d,
            "singular_values": to_float_list(singular_3d),
        },
        "conclusion": conclusion,
        "recommended_next_model": recommended_next_model,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        yaml.safe_dump(summary, sort_keys=False),
        encoding="utf-8",
    )

    print("Conclusion:")
    print(f"  {conclusion}")
    print("Recommended next model:")
    print(f"  {recommended_next_model}")
    print()
    print(f"Saved:\n  {OUTPUT_PATH}")
    print("DOUBLE-SUPPORT CONTACT-RANK AUDIT PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nCONTACT-RANK AUDIT FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
