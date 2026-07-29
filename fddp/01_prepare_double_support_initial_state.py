"""Stage 6B: prepare a contact-consistent double-support state for Crocoddyl/FDDP.

This script does NOT solve FDDP yet. It establishes one reproducible
free-flyer initial state x0 for which both foot-contact frames lie on z=0.
That x0 and the two world-frame contact references are saved for the next
FDDP standing problem.
"""
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import yaml
import pinocchio as pin

PROJECT = Path.home() / "biped_fddp_project"
URDF_PATH = PROJECT / "model" / "biped_v1.urdf"
OUT_DIR = PROJECT / "transfer" / "initialisation"

LEFT_FRAME = "left_foot_contact_fixed"
RIGHT_FRAME = "right_foot_contact_fixed"


def frame_id_or_fail(model: pin.Model, name: str) -> int:
    fid = model.getFrameId(name)
    # Pinocchio returns model.nframes when the frame cannot be found.
    if fid >= model.nframes:
        raise RuntimeError(f"Required frame not found: {name}")
    return fid


def frame_placements(model: pin.Model, data: pin.Data, q: np.ndarray, ids: list[int]):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return [data.oMf[fid].copy() for fid in ids]


def main() -> None:
    if not URDF_PATH.exists():
        raise FileNotFoundError(f"URDF not found: {URDF_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
    data = model.createData()

    left_id = frame_id_or_fail(model, LEFT_FRAME)
    right_id = frame_id_or_fail(model, RIGHT_FRAME)

    # Pinocchio free-flyer configuration layout:
    # [base_x, base_y, base_z, quat_x, quat_y, quat_z, quat_w, 6 joint angles]
    q0 = pin.neutral(model)
    v0 = np.zeros(model.nv)

    # At neutral q, the feet are below the root. Translate the root so that
    # the average foot-contact height becomes exactly z=0.
    left0, right0 = frame_placements(model, data, q0, [left_id, right_id])
    mean_foot_z = 0.5 * (left0.translation[2] + right0.translation[2])
    q0[2] -= mean_foot_z

    left_ref, right_ref = frame_placements(model, data, q0, [left_id, right_id])

    max_height_error = max(
        abs(float(left_ref.translation[2])),
        abs(float(right_ref.translation[2])),
    )
    if max_height_error > 1e-8:
        raise RuntimeError(
            "Foot contacts were not placed on ground as expected. "
            f"max error={max_height_error:.3e} m"
        )

    # World-frame CoM in the contact-consistent standing state.
    pin.centerOfMass(model, data, q0, v0)
    com_world = np.array(data.com[0], dtype=float)

    x0 = np.concatenate([q0, v0])
    np.savez(
        OUT_DIR / "fddp_double_support_x0.npz",
        q0=q0,
        v0=v0,
        x0=x0,
        left_contact_translation=left_ref.translation,
        left_contact_rotation=left_ref.rotation,
        right_contact_translation=right_ref.translation,
        right_contact_rotation=right_ref.rotation,
    )

    contact_data = {
        "model": {
            "urdf": str(URDF_PATH),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nv - 6),
        },
        "standing_root_pose": {
            "translation_m": [float(v) for v in q0[:3]],
            "quaternion_xyzw": [float(v) for v in q0[3:7]],
        },
        "support_contacts": {
            "left": {
                "frame": LEFT_FRAME,
                "translation_m": [float(v) for v in left_ref.translation],
                "rotation_row_major": [float(v) for v in left_ref.rotation.reshape(-1)],
            },
            "right": {
                "frame": RIGHT_FRAME,
                "translation_m": [float(v) for v in right_ref.translation],
                "rotation_row_major": [float(v) for v in right_ref.rotation.reshape(-1)],
            },
        },
        "neutral_joint_reference_rad": [float(v) for v in q0[7:]],
        "com_world_m": [float(v) for v in com_world],
        "notes": [
            "Both contact frames are placed on z=0 for the FDDP standing initialisation.",
            "This is an ideal rigid double-support contact assumption for the first FDDP task.",
            "This file is not yet a walking contact schedule.",
        ],
    }
    with (OUT_DIR / "fddp_double_support_contacts.yaml").open("w") as f:
        yaml.safe_dump(contact_data, f, sort_keys=False)

    print("===== STAGE 6B: DOUBLE-SUPPORT FDDP INITIALISATION =====")
    print(f"URDF: {URDF_PATH}")
    print(f"Free-flyer model: nq={model.nq}, nv={model.nv}, nu={model.nv - 6}")
    print("\nRoot pose selected for standing:")
    print(
        f"  translation = [{q0[0]:+.5f}, {q0[1]:+.5f}, {q0[2]:+.5f}] m"
    )
    print(f"  quaternion  = {q0[3:7]}")
    print("\nSupport contact references in the world frame:")
    print(
        "  left : "
        f"[{left_ref.translation[0]:+.5f}, {left_ref.translation[1]:+.5f}, "
        f"{left_ref.translation[2]:+.8f}] m"
    )
    print(
        "  right: "
        f"[{right_ref.translation[0]:+.5f}, {right_ref.translation[1]:+.5f}, "
        f"{right_ref.translation[2]:+.8f}] m"
    )
    print(
        f"\nWorld CoM at this standing state: "
        f"[{com_world[0]:+.5f}, {com_world[1]:+.5f}, {com_world[2]:+.5f}] m"
    )
    print(f"Foot ground-height error: {max_height_error:.3e} m")
    print("\nSaved:")
    print(f"  {OUT_DIR / 'fddp_double_support_x0.npz'}")
    print(f"  {OUT_DIR / 'fddp_double_support_contacts.yaml'}")
    print("\nDOUBLE-SUPPORT INITIALISATION PASSED")


if __name__ == "__main__":
    main()
