from __future__ import annotations

from pathlib import Path
import math
import sys

import numpy as np

try:
    import pinocchio as pin
except ModuleNotFoundError as exc:
    print("ERROR: Pinocchio is not available in the active Python environment.")
    print("Activate the fddp_biped environment, then install/repair Pinocchio.")
    raise exc

URDF_PATH = Path.home() / "biped_fddp_project" / "model" / "biped_v1.urdf"

EXPECTED_JOINTS = [
    "left_hip_pitch_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
]

print("\n===== STAGE 6: PINOCCHIO / FDDP MODEL AUDIT =====")
print(f"URDF: {URDF_PATH}")

if not URDF_PATH.exists():
    raise FileNotFoundError(
        f"URDF not found: {URDF_PATH}\n"
        "Expected location: ~/biped_fddp_project/model/biped_v1.urdf"
    )

try:
    import crocoddyl
    print(f"Crocoddyl import: OK")
except ModuleNotFoundError:
    print("Crocoddyl import: NOT AVAILABLE")
    print("This is not fatal for this audit, but FDDP cannot start until it is installed.")

fixed_model = pin.buildModelFromUrdf(str(URDF_PATH))
free_model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())

print("\n===== MODEL DIMENSIONS =====")
print(f"Fixed base    : nq={fixed_model.nq}, nv={fixed_model.nv}")
print(f"Free flyer   : nq={free_model.nq}, nv={free_model.nv}")

print("\n===== EXPECTED PITCH JOINTS =====")
all_ok = True
for name in EXPECTED_JOINTS:
    exists = free_model.existJointName(name)
    all_ok &= exists
    print(f"{name:30s}: {'OK' if exists else 'MISSING'}")

if not all_ok:
    raise RuntimeError("One or more required pitch joints are missing. Stop here.")

print("\n===== JOINT LIMITS =====")
for name in EXPECTED_JOINTS:
    jid = free_model.getJointId(name)
    idx_q = free_model.idx_qs[jid]
    q_min = free_model.lowerPositionLimit[idx_q]
    q_max = free_model.upperPositionLimit[idx_q]
    print(
        f"{name:30s}: "
        f"[{math.degrees(float(q_min)):+.2f}, {math.degrees(float(q_max)):+.2f}] deg"
    )

data = free_model.createData()
q0 = pin.neutral(free_model)
pin.forwardKinematics(free_model, data, q0)
pin.updateFramePlacements(free_model, data)
pin.centerOfMass(free_model, data, q0)

total_mass = pin.computeTotalMass(free_model)
com = data.com[0]

print("\n===== MASS AND NEUTRAL-POSE CoM =====")
print(f"Total URDF mass: {total_mass:.6f} kg")
print(
    "Neutral-pose CoM (root frame): "
    f"x={com[0]:+.5f} m, y={com[1]:+.5f} m, z={com[2]:+.5f} m"
)

print("\n===== FOOT-RELATED FRAMES =====")
foot_frames = []
for frame_id, frame in enumerate(free_model.frames):
    lname = frame.name.lower()
    if "foot" in lname or "ankle" in lname:
        foot_frames.append((frame_id, frame.name, frame.parentJoint))

if not foot_frames:
    print("No frame name contains 'foot' or 'ankle'.")
    print("This does not necessarily mean the URDF is wrong; inspect the frame list later.")
else:
    for frame_id, frame_name, parent_joint in foot_frames:
        pos = data.oMf[frame_id].translation
        print(
            f"{frame_id:3d} | {frame_name:35s} | parent_joint={parent_joint:2d} | "
            f"pos=[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}] m"
        )

print("\n===== INTERPRETATION =====")
print("The FDDP model will use the free-flyer version (nq=13, nv=12).")
print("The MuJoCo planar model remains the closed-loop evaluation environment.")
print("Do not expect the neutral URDF pose to be a perfect physical standing pose yet.")

print("\nPINOCCHIO / FDDP MODEL AUDIT PASSED")
