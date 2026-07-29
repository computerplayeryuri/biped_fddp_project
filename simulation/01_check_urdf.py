from pathlib import Path
import pinocchio as pin

URDF_PATH = Path.home() / "biped_fddp_project/model/biped_v1.urdf"

# 固定根模型：只检查六个实体关节
fixed_model = pin.buildModelFromUrdf(str(URDF_PATH))

# 自由浮动根模型：后续 FDDP 使用
free_model = pin.buildModelFromUrdf(
    str(URDF_PATH),
    pin.JointModelFreeFlyer()
)

print("\n===== Fixed-base URDF =====")
print("nq =", fixed_model.nq)
print("nv =", fixed_model.nv)

print("\n===== Free-flyer model for FDDP =====")
print("nq =", free_model.nq)
print("nv =", free_model.nv)

print("\n===== Joints =====")
for joint_id, name in enumerate(free_model.names):
    if joint_id == 0:
        continue
    joint = free_model.joints[joint_id]
    print(
        f"{joint_id:2d} | {name:30s} | "
        f"nq={joint.nq} nv={joint.nv}"
    )

expected = [
    "left_hip_pitch_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
]

print("\n===== Required joint check =====")
for name in expected:
    exists = free_model.existJointName(name)
    print(f"{name:30s}: {'OK' if exists else 'MISSING'}")