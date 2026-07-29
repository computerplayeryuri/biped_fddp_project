from __future__ import annotations
import mujoco
from common import MODEL_PATH, ACTUATED_JOINTS, qpos_address, qvel_address

print(f"Loading: {MODEL_PATH}")
model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print("\n===== MODEL COUNTS =====")
print("nq       =", model.nq)
print("nv       =", model.nv)
print("nu       =", model.nu)
print("nbody    =", model.nbody)
print("nsensor  =", model.nsensor)
print("timestep =", model.opt.timestep)

print("\n===== ACTUATED JOINTS =====")
for name in ACTUATED_JOINTS:
    print(f"{name:28s} qpos_adr={qpos_address(model, name)} qvel_adr={qvel_address(model, name)}")

print("\n===== ACTUATORS =====")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"{i}: {name}")

print("\n===== INITIAL CONTACT SENSOR VALUES =====")
for i in range(model.nsensor):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
    print(f"{name}: dim={model.sensor_dim[i]}")

print("\nMODEL LOAD CHECK PASSED")
