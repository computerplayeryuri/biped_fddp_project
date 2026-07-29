from pathlib import Path
import mujoco

PROJECT = Path.home() / "biped_fddp_project"
MODEL_PATH = PROJECT / "model" / "biped_planar.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

# 载入 XML 中名为 standing 的初始姿态。
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# 运行 0.5 秒，让脚底与地面接触稳定。
for _ in range(250):
    mujoco.mj_step(model, data)

print("\n===== CONTACT DIAGNOSTIC =====")
print(f"Simulation time: {data.time:.3f} s")
print(f"Number of active contacts: {data.ncon}")

print("\n===== ACTIVE GEOM CONTACT PAIRS =====")
if data.ncon == 0:
    print("No contact pair was detected.")
else:
    for i in range(data.ncon):
        contact = data.contact[i]

        geom1_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)
        )
        geom2_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)
        )

        print(
            f"Contact {i}: "
            f"{geom1_name} <-> {geom2_name}, "
            f"world position = {contact.pos}"
        )

print("\n===== CURRENT TOUCH-SENSOR VALUES =====")
sensor_names = [
    "left_toe_touch",
    "left_heel_touch",
    "right_toe_touch",
    "right_heel_touch",
]

for sensor_name in sensor_names:
    sensor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name
    )
    address = int(model.sensor_adr[sensor_id])
    value = float(data.sensordata[address])
    print(f"{sensor_name:20s}: {value:.6f}")

print("\nCONTACT DIAGNOSTIC FINISHED")
