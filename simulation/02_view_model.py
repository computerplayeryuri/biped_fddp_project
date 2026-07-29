from __future__ import annotations
import time
import mujoco
import mujoco.viewer
from common import MODEL_PATH

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

print("A MuJoCo viewer window should open. Close the viewer window to exit.")
print("Expected visual check: both feet touch the ground, legs are vertical, hip plate is above them.")
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(0.02)
