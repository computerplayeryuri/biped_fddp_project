from __future__ import annotations
import time
import mujoco
import numpy as np
from common import MODEL_PATH, ensure_output_dirs, LOG_DIR, get_actuated_q, get_actuated_dq, torso_pitch_rad, hip_height_m, contact_sensor_values, append_csv, is_fallen

DURATION_S = 3.0
RECORD_EVERY = 10

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)
# Disable position-servo gains for this contact/gravity sanity check.
model.actuator_gainprm[:, :] = 0.0
model.actuator_biasprm[:, :] = 0.0
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

ensure_output_dirs()
rows = []
start_wall = time.perf_counter()
step = 0
while data.time < DURATION_S:
    mujoco.mj_step(model, data)
    if step % RECORD_EVERY == 0:
        contacts = contact_sensor_values(model, data)
        row = {
            "time_s": float(data.time),
            "hip_height_m": hip_height_m(model, data),
            "torso_pitch_rad": torso_pitch_rad(model, data),
            "fall_flag": int(is_fallen(model, data)),
        }
        row.update({f"q_{i}": float(v) for i, v in enumerate(get_actuated_q(model, data))})
        row.update({f"dq_{i}": float(v) for i, v in enumerate(get_actuated_dq(model, data))})
        row.update(contacts)
        rows.append(row)
    step += 1

path = LOG_DIR / "passive_drop.csv"
append_csv(path, rows)
print(f"Saved: {path}")
print(f"Final time: {data.time:.3f} s")
print(f"Final hip height: {hip_height_m(model, data):.4f} m")
print(f"Final torso pitch: {np.degrees(torso_pitch_rad(model, data)):.2f} deg")
print("PASSIVE DROP CHECK FINISHED")
