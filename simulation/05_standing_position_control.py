from __future__ import annotations

import math
import mujoco
import numpy as np

from common import (
    MODEL_PATH,
    LOG_DIR,
    ContactDebouncer,
    append_csv,
    contact_sensor_values,
    ensure_output_dirs,
    get_actuated_dq,
    get_actuated_q,
    hip_height_m,
    is_fallen,
    torso_pitch_rad,
)

# Stage 3.5 experiment: fixed neutral-pose position control.
# This is NOT FDDP or MPC yet. It verifies that the six position actuators,
# contact-switch logic, logging, and fall detector work together.
DURATION_S = 10.0
CONTROL_DT = 0.020  # 50 Hz control update
Q_REF = np.zeros(6, dtype=float)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

ensure_output_dirs()
debouncer = ContactDebouncer()
rows: list[dict] = []
next_control_time = 0.0
next_record_time = 0.0
fall_detected = False

while data.time < DURATION_S:
    # Apply the neutral position reference at 50 Hz.
    if data.time + 1e-12 >= next_control_time:
        data.ctrl[:] = Q_REF
        next_control_time += CONTROL_DT

    mujoco.mj_step(model, data)

    # Record once per control period, after contact dynamics have been updated.
    if data.time + 1e-12 >= next_record_time:
        q = get_actuated_q(model, data)
        dq = get_actuated_dq(model, data)
        raw_contacts = contact_sensor_values(model, data)
        switches = debouncer.update(raw_contacts)
        pitch_rad = torso_pitch_rad(model, data)
        fallen_now = is_fallen(model, data)

        row = {
            "time_s": float(data.time),
            "hip_height_m": hip_height_m(model, data),
            "torso_pitch_rad": pitch_rad,
            "torso_pitch_deg": math.degrees(pitch_rad),
            "fall_flag": int(fallen_now),
            "joint_rmse_rad": float(np.sqrt(np.mean((q - Q_REF) ** 2))),
            "joint_rmse_deg": float(math.degrees(np.sqrt(np.mean((q - Q_REF) ** 2)))),
        }
        row.update({f"q_ref_{i}": float(Q_REF[i]) for i in range(6)})
        row.update({f"q_cmd_{i}": float(data.ctrl[i]) for i in range(6)})
        row.update({f"q_{i}": float(q[i]) for i in range(6)})
        row.update({f"dq_{i}": float(dq[i]) for i in range(6)})
        row.update(raw_contacts)
        row.update(switches)
        rows.append(row)

        next_record_time += CONTROL_DT

        if fallen_now:
            fall_detected = True
            print(f"Fall criterion reached at t={data.time:.3f} s.")
            break

path = LOG_DIR / "standing_position_control.csv"
append_csv(path, rows)

print(f"Saved: {path}")
print(f"Completed simulation time: {data.time:.3f} s")
print(f"Final hip height: {hip_height_m(model, data):.4f} m")
print(f"Final torso pitch: {math.degrees(torso_pitch_rad(model, data)):.2f} deg")
print(f"Fall detected: {fall_detected}")
if rows:
    print(f"Final left full support: {rows[-1]['left_full_support']}")
    print(f"Final right full support: {rows[-1]['right_full_support']}")
print("STANDING POSITION-CONTROL CHECK FINISHED")
