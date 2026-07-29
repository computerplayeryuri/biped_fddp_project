from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import yaml

from actuator_reference_filter import ActuatorReferenceFilter
from common import (
    MODEL_PATH,
    PROJECT,
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

# This is the first non-zero motion test. It is still NOT FDDP or MPC.
# It validates the controller pipeline:
# nominal reference -> actuator delay/lag/rate limit -> MuJoCo position actuator -> logged joint response.
DURATION_S = 8.0
RAMP_IN_START_S = 1.0
RAMP_IN_END_S = 2.5
HOLD_END_S = 4.5
RAMP_OUT_END_S = 6.0

# Same command on both legs.
# Hip +4.6 deg, knee -9.2 deg, ankle +4.6 deg keeps the nominal foot orientation close to level.
SQUAT_POSE = np.array([
    +0.080, -0.160, +0.080,
    +0.080, -0.160, +0.080,
], dtype=float)


def smoothstep01(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def nominal_reference(time_s: float) -> np.ndarray:
    if time_s < RAMP_IN_START_S:
        scale = 0.0
    elif time_s < RAMP_IN_END_S:
        scale = smoothstep01(
            (time_s - RAMP_IN_START_S) / (RAMP_IN_END_S - RAMP_IN_START_S)
        )
    elif time_s < HOLD_END_S:
        scale = 1.0
    elif time_s < RAMP_OUT_END_S:
        scale = 1.0 - smoothstep01(
            (time_s - HOLD_END_S) / (RAMP_OUT_END_S - HOLD_END_S)
        )
    else:
        scale = 0.0
    return scale * SQUAT_POSE


config_path = PROJECT / "config" / "actuator_model.yaml"
with config_path.open("r") as file:
    actuator_config = yaml.safe_load(file)

CONTROL_DT = float(actuator_config["control_timestep_s"])

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

ctrl_min = model.actuator_ctrlrange[:, 0].copy()
ctrl_max = model.actuator_ctrlrange[:, 1].copy()

filter_ = ActuatorReferenceFilter(
    n_joints=6,
    control_dt_s=CONTROL_DT,
    delay_s=float(actuator_config["servo_delay_s"]),
    time_constant_s=float(actuator_config["servo_time_constant_s"]),
    safe_speed_rad_s=float(actuator_config["safe_speed_rad_s"]),
    max_change_per_cycle_rad=float(actuator_config["max_reference_change_per_cycle_rad"]),
)

ensure_output_dirs()
debouncer = ContactDebouncer()
rows: list[dict] = []
next_control_time = 0.0
next_record_time = 0.0
fall_detected = False

while data.time < DURATION_S:
    if data.time + 1e-12 >= next_control_time:
        q_nom = nominal_reference(data.time)
        q_cmd = filter_.update(q_nom)
        q_cmd = np.clip(q_cmd, ctrl_min, ctrl_max)
        data.ctrl[:] = q_cmd
        next_control_time += CONTROL_DT

    mujoco.mj_step(model, data)

    if data.time + 1e-12 >= next_record_time:
        q = get_actuated_q(model, data)
        dq = get_actuated_dq(model, data)
        raw_contacts = contact_sensor_values(model, data)
        switches = debouncer.update(raw_contacts)
        pitch = torso_pitch_rad(model, data)
        fallen_now = is_fallen(model, data)

        row = {
            "time_s": float(data.time),
            "hip_height_m": hip_height_m(model, data),
            "torso_pitch_rad": pitch,
            "torso_pitch_deg": math.degrees(pitch),
            "fall_flag": int(fallen_now),
            "joint_rmse_to_cmd_rad": float(np.sqrt(np.mean((q - q_cmd) ** 2))),
            "joint_rmse_to_cmd_deg": float(math.degrees(np.sqrt(np.mean((q - q_cmd) ** 2)))),
        }
        row.update({f"q_nom_{i}": float(q_nom[i]) for i in range(6)})
        row.update({f"q_cmd_{i}": float(q_cmd[i]) for i in range(6)})
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

path = LOG_DIR / "actuator_squat_test.csv"
append_csv(path, rows)

print(f"Saved: {path}")
print(f"Completed simulation time: {data.time:.3f} s")
print(f"Fall detected: {fall_detected}")
if rows:
    print(f"Final torso pitch: {rows[-1]['torso_pitch_deg']:.2f} deg")
    print(f"Peak joint RMSE to filtered command: {max(row['joint_rmse_to_cmd_deg'] for row in rows):.3f} deg")
    print(f"Final left full support: {rows[-1]['left_full_support']}")
    print(f"Final right full support: {rows[-1]['right_full_support']}")
print("ACTUATOR SQUAT TEST FINISHED")
