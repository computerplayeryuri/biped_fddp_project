from __future__ import annotations

from pathlib import Path
import csv
import math
import numpy as np

PROJECT = Path.home() / "biped_fddp_project"
MODEL_PATH = PROJECT / "model" / "biped_planar.xml"
LOG_DIR = PROJECT / "results" / "logs"
FIG_DIR = PROJECT / "results" / "figures"

ACTUATED_JOINTS = [
    "left_hip_pitch_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
]

CONTACT_SENSOR_NAMES = [
    "left_toe_touch",
    "left_heel_touch",
    "right_toe_touch",
    "right_heel_touch",
]

# Simulation-only contact thresholds. Later, hardware microswitches provide 0/1 directly.
CONTACT_ON_THRESHOLD = 0.30
CONTACT_OFF_THRESHOLD = 0.15
CONTACT_CONFIRM_SAMPLES = 2


def ensure_output_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def qpos_address(model, joint_name: str) -> int:
    import mujoco

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(f"Joint not found: {joint_name}")
    return int(model.jnt_qposadr[jid])


def qvel_address(model, joint_name: str) -> int:
    import mujoco

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(f"Joint not found: {joint_name}")
    return int(model.jnt_dofadr[jid])


def get_actuated_q(model, data) -> np.ndarray:
    return np.array(
        [data.qpos[qpos_address(model, name)] for name in ACTUATED_JOINTS],
        dtype=float,
    )


def get_actuated_dq(model, data) -> np.ndarray:
    return np.array(
        [data.qvel[qvel_address(model, name)] for name in ACTUATED_JOINTS],
        dtype=float,
    )


def contact_sensor_values(model, data) -> dict[str, float]:
    import mujoco

    values: dict[str, float] = {}
    for sensor_name in CONTACT_SENSOR_NAMES:
        sensor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name
        )
        if sensor_id < 0:
            raise ValueError(f"Sensor not found: {sensor_name}")
        address = int(model.sensor_adr[sensor_id])
        values[sensor_name] = float(data.sensordata[address])
    return values


class ContactDebouncer:
    """Turn continuous MuJoCo touch values into robust 0/1 toe/heel switches."""

    def __init__(
        self,
        on_threshold: float = CONTACT_ON_THRESHOLD,
        off_threshold: float = CONTACT_OFF_THRESHOLD,
        confirm_samples: int = CONTACT_CONFIRM_SAMPLES,
    ) -> None:
        if off_threshold >= on_threshold:
            raise ValueError("off_threshold must be lower than on_threshold")

        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.confirm_samples = confirm_samples
        self.state = {name: False for name in CONTACT_SENSOR_NAMES}
        self.change_counter = {name: 0 for name in CONTACT_SENSOR_NAMES}

    def update(self, raw_values: dict[str, float]) -> dict[str, int]:
        output: dict[str, int] = {}

        for sensor_name in CONTACT_SENSOR_NAMES:
            raw_value = raw_values[sensor_name]
            current_state = self.state[sensor_name]
            desired_state = (
                raw_value > self.off_threshold
                if current_state
                else raw_value >= self.on_threshold
            )

            if desired_state != current_state:
                self.change_counter[sensor_name] += 1
                if self.change_counter[sensor_name] >= self.confirm_samples:
                    self.state[sensor_name] = desired_state
                    self.change_counter[sensor_name] = 0
            else:
                self.change_counter[sensor_name] = 0

            output[sensor_name.replace("_touch", "_switch")] = int(
                self.state[sensor_name]
            )

        left_toe = bool(output["left_toe_switch"])
        left_heel = bool(output["left_heel_switch"])
        right_toe = bool(output["right_toe_switch"])
        right_heel = bool(output["right_heel_switch"])

        output["left_any_contact"] = int(left_toe or left_heel)
        output["right_any_contact"] = int(right_toe or right_heel)
        output["left_full_support"] = int(left_toe and left_heel)
        output["right_full_support"] = int(right_toe and right_heel)

        return output


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def torso_pitch_rad(model, data) -> float:
    jid = qpos_address(model, "root_pitch_x")
    return float(data.qpos[jid])


def hip_height_m(model, data) -> float:
    import mujoco

    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "hip_plate_link"
    )
    return float(data.xpos[body_id, 2])


def is_fallen(
    model,
    data,
    pitch_limit_deg: float = 30.0,
    min_hip_height_m: float = 0.12,
) -> bool:
    pitch_exceeded = abs(torso_pitch_rad(model, data)) > math.radians(
        pitch_limit_deg
    )
    height_too_low = hip_height_m(model, data) < min_hip_height_m
    return pitch_exceeded or height_too_low
