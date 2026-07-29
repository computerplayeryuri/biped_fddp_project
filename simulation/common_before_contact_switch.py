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
    return np.array([data.qpos[qpos_address(model, name)] for name in ACTUATED_JOINTS], dtype=float)


def get_actuated_dq(model, data) -> np.ndarray:
    return np.array([data.qvel[qvel_address(model, name)] for name in ACTUATED_JOINTS], dtype=float)


def contact_sensor_values(model, data) -> dict[str, float]:
    import mujoco
    values: dict[str, float] = {}
    for sensor_name in ["left_toe_touch", "left_heel_touch", "right_toe_touch", "right_heel_touch"]:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        adr = int(model.sensor_adr[sid])
        values[sensor_name] = float(data.sensordata[adr])
    return values


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def torso_pitch_rad(model, data) -> float:
    # Root pitch joint is explicitly the third qpos variable in this model.
    jid = qpos_address(model, "root_pitch_x")
    return float(data.qpos[jid])


def hip_height_m(model, data) -> float:
    import mujoco
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hip_plate_link")
    return float(data.xpos[bid, 2])


def is_fallen(model, data, pitch_limit_deg: float = 30.0, min_hip_height_m: float = 0.12) -> bool:
    return abs(torso_pitch_rad(model, data)) > math.radians(pitch_limit_deg) or hip_height_m(model, data) < min_hip_height_m
