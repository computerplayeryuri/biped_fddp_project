#!/usr/bin/env python3
"""
STAGE 7C — MuJoCo replay of validated FDDP CoM-shift references.

This script replays the exported FDDP joint-position references in the
existing MuJoCo planar model through a simple actuator-aware position-command
proxy. It does NOT run FDDP and does NOT change any FDDP trajectory file.

Inputs:
  transfer/trajectories/com_shift_forward_ref.npz
  transfer/trajectories/com_shift_backward_ref.npz

Outputs:
  results/logs/fddp_com_shift_mujoco_replay_{forward,backward}.csv
  results/figures/fddp_com_shift_mujoco_replay_{forward,backward}_joints.png
  results/figures/fddp_com_shift_mujoco_replay_{forward,backward}_body.png
  results/figures/fddp_com_shift_mujoco_replay_{forward,backward}_contacts.png
  results/logs/fddp_com_shift_mujoco_replay_summary.yaml

Important scope:
  - This is a MuJoCo execution-validation stage, not hardware validation.
  - tau_ref is never used as a servo command.
  - q_ref is used as the position command reference.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import argparse
import math
import sys

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
import yaml


# ------------------------------- Configuration -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = PROJECT_ROOT / "model" / "biped_planar.xml"
TRAJECTORY_DIR = PROJECT_ROOT / "transfer" / "trajectories"
LOG_DIR = PROJECT_ROOT / "results" / "logs"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

# FDDP reference sampling is 100 Hz, MuJoCo should be 500 Hz according to the
# existing model's validated dt=0.002 s. The script checks actual XML timestep.
FDDP_REFERENCE_DT_S = 0.010

# Simplified actuator-aware position-command proxy. These are simulation
# assumptions, not a hardware identification of ST3215 dynamics.
COMMAND_DELAY_S = 0.020
FIRST_ORDER_TIME_CONSTANT_S = 0.030
MAX_COMMAND_SPEED_RAD_PER_S = 8.0

# Conservative execution integrity gates.
MAX_JOINT_RMSE_DEG = 5.0
MAX_ABS_TORSO_PITCH_DEG = 15.0
MIN_TORSO_HEIGHT_RATIO = 0.65
MIN_TOTAL_CONTACTS = 1

# FDDP/URDF joint order used by the exported q_ref:
JOINT_ORDER = [
    "left_hip_pitch_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
]

# Names are searched in this order. The script prints a diagnostic if the XML
# uses a different body naming scheme.
TORSO_BODY_CANDIDATES = [
    "hip_plate_link",
    "electronics_lump_link",
    "torso",
    "pelvis",
    "base",
    "body",
    "top_electronics",
    "hip_bridge",
]

# Optional contact-sensor pattern candidates. The script still runs without
# them and records global MuJoCo contact count ncon.
CONTACT_SENSOR_PATTERNS = {
    "left_toe": ("left", "toe"),
    "left_heel": ("left", "heel"),
    "right_toe": ("right", "toe"),
    "right_heel": ("right", "heel"),
}


class PositionActuatorProxy:
    """Delay + first-order lag + command speed limit for position commands."""

    def __init__(self, n: int, dt: float):
        if dt <= 0:
            raise ValueError("MuJoCo timestep must be positive.")
        if FIRST_ORDER_TIME_CONSTANT_S <= 0:
            raise ValueError("FIRST_ORDER_TIME_CONSTANT_S must be positive.")
        self.n = n
        self.dt = float(dt)
        self.delay_steps = int(round(COMMAND_DELAY_S / self.dt))
        self.history = deque(maxlen=self.delay_steps + 1)
        self.output = np.zeros(n, dtype=float)

    def reset(self, q_initial: np.ndarray) -> None:
        q_initial = np.asarray(q_initial, dtype=float).reshape(self.n)
        self.history.clear()
        for _ in range(self.delay_steps + 1):
            self.history.append(q_initial.copy())
        self.output = q_initial.copy()

    def update(self, q_command: np.ndarray) -> np.ndarray:
        q_command = np.asarray(q_command, dtype=float).reshape(self.n)
        self.history.append(q_command.copy())
        delayed_command = self.history[0]

        alpha = min(1.0, self.dt / FIRST_ORDER_TIME_CONSTANT_S)
        desired_increment = alpha * (delayed_command - self.output)
        max_increment = MAX_COMMAND_SPEED_RAD_PER_S * self.dt
        desired_increment = np.clip(
            desired_increment,
            -max_increment,
            max_increment,
        )
        self.output = self.output + desired_increment
        return self.output.copy()


def model_name(model, object_type, object_id: int) -> str | None:
    value = mujoco.mj_id2name(model, object_type, object_id)
    return value if value else None


def joint_id_or_error(model, joint_name: str) -> int:
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )
    if joint_id < 0:
        all_names = [
            model_name(model, mujoco.mjtObj.mjOBJ_JOINT, idx)
            for idx in range(model.njnt)
        ]
        raise RuntimeError(
            f"Required MuJoCo joint '{joint_name}' was not found.\n"
            f"Available joints: {all_names}"
        )
    return int(joint_id)


def resolve_joint_and_actuator_mapping(model):
    """
    Resolve each FDDP joint name to:
      - MuJoCo qpos address,
      - MuJoCo qvel address,
      - an actuator ctrl index.

    The preferred path derives actuator-to-joint mapping from actuator_trnid.
    A strict fallback uses matching actuator names only if required.
    """
    joint_ids = [joint_id_or_error(model, name) for name in JOINT_ORDER]
    qpos_addresses = [int(model.jnt_qposadr[jid]) for jid in joint_ids]
    dof_addresses = [int(model.jnt_dofadr[jid]) for jid in joint_ids]

    actuator_by_joint = {}
    for actuator_id in range(model.nu):
        transmitted_joint_id = int(model.actuator_trnid[actuator_id, 0])
        if transmitted_joint_id in joint_ids:
            actuator_by_joint[transmitted_joint_id] = actuator_id

    actuator_ids = [actuator_by_joint.get(joint_id, -1) for joint_id in joint_ids]

    if any(actuator_id < 0 for actuator_id in actuator_ids):
        # Fallback: match actuator names that contain the joint name.
        actuator_names = {
            actuator_id: (
                model_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or ""
            ).lower()
            for actuator_id in range(model.nu)
        }
        for index, joint_name in enumerate(JOINT_ORDER):
            if actuator_ids[index] >= 0:
                continue
            target = joint_name.lower()
            matches = [
                actuator_id
                for actuator_id, actuator_name in actuator_names.items()
                if target in actuator_name
            ]
            if len(matches) == 1:
                actuator_ids[index] = matches[0]

    if any(actuator_id < 0 for actuator_id in actuator_ids):
        actuator_labels = [
            model_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx)
            for idx in range(model.nu)
        ]
        raise RuntimeError(
            "Could not map every required joint to a MuJoCo actuator.\n"
            f"Required joints: {JOINT_ORDER}\n"
            f"Available actuators: {actuator_labels}\n"
            "Expected a joint-transmission actuator or an actuator name "
            "containing the corresponding joint name."
        )

    if len(set(actuator_ids)) != len(actuator_ids):
        raise RuntimeError(
            f"Actuator mapping is not one-to-one: {actuator_ids}"
        )

    return {
        "joint_ids": joint_ids,
        "qpos_addresses": qpos_addresses,
        "dof_addresses": dof_addresses,
        "actuator_ids": actuator_ids,
    }


def resolve_torso_body(model) -> int:
    for candidate in TORSO_BODY_CANDIDATES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate)
        if body_id >= 0:
            return int(body_id)

    available = [
        model_name(model, mujoco.mjtObj.mjOBJ_BODY, idx)
        for idx in range(model.nbody)
    ]
    raise RuntimeError(
        "Could not identify a torso body for height/pitch metrics.\n"
        f"Tried: {TORSO_BODY_CANDIDATES}\n"
        f"Available bodies: {available}"
    )


def find_optional_contact_sensors(model):
    """Return sensor IDs matching the conventional toe/heel name patterns."""
    sensor_names = {
        sensor_id: (
            model_name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id) or ""
        ).lower()
        for sensor_id in range(model.nsensor)
    }
    found = {}
    for logical_name, tokens in CONTACT_SENSOR_PATTERNS.items():
        matches = [
            sensor_id
            for sensor_id, sensor_name in sensor_names.items()
            if all(token in sensor_name for token in tokens)
        ]
        if len(matches) == 1:
            found[logical_name] = matches[0]
    return found


def sensor_scalar(model, data, sensor_id: int) -> float:
    address = int(model.sensor_adr[sensor_id])
    dimension = int(model.sensor_dim[sensor_id])
    values = np.asarray(data.sensordata[address : address + dimension], dtype=float)
    return float(values[0]) if values.size else float("nan")


def torso_pitch_about_x(data, torso_body_id: int) -> float:
    """
    Coordinate convention: pitch axis +X. Convert the torso world rotation
    matrix into an x-axis roll/pitch-angle-like scalar.
    """
    rotation = np.asarray(data.xmat[torso_body_id], dtype=float).reshape(3, 3)
    return float(math.atan2(rotation[2, 1], rotation[2, 2]))


def total_mass_com(model, data) -> np.ndarray:
    """Mass-weighted world CoM using body inertial positions."""
    masses = np.asarray(model.body_mass, dtype=float)
    total_mass = float(np.sum(masses))
    if total_mass <= 0:
        raise RuntimeError("Model total body mass is non-positive.")
    xipos = np.asarray(data.xipos, dtype=float).reshape(model.nbody, 3)
    return (masses[:, None] * xipos).sum(axis=0) / total_mass


def load_reference(direction: str):
    path = TRAJECTORY_DIR / f"com_shift_{direction}_ref.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing FDDP reference: {path}\n"
            "Run the successful Stage 7C continuation first."
        )

    with np.load(path) as loaded:
        required = ("t_ref", "q_ref", "dq_ref", "com_y_target", "phase")
        missing = [key for key in required if key not in loaded.files]
        if missing:
            raise RuntimeError(f"{path} is missing keys {missing}; has {loaded.files}")

        t_ref = np.asarray(loaded["t_ref"], dtype=float).reshape(-1)
        q_ref = np.asarray(loaded["q_ref"], dtype=float)
        dq_ref = np.asarray(loaded["dq_ref"], dtype=float)
        com_y_target = np.asarray(loaded["com_y_target"], dtype=float).reshape(-1)
        phase = np.asarray(loaded["phase"]).astype(str)

    if q_ref.shape != (t_ref.size, 6):
        raise RuntimeError(f"{path}: q_ref shape {q_ref.shape}; expected ({t_ref.size}, 6)")
    if dq_ref.shape != (t_ref.size, 6):
        raise RuntimeError(f"{path}: dq_ref shape {dq_ref.shape}; expected ({t_ref.size}, 6)")
    if com_y_target.shape != t_ref.shape:
        raise RuntimeError(f"{path}: com_y_target shape mismatch.")
    if phase.shape != t_ref.shape:
        raise RuntimeError(f"{path}: phase shape mismatch.")
    if not np.all(np.isfinite(q_ref)) or not np.all(np.isfinite(dq_ref)):
        raise RuntimeError(f"{path}: non-finite FDDP reference values.")

    return path, t_ref, q_ref, dq_ref, com_y_target, phase


def interpolate_reference(t: float, t_ref: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation for values shaped [N, d]."""
    return np.asarray(
        [np.interp(t, t_ref, values[:, column]) for column in range(values.shape[1])],
        dtype=float,
    )


def replay_direction(direction: str):
    reference_path, t_ref, q_ref, dq_ref, com_y_target, phase = load_reference(direction)

    if not XML_PATH.exists():
        raise FileNotFoundError(f"Missing MuJoCo XML: {XML_PATH}")

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    dt = float(model.opt.timestep)
    if dt <= 0:
        raise RuntimeError("MuJoCo XML timestep must be positive.")

    mapping = resolve_joint_and_actuator_mapping(model)
    torso_body_id = resolve_torso_body(model)
    contact_sensor_ids = find_optional_contact_sensors(model)

    # Reset to the XML's declared nominal state. The FDDP trajectory itself
    # begins with a 0.20 s neutral period, so no separate settle period is
    # added here.
    mujoco.mj_resetData(model, data)
    for qpos_address, q_initial in zip(mapping["qpos_addresses"], q_ref[0]):
        data.qpos[qpos_address] = q_initial
    for dof_address in mapping["dof_addresses"]:
        data.qvel[dof_address] = 0.0
    mujoco.mj_forward(model, data)

    initial_torso_height = float(data.xpos[torso_body_id][2])
    initial_com_y = float(total_mass_com(model, data)[1])

    if initial_torso_height <= 0:
        raise RuntimeError(
            f"Unexpected initial torso height {initial_torso_height:.6f} m."
        )

    actuator_proxy = PositionActuatorProxy(6, dt)
    actuator_proxy.reset(q_ref[0])

    duration = float(t_ref[-1])
    total_steps = int(round(duration / dt)) + 1

    records = []
    finite = True

    for _ in range(total_steps):
        simulation_time = min(float(data.time), duration)
        q_reference = interpolate_reference(simulation_time, t_ref, q_ref)
        dq_reference = interpolate_reference(simulation_time, t_ref, dq_ref)
        q_proxy = actuator_proxy.update(q_reference)

        # Position-actuator controls: command only the mapped six actuator IDs.
        data.ctrl[:] = 0.0
        for actuator_id, command in zip(mapping["actuator_ids"], q_proxy):
            data.ctrl[actuator_id] = command

        mujoco.mj_step(model, data)

        q_measured = np.asarray(
            [data.qpos[address] for address in mapping["qpos_addresses"]],
            dtype=float,
        )
        dq_measured = np.asarray(
            [data.qvel[address] for address in mapping["dof_addresses"]],
            dtype=float,
        )
        torso_height = float(data.xpos[torso_body_id][2])
        torso_pitch = torso_pitch_about_x(data, torso_body_id)
        com_y = float(total_mass_com(model, data)[1])

        if not (
            np.all(np.isfinite(q_measured))
            and np.all(np.isfinite(dq_measured))
            and math.isfinite(torso_height)
            and math.isfinite(torso_pitch)
            and math.isfinite(com_y)
        ):
            finite = False
            break

        record = {
            "time_s": float(data.time),
            "reference_time_s": simulation_time,
            "torso_height_m": torso_height,
            "torso_pitch_rad": torso_pitch,
            "torso_pitch_deg": float(np.rad2deg(torso_pitch)),
            "com_y_m": com_y,
            "com_y_change_m": com_y - initial_com_y,
            "com_y_target_m": float(np.interp(simulation_time, t_ref, com_y_target)),
            "ncon": int(data.ncon),
        }

        for index, name in enumerate(JOINT_ORDER):
            record[f"{name}_ref_rad"] = float(q_reference[index])
            record[f"{name}_proxy_cmd_rad"] = float(q_proxy[index])
            record[f"{name}_meas_rad"] = float(q_measured[index])
            record[f"{name}_ref_vel_rad_s"] = float(dq_reference[index])
            record[f"{name}_meas_vel_rad_s"] = float(dq_measured[index])

        for logical_name, sensor_id in contact_sensor_ids.items():
            record[f"{logical_name}_sensor"] = sensor_scalar(model, data, sensor_id)

        records.append(record)

    if not records:
        raise RuntimeError("Replay produced no simulation records.")

    dataframe = pd.DataFrame(records)
    q_error_columns = [
        f"{joint_name}_meas_rad" for joint_name in JOINT_ORDER
    ]
    q_reference_columns = [
        f"{joint_name}_ref_rad" for joint_name in JOINT_ORDER
    ]
    q_errors = np.column_stack(
        [
            dataframe[measured].to_numpy() - dataframe[reference].to_numpy()
            for measured, reference in zip(q_error_columns, q_reference_columns)
        ]
    )
    joint_rmse_rad = np.sqrt(np.mean(q_errors**2, axis=0))
    joint_rmse_deg = np.rad2deg(joint_rmse_rad)

    max_abs_torso_pitch_deg = float(np.max(np.abs(dataframe["torso_pitch_deg"])))
    min_torso_height = float(np.min(dataframe["torso_height_m"]))
    min_ncon = int(np.min(dataframe["ncon"]))
    achieved_com_shift = (
        float(np.max(dataframe["com_y_change_m"]))
        if direction == "forward"
        else float(np.min(dataframe["com_y_change_m"]))
    )
    target_com_shift = (
        float(np.max(com_y_target - com_y_target[0]))
        if direction == "forward"
        else float(np.min(com_y_target - com_y_target[0]))
    )

    # The desired joint motion is small. The replay gate is deliberately about
    # stable position-command execution, not exact dynamic equivalence between
    # Pinocchio/Crocoddyl and MuJoCo.
    failures = []
    if not finite:
        failures.append("non-finite MuJoCo state encountered")
    if min_torso_height < MIN_TORSO_HEIGHT_RATIO * initial_torso_height:
        failures.append(
            "torso height dropped below "
            f"{MIN_TORSO_HEIGHT_RATIO:.2f} of initial height"
        )
    if max_abs_torso_pitch_deg > MAX_ABS_TORSO_PITCH_DEG:
        failures.append(
            f"max torso pitch {max_abs_torso_pitch_deg:.3f} deg exceeds "
            f"{MAX_ABS_TORSO_PITCH_DEG:.1f} deg"
        )
    if float(np.max(joint_rmse_deg)) > MAX_JOINT_RMSE_DEG:
        failures.append(
            f"max joint RMSE {float(np.max(joint_rmse_deg)):.3f} deg exceeds "
            f"{MAX_JOINT_RMSE_DEG:.1f} deg"
        )
    if min_ncon < MIN_TOTAL_CONTACTS:
        failures.append(
            f"minimum global contact count {min_ncon} is below {MIN_TOTAL_CONTACTS}"
        )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = LOG_DIR / f"fddp_com_shift_mujoco_replay_{direction}.csv"
    dataframe.to_csv(csv_path, index=False)

    make_joint_plot(dataframe, direction)
    make_body_plot(dataframe, direction)
    make_contact_plot(dataframe, direction, contact_sensor_ids)

    summary = {
        "stage": "7C_mujoco_replay_of_damped_6d_fddp_reference",
        "direction": direction,
        "purpose": (
            "Replay FDDP q_ref in the MuJoCo planar model through a simplified "
            "position-command actuator proxy and assess stable execution."
        ),
        "inputs": {
            "mujoco_xml": str(XML_PATH),
            "fddp_reference": str(reference_path),
            "q_ref_shape": [int(value) for value in q_ref.shape],
            "dq_ref_shape": [int(value) for value in dq_ref.shape],
            "reference_duration_s": float(duration),
            "fddp_reference_dt_s": float(FDDP_REFERENCE_DT_S),
            "mujoco_dt_s": float(dt),
        },
        "joint_mapping": {
            "joint_order": JOINT_ORDER,
            "mujoco_actuator_ids": [int(value) for value in mapping["actuator_ids"]],
        },
        "actuator_proxy": {
            "command_delay_s": float(COMMAND_DELAY_S),
            "first_order_time_constant_s": float(FIRST_ORDER_TIME_CONSTANT_S),
            "max_command_speed_rad_per_s": float(MAX_COMMAND_SPEED_RAD_PER_S),
        },
        "optional_contact_sensors_found": {
            key: int(value) for key, value in contact_sensor_ids.items()
        },
        "execution_metrics": {
            "finite": bool(finite),
            "initial_torso_height_m": float(initial_torso_height),
            "min_torso_height_m": float(min_torso_height),
            "max_abs_torso_pitch_deg": float(max_abs_torso_pitch_deg),
            "min_global_contact_count": int(min_ncon),
            "joint_rmse_deg": {
                joint_name: float(rmse)
                for joint_name, rmse in zip(JOINT_ORDER, joint_rmse_deg)
            },
            "max_joint_rmse_deg": float(np.max(joint_rmse_deg)),
            "mujoco_achieved_com_y_shift_m": float(achieved_com_shift),
            "fddp_target_com_y_shift_m": float(target_com_shift),
        },
        "integrity_passed": bool(not failures),
        "failures": failures,
        "outputs": {
            "csv": str(csv_path),
            "joint_plot": str(
                FIGURE_DIR
                / f"fddp_com_shift_mujoco_replay_{direction}_joints.png"
            ),
            "body_plot": str(
                FIGURE_DIR
                / f"fddp_com_shift_mujoco_replay_{direction}_body.png"
            ),
            "contact_plot": str(
                FIGURE_DIR
                / f"fddp_com_shift_mujoco_replay_{direction}_contacts.png"
            ),
        },
    }

    print()
    print(f"----- {direction.upper()} MUJOCO REPLAY -----")
    print(f"FDDP reference: {reference_path}")
    print(
        "Joint RMSE per joint (deg): "
        + ", ".join(f"{value:.4f}" for value in joint_rmse_deg)
    )
    print(f"Max joint RMSE (deg): {float(np.max(joint_rmse_deg)):.4f}")
    print(f"Max |torso pitch| (deg): {max_abs_torso_pitch_deg:.4f}")
    print(
        "Min torso height / initial height (m): "
        f"{min_torso_height:.4f} / {initial_torso_height:.4f}"
    )
    print(f"Min global contact count: {min_ncon}")
    print(
        "FDDP target / MuJoCo achieved CoM_y shift (mm): "
        f"{target_com_shift * 1000.0:+.4f} / {achieved_com_shift * 1000.0:+.4f}"
    )
    print(f"Replay integrity: {'PASSED' if not failures else 'FAILED'}")
    for failure in failures:
        print(f"  - {failure}")
    print(f"Saved CSV: {csv_path}")

    return summary


def make_joint_plot(dataframe: pd.DataFrame, direction: str) -> None:
    figure, axis = plt.subplots(figsize=(12, 7))
    for joint_name in JOINT_ORDER:
        axis.plot(
            dataframe["time_s"],
            np.rad2deg(dataframe[f"{joint_name}_ref_rad"]),
            label=f"{joint_name} ref",
            linewidth=1.7,
        )
        axis.plot(
            dataframe["time_s"],
            np.rad2deg(dataframe[f"{joint_name}_meas_rad"]),
            label=f"{joint_name} meas",
            linewidth=1.0,
            linestyle="--",
        )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Joint angle (deg)")
    axis.set_title(f"FDDP {direction} CoM-shift: joint-reference replay")
    axis.grid(True)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / f"fddp_com_shift_mujoco_replay_{direction}_joints.png",
        dpi=180,
    )
    plt.close(figure)


def make_body_plot(dataframe: pd.DataFrame, direction: str) -> None:
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(
        dataframe["time_s"],
        dataframe["torso_pitch_deg"],
        label="torso pitch",
        linewidth=1.7,
    )
    axis.plot(
        dataframe["time_s"],
        dataframe["com_y_change_m"] * 1000.0,
        label="MuJoCo CoM_y change",
        linewidth=1.7,
    )
    axis.plot(
        dataframe["time_s"],
        (dataframe["com_y_target_m"] - dataframe["com_y_target_m"].iloc[0]) * 1000.0,
        label="FDDP CoM_y target",
        linewidth=1.3,
        linestyle="--",
    )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Pitch (deg) / CoM change (mm)")
    axis.set_title(f"FDDP {direction} CoM-shift: body response")
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / f"fddp_com_shift_mujoco_replay_{direction}_body.png",
        dpi=180,
    )
    plt.close(figure)


def make_contact_plot(
    dataframe: pd.DataFrame,
    direction: str,
    contact_sensor_ids: dict[str, int],
) -> None:
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(
        dataframe["time_s"],
        dataframe["ncon"],
        label="global contact count",
        linewidth=1.7,
    )
    for logical_name in contact_sensor_ids:
        column = f"{logical_name}_sensor"
        if column in dataframe.columns:
            axis.plot(
                dataframe["time_s"],
                dataframe[column],
                label=logical_name,
                linewidth=1.2,
            )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Contact count / sensor value")
    axis.set_title(f"FDDP {direction} CoM-shift: contact response")
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / f"fddp_com_shift_mujoco_replay_{direction}_contacts.png",
        dpi=180,
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay Stage 7C FDDP references in MuJoCo."
    )
    parser.add_argument(
        "--direction",
        choices=("forward", "backward", "both"),
        default="both",
        help="Reference direction to replay (default: both).",
    )
    args = parser.parse_args()

    print("===== STAGE 7C: MUJOCO REPLAY OF VALIDATED FDDP REFERENCES =====")
    print("This stage uses q_ref only. tau_ref is never sent to MuJoCo controls.")
    print(f"MuJoCo XML: {XML_PATH}")
    print(
        "Actuator proxy: "
        f"delay={COMMAND_DELAY_S:.3f}s, "
        f"tau={FIRST_ORDER_TIME_CONSTANT_S:.3f}s, "
        f"speed limit={MAX_COMMAND_SPEED_RAD_PER_S:.2f} rad/s"
    )

    directions = ("forward", "backward") if args.direction == "both" else (args.direction,)
    summaries = []
    for direction in directions:
        summaries.append(replay_direction(direction))

    summary_path = LOG_DIR / "fddp_com_shift_mujoco_replay_summary.yaml"
    combined = {
        "stage": "7C_mujoco_replay_of_damped_6d_fddp_references",
        "purpose": (
            "Validate stable actuator-aware MuJoCo execution of exported "
            "forward/backward FDDP nominal CoM-shift references."
        ),
        "runs": summaries,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(yaml.safe_dump(combined, sort_keys=False), encoding="utf-8")

    overall_pass = all(item["integrity_passed"] for item in summaries)
    print()
    print(
        "STAGE 7C MUJOCO REPLAY "
        + ("PASSED" if overall_pass else "COMPLETED WITH FAILURES")
    )
    print(f"Saved summary:\n  {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nMUJOCO FDDP REPLAY FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
