#!/usr/bin/env python3
"""Stage 8 scripted alternating-step locomotion preview.

This script validates the software plumbing needed for alternating left/right
contact transitions. It is deliberately conservative: the reference is scripted
joint-position motion, not an FDDP walking trajectory and not hardware
validation. The output metrics are useful for the thesis claim boundary because
they separate current contact-switching evidence from future FDDP walking work.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
C2_PATH = ROOT / "mpc" / "08_run_contact_phase_synchronised_fir_qp_mpc.py"
XML = ROOT / "model" / "biped_planar.xml"
LOG = ROOT / "results" / "logs"
FIG = ROOT / "results" / "figures"

JOINTS = [
    "left_hip_pitch_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
]
TORSO = "hip_plate_link"

DURATION_S = 2.35
TOUCH_THRESHOLD_N = 0.25
COMMAND_DELAY_S = 0.020
ACTUATOR_TAU_S = 0.030
MAX_COMMAND_SPEED_RAD_S = 8.0
FALL_PITCH_DEG = 30.0
FALL_HEIGHT_M = 0.12

LEFT_STEP = {
    "leg": "left",
    "lift_start_s": 0.25,
    "lift_full_s": 0.45,
    "lower_start_s": 0.78,
    "touchdown_s": 1.00,
    "liftoff_event_s": 0.36,
}
RIGHT_STEP = {
    "leg": "right",
    "lift_start_s": 1.18,
    "lift_full_s": 1.38,
    "lower_start_s": 1.70,
    "touchdown_s": 1.95,
    "liftoff_event_s": 1.30,
}
STEP_SCHEDULE = [LEFT_STEP, RIGHT_STEP]

EXPECTED_EVENTS = [
    {"time_s": LEFT_STEP["liftoff_event_s"], "kind": "left_liftoff", "source": "scripted_alternating_step"},
    {"time_s": LEFT_STEP["touchdown_s"], "kind": "left_touchdown", "source": "scripted_alternating_step"},
    {"time_s": RIGHT_STEP["liftoff_event_s"], "kind": "right_liftoff", "source": "scripted_alternating_step"},
    {"time_s": RIGHT_STEP["touchdown_s"], "kind": "right_touchdown", "source": "scripted_alternating_step"},
]


def load_c2_module():
    spec = importlib.util.spec_from_file_location("c2_phase_sync", C2_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def step_profile(t: float, step: dict[str, Any]) -> float:
    if t < step["lift_start_s"]:
        return 0.0
    if t < step["lift_full_s"]:
        return smoothstep((t - step["lift_start_s"]) / (step["lift_full_s"] - step["lift_start_s"]))
    if t < step["lower_start_s"]:
        return 1.0
    if t < step["touchdown_s"]:
        return 1.0 - smoothstep((t - step["lower_start_s"]) / (step["touchdown_s"] - step["lower_start_s"]))
    return 0.0


def q_reference(t: float) -> np.ndarray:
    q = np.zeros(6, dtype=float)
    left = step_profile(t, LEFT_STEP)
    right = step_profile(t, RIGHT_STEP)

    # Same conservative lift pattern as the C2 single-foot mechanism demo.
    q[0] = math.radians(25.0) * left
    q[1] = math.radians(-54.0) * left
    q[2] = math.radians(25.0) * left
    q[3] = math.radians(25.0) * right
    q[4] = math.radians(-54.0) * right
    q[5] = math.radians(25.0) * right
    return q


class ActuatorProxy:
    def __init__(self, count: int, dt: float):
        self.count = int(count)
        self.dt = float(dt)
        self.delay_steps = int(round(COMMAND_DELAY_S / self.dt))
        self.history = deque(maxlen=self.delay_steps + 1)
        self.output = np.zeros(self.count, dtype=float)

    def reset(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=float).reshape(self.count)
        self.history.clear()
        for _ in range(self.delay_steps + 1):
            self.history.append(q.copy())
        self.output = q.copy()

    def update(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(self.count)
        self.history.append(q.copy())
        delayed = self.history[0]
        gain = min(1.0, self.dt / ACTUATOR_TAU_S)
        step = gain * (delayed - self.output)
        step_limit = MAX_COMMAND_SPEED_RAD_S * self.dt
        self.output = self.output + np.clip(step, -step_limit, step_limit)
        return self.output.copy()


def require_id(model, kind, name: str) -> int:
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise RuntimeError(f"Required MuJoCo name not found: {name}")
    return int(value)


def map_actuators(model, joint_ids: list[int]) -> list[int]:
    result = []
    for joint_id in joint_ids:
        candidates = [idx for idx in range(model.nu) if int(model.actuator_trnid[idx, 0]) == joint_id]
        if len(candidates) != 1:
            raise RuntimeError(f"Joint actuator mapping failed for joint id {joint_id}: {candidates}")
        result.append(candidates[0])
    return result


def simulate() -> tuple[pd.DataFrame, Any, list[dict[str, Any]], list[dict[str, Any]]]:
    c2 = load_c2_module()
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)

    joint_ids = [require_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINTS]
    qadr = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
    act_ids = map_actuators(model, joint_ids)
    torso_id = require_id(model, mujoco.mjtObj.mjOBJ_BODY, TORSO)
    root_pitch_id = require_id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_pitch_x")
    root_pitch_qadr = int(model.jnt_qposadr[root_pitch_id])

    mujoco.mj_resetData(model, data)
    for addr, q0 in zip(qadr, q_reference(0.0)):
        data.qpos[addr] = q0
    mujoco.mj_forward(model, data)

    proxy = ActuatorProxy(6, dt)
    proxy.reset(q_reference(0.0))
    observer = c2.ContactObserver(model, data, dt)
    sync = c2.ContactPhaseSynchronizer([dict(event) for event in EXPECTED_EVENTS])

    rows = []
    event_records = []
    observed_events_flat = []
    nsteps = int(round(DURATION_S / dt)) + 1

    initial_torso_y = float(data.xpos[torso_id][1])
    for _ in range(nsteps):
        wall_t = min(float(data.time), DURATION_S)
        contact_values, site_active, leg_active, observed_events = observer.update(model, data, wall_t)
        observed_events_flat.extend(observed_events)
        event_records.extend(sync.update(wall_t, observed_events))
        reference_time = sync.reference_time(wall_t, DURATION_S)

        q_ref = q_reference(reference_time)
        q_proxy = proxy.update(q_ref)
        data.ctrl[:] = 0.0
        for actuator_id, command in zip(act_ids, q_proxy):
            data.ctrl[actuator_id] = command
        mujoco.mj_step(model, data)

        torso_pos = data.xpos[torso_id].copy()
        row = {
            "time_s": float(data.time),
            "reference_time_s": float(reference_time),
            "phase_offset_s": float(sync.offset_s),
            "left_contact_active": bool(leg_active["left"]),
            "right_contact_active": bool(leg_active["right"]),
            "ncon": int(data.ncon),
            "torso_pitch_rad": float(data.qpos[root_pitch_qadr]),
            "torso_height_m": float(torso_pos[2]),
            "torso_y_m": float(torso_pos[1]),
            "torso_forward_displacement_m": float(torso_pos[1] - initial_torso_y),
            "matched_events": int(len(sync.matched)),
            "skipped_events": int(len(sync.skipped)),
        }
        for name, value in contact_values.items():
            row[name] = float(value)
        for name, active in site_active.items():
            row[f"{name}_active"] = bool(active)
        q_meas = [float(data.qpos[addr]) for addr in qadr]
        for index, name in enumerate(JOINTS):
            row[f"{name}_ref_rad"] = float(q_ref[index])
            row[f"{name}_meas_rad"] = q_meas[index]
            row[f"{name}_cmd_rad"] = float(q_proxy[index])
        rows.append(row)

    return pd.DataFrame(rows), sync, event_records, observed_events_flat


def _metric_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _observed_event_kinds(observed_events: list[dict[str, Any]]) -> set[str]:
    return {str(event.get("kind")) for event in observed_events}


def build_summary(
    rows: list[dict[str, Any]],
    observed_events: list[dict[str, Any]],
    matched_count: int,
    skipped_count: int,
    outputs: dict[str, str] | None = None,
) -> dict[str, Any]:
    pitch = _metric_array(rows, "torso_pitch_rad")
    height = _metric_array(rows, "torso_height_m")
    ncon = _metric_array(rows, "ncon")
    phase = _metric_array(rows, "phase_offset_s")
    duration = float(max(row["time_s"] for row in rows) - min(row["time_s"] for row in rows))
    observed_kinds = _observed_event_kinds(observed_events)
    expected_kinds = {event["kind"] for event in EXPECTED_EVENTS}
    forward = _metric_array(rows, "torso_forward_displacement_m") if "torso_forward_displacement_m" in rows[0] else np.zeros(len(rows))

    metrics = {
        "duration_s": duration,
        "scripted_step_count": len(STEP_SCHEDULE),
        "scheduled_contact_events": len(EXPECTED_EVENTS),
        "observed_contact_events": len(observed_events),
        "matched_contact_events": int(matched_count),
        "skipped_contact_events": int(skipped_count),
        "max_abs_phase_offset_ms": float(1000.0 * np.max(np.abs(phase))),
        "pitch_rms_deg": float(np.sqrt(np.mean(np.rad2deg(pitch) ** 2))),
        "peak_abs_pitch_deg": float(np.max(np.abs(np.rad2deg(pitch)))),
        "min_torso_height_m": float(np.min(height)),
        "min_contacts": int(np.min(ncon)),
        "forward_displacement_m": float(forward[-1]),
        "left_contact_was_lost": "left_liftoff" in observed_kinds,
        "left_contact_was_restored": "left_touchdown" in observed_kinds,
        "right_contact_was_lost": "right_liftoff" in observed_kinds,
        "right_contact_was_restored": "right_touchdown" in observed_kinds,
        "fall_detected": bool(np.max(np.abs(np.rad2deg(pitch))) > FALL_PITCH_DEG or np.min(height) < FALL_HEIGHT_M),
    }
    summary = {
        "stage": "stage8_scripted_alternating_step_locomotion_preview",
        "scope_note": (
            "Scripted alternating left/right contact-transition preview only; "
            "not FDDP walking, not hardware validation, and not a final locomotion controller."
        ),
        "step_schedule": STEP_SCHEDULE,
        "expected_events": EXPECTED_EVENTS,
        "observed_events": observed_events,
        "metrics": metrics,
        "outputs": outputs or {},
    }
    summary["acceptance"] = {
        "all_expected_event_kinds_observed": expected_kinds.issubset(observed_kinds),
        "all_expected_events_matched": matched_count == len(EXPECTED_EVENTS),
        "no_reference_events_skipped": skipped_count == 0,
        "phase_offset_bound_respected": metrics["max_abs_phase_offset_ms"] <= 60.0 + 1e-9,
        "support_contact_retained": metrics["min_contacts"] >= 1,
        "no_fall_detected": not metrics["fall_detected"],
    }
    return summary


def plot_results(df: pd.DataFrame) -> tuple[Path, Path, Path]:
    FIG.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    for prefix, label in (
        ("left_toe", "left toe"),
        ("left_heel", "left heel"),
        ("right_toe", "right toe"),
        ("right_heel", "right heel"),
    ):
        ax.plot(df.time_s, df[f"{prefix}_force_N"], label=label)
    for event in EXPECTED_EVENTS:
        ax.axvline(event["time_s"], linestyle="--", linewidth=1.0, label=f"expected {event['kind']}")
    ax.axhline(TOUCH_THRESHOLD_N, linestyle=":", color="black", label="touch threshold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Touch sensor force (N)")
    ax.set_title("Stage 8 scripted alternating-step contact transitions")
    ax.grid(True)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    contacts_path = FIG / "stage8_scripted_alternating_step_contacts.png"
    fig.savefig(contacts_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.time_s, 1000.0 * df.phase_offset_s, label="C2 phase offset")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Phase offset (ms)")
    ax.set_title("Stage 8 contact-event phase correction")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    phase_path = FIG / "stage8_scripted_alternating_step_phase_offset.png"
    fig.savefig(phase_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(df.time_s, np.rad2deg(df.torso_pitch_rad), label="torso pitch")
    axes[0].set_ylabel("Pitch (deg)")
    axes[0].grid(True)
    axes[1].plot(df.time_s, df.torso_height_m, label="torso height", color="tab:green")
    axes[1].axhline(FALL_HEIGHT_M, color="black", linestyle=":", label="fall threshold")
    axes[1].set_ylabel("Height (m)")
    axes[1].grid(True)
    axes[2].plot(df.time_s, df.torso_forward_displacement_m, label="forward displacement", color="tab:purple")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Y displacement (m)")
    axes[2].grid(True)
    for axis in axes:
        axis.legend(loc="best")
    fig.suptitle("Stage 8 scripted alternating-step body metrics")
    fig.tight_layout()
    body_path = FIG / "stage8_scripted_alternating_step_body.png"
    fig.savefig(body_path, dpi=180)
    plt.close(fig)
    return contacts_path, phase_path, body_path


def main() -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    df, sync, event_records, observed_events = simulate()
    csv_path = LOG / "stage8_scripted_alternating_step_locomotion.csv"
    df.to_csv(csv_path, index=False)
    contacts_path, phase_path, body_path = plot_results(df)

    outputs = {
        "csv": str(csv_path),
        "contacts_plot": str(contacts_path),
        "phase_offset_plot": str(phase_path),
        "body_plot": str(body_path),
    }
    summary = build_summary(
        df.to_dict(orient="records"),
        observed_events,
        matched_count=len(sync.matched),
        skipped_count=len(sync.skipped),
        outputs=outputs,
    )
    summary["matched_event_records"] = sync.matched
    summary["skipped_event_records"] = sync.skipped
    summary["event_records"] = event_records

    summary_path = LOG / "stage8_scripted_alternating_step_locomotion_summary.yaml"
    summary["outputs"]["summary"] = str(summary_path)
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")

    print("Stage 8 scripted alternating-step locomotion preview")
    print(f"Matched events: {summary['metrics']['matched_contact_events']}/{summary['metrics']['scheduled_contact_events']}")
    print(f"Observed events: {summary['metrics']['observed_contact_events']}")
    print(f"Pitch RMS: {summary['metrics']['pitch_rms_deg']:.3f} deg")
    print(f"Forward displacement: {summary['metrics']['forward_displacement_m']:.4f} m")
    print(f"Max phase offset: {summary['metrics']['max_abs_phase_offset_ms']:.3f} ms")
    print(f"Saved summary: {summary_path}")
    if not all(summary["acceptance"].values()):
        print(summary["acceptance"])
        raise SystemExit("Stage 8 scripted alternating-step preview failed acceptance checks.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"STAGE 8 SCRIPTED ALTERNATING-STEP FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
