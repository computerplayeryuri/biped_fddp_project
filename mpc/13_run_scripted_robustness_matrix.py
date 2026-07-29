#!/usr/bin/env python3
"""Stage 9 software robustness matrix for the scripted C2 walking preview.

This script extends the Stage 8 alternating-step contact-transition preview into
repeatable software stress tests. It deliberately does not claim a full FDDP
walking gait. The current reference is a periodic scripted joint-position
preview used to test the C2 contact-event timing stack, actuator proxy, and
MuJoCo plant under longer and rougher software-only conditions.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import asdict, dataclass
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
PUSH_BODY = "electronics_lump_link"

CYCLE_S = 2.0
DURATION_SCENARIOS_S = (10.0, 30.0, 60.0)
INITIAL_FOOT_CENTER_Y_M = 0.0
FOOT_HALF_LENGTH_Y_M = 0.0725
STEP_OBSTACLE_HALF_LENGTH_Y_M = 0.012
MINIMUM_STEP_INITIAL_CLEARANCE_M = 0.005


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
STEP_SCHEDULE = (LEFT_STEP, RIGHT_STEP)

COMMAND_DELAY_S = 0.020
ACTUATOR_TAU_S = 0.030
MAX_COMMAND_SPEED_RAD_S = 8.0

FALL_PITCH_DEG = 30.0
FALL_HEIGHT_M = 0.12
MIN_MATCH_RATIO = 0.85
MAX_PHASE_OFFSET_MS = 90.0

DEFAULT_FRICTION_MU = 0.80
DEFAULT_PUSH_START_S = 2.0
DEFAULT_PUSH_DURATION_S = 0.12


@dataclass(frozen=True)
class Scenario:
    label: str
    category: str
    duration_s: float = 10.0
    friction_mu: float = DEFAULT_FRICTION_MU
    slope_deg: float = 0.0
    step_height_m: float = 0.0
    push_force_n: float = 0.0
    push_start_s: float = DEFAULT_PUSH_START_S
    push_duration_s: float = DEFAULT_PUSH_DURATION_S


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


def load_c2_module():
    spec = importlib.util.spec_from_file_location("c2_phase_sync", C2_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def step_profile_at_cycle_time(cycle_t: float, step: dict[str, Any]) -> float:
    if cycle_t < step["lift_start_s"]:
        return 0.0
    if cycle_t < step["lift_full_s"]:
        return smoothstep((cycle_t - step["lift_start_s"]) / (step["lift_full_s"] - step["lift_start_s"]))
    if cycle_t < step["lower_start_s"]:
        return 1.0
    if cycle_t < step["touchdown_s"]:
        return 1.0 - smoothstep((cycle_t - step["lower_start_s"]) / (step["touchdown_s"] - step["lower_start_s"]))
    return 0.0


def q_reference(t: float) -> np.ndarray:
    cycle_t = float(t % CYCLE_S)
    q = np.zeros(6, dtype=float)
    left = step_profile_at_cycle_time(cycle_t, LEFT_STEP)
    right = step_profile_at_cycle_time(cycle_t, RIGHT_STEP)

    q[0] = math.radians(25.0) * left
    q[1] = math.radians(-54.0) * left
    q[2] = math.radians(25.0) * left
    q[3] = math.radians(25.0) * right
    q[4] = math.radians(-54.0) * right
    q[5] = math.radians(25.0) * right
    return q


def expected_events_for_duration(duration_s: float) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    cycle_index = 0
    cycle_start = 0.0
    while cycle_start <= duration_s + 1e-12:
        for step in STEP_SCHEDULE:
            liftoff_time = cycle_start + float(step["liftoff_event_s"])
            touchdown_time = cycle_start + float(step["touchdown_s"])
            if liftoff_time <= duration_s + 1e-12:
                events.append(
                    {
                        "time_s": liftoff_time,
                        "kind": f"{step['leg']}_liftoff",
                        "source": "stage9_periodic_scripted_step",
                        "cycle_index": cycle_index,
                    }
                )
            if touchdown_time <= duration_s + 1e-12:
                events.append(
                    {
                        "time_s": touchdown_time,
                        "kind": f"{step['leg']}_touchdown",
                        "source": "stage9_periodic_scripted_step",
                        "cycle_index": cycle_index,
                    }
                )
        cycle_index += 1
        cycle_start = cycle_index * CYCLE_S
    return sorted(events, key=lambda item: item["time_s"])


def _fmt_decimal(value: float, decimals: int = 2) -> str:
    text = f"{value:.{decimals}f}"
    return text.replace("-", "minus").replace(".", "p")


def _force_label(force_n: float) -> str:
    sign = "plus" if force_n > 0 else "minus"
    return f"push_{sign}{int(abs(force_n))}N"


def build_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = [
        Scenario(label=f"duration_{int(duration)}s", category="duration", duration_s=duration)
        for duration in DURATION_SCENARIOS_S
    ]

    for mu in (0.50, 0.65, 0.95, 1.10):
        scenarios.append(
            Scenario(
                label=f"friction_mu_{_fmt_decimal(mu)}",
                category="friction",
                friction_mu=mu,
            )
        )

    for slope in (-6.0, -3.0, 3.0, 6.0):
        sign = "plus" if slope > 0 else "minus"
        scenarios.append(
            Scenario(
                label=f"slope_{sign}{int(abs(slope))}deg",
                category="slope_equivalent",
                slope_deg=slope,
            )
        )

    for height_mm in (2.0, 4.0, 6.0):
        scenarios.append(
            Scenario(
                label=f"step_{int(height_mm)}mm",
                category="small_step",
                step_height_m=height_mm / 1000.0,
            )
        )

    for force_n in (-6.0, -4.0, -2.0, 2.0, 4.0, 6.0):
        scenarios.append(
            Scenario(
                label=_force_label(force_n),
                category="push",
                push_force_n=force_n,
            )
        )

    return scenarios


def step_obstacle_geometry(
    initial_clearance_m: float = MINIMUM_STEP_INITIAL_CLEARANCE_M,
) -> dict[str, float]:
    clearance = float(initial_clearance_m)
    if (
        not np.isfinite(clearance)
        or clearance < MINIMUM_STEP_INITIAL_CLEARANCE_M
    ):
        raise ValueError(
            "step obstacle initial clearance must be at least 0.005 m"
        )
    initial_toe = INITIAL_FOOT_CENTER_Y_M + FOOT_HALF_LENGTH_Y_M
    rear_edge = initial_toe + clearance
    center = rear_edge + STEP_OBSTACLE_HALF_LENGTH_Y_M
    return {
        "initial_foot_center_y_m": INITIAL_FOOT_CENTER_Y_M,
        "initial_toe_y_m": initial_toe,
        "obstacle_half_length_y_m": STEP_OBSTACLE_HALF_LENGTH_Y_M,
        "obstacle_rear_edge_y_m": rear_edge,
        "initial_clearance_m": clearance,
        "obstacle_center_y_m": center,
    }


def build_step_xml(step_height_m: float) -> str:
    if step_height_m <= 0.0:
        return XML.read_text(encoding="utf-8")

    tree = ET.ElementTree(ET.fromstring(XML.read_text(encoding="utf-8")))
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("MuJoCo XML has no worldbody.")

    half_height = 0.5 * float(step_height_m)
    geometry = step_obstacle_geometry()
    for side, x_pos in (("left", -0.0525), ("right", 0.0525)):
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": f"stage9_{side}_toe_step",
                "type": "box",
                "pos": (
                    f"{x_pos:.4f} "
                    f"{geometry['obstacle_center_y_m']:.4f} "
                    f"{half_height:.6g}"
                ),
                "size": (
                    f"0.050 {geometry['obstacle_half_length_y_m']:.4f} "
                    f"{half_height:.6g}"
                ),
                "material": "ground",
                "contype": "1",
                "conaffinity": "1",
                "friction": "0.8 0.01 0.001",
            },
        )

    return ET.tostring(root, encoding="unicode")


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


def make_model(scenario: Scenario):
    if scenario.step_height_m > 0.0:
        model = mujoco.MjModel.from_xml_string(build_step_xml(scenario.step_height_m))
    else:
        model = mujoco.MjModel.from_xml_path(str(XML))

    friction = np.array([scenario.friction_mu, 0.01, 0.001], dtype=float)
    for geom_name in ("floor", "left_foot_geom", "right_foot_geom", "stage9_left_toe_step", "stage9_right_toe_step"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            model.geom_friction[geom_id, :] = friction

    if abs(scenario.slope_deg) > 1e-12:
        theta = math.radians(scenario.slope_deg)
        model.opt.gravity[:] = [0.0, -9.81 * math.sin(theta), -9.81 * math.cos(theta)]

    return model


def scenario_passed(metrics: dict[str, Any]) -> dict[str, Any]:
    scheduled = int(metrics.get("scheduled_contact_events", 0))
    matched = int(metrics.get("matched_contact_events", 0))
    ratio = 1.0 if scheduled == 0 else float(matched / max(scheduled, 1))
    acceptance = {
        "no_fall_detected": not bool(metrics.get("fall_detected", False)),
        "support_contact_retained": int(metrics.get("min_contacts", 0)) >= 1,
        "pitch_bound_respected": float(metrics.get("peak_abs_pitch_deg", float("inf"))) <= FALL_PITCH_DEG,
        "height_bound_respected": float(metrics.get("min_torso_height_m", 0.0)) >= FALL_HEIGHT_M,
        "event_match_ratio_ok": ratio >= MIN_MATCH_RATIO,
        "phase_offset_bound_respected": float(metrics.get("max_abs_phase_offset_ms", float("inf"))) <= MAX_PHASE_OFFSET_MS,
        "matched_event_ratio": ratio,
    }
    bool_flags = [value for key, value in acceptance.items() if key != "matched_event_ratio"]
    acceptance["robustness_pass"] = bool(all(bool_flags))
    return acceptance


def _metric_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def build_metrics(
    scenario: Scenario,
    rows: list[dict[str, Any]],
    observed_events: list[dict[str, Any]],
    matched_count: int,
    skipped_count: int,
) -> dict[str, Any]:
    if not rows:
        raise RuntimeError(f"{scenario.label}: no rows recorded.")

    pitch = _metric_array(rows, "torso_pitch_rad")
    height = _metric_array(rows, "torso_height_m")
    contacts = _metric_array(rows, "ncon")
    phase = _metric_array(rows, "phase_offset_s")
    forward = _metric_array(rows, "torso_forward_displacement_m")
    time_s = _metric_array(rows, "time_s")
    fall_flags = _metric_array(rows, "fall_detected")

    scheduled = len(expected_events_for_duration(scenario.duration_s))
    return {
        "scenario_label": scenario.label,
        "scenario_category": scenario.category,
        "duration_s": float(time_s[-1] - time_s[0]),
        "configured_duration_s": float(scenario.duration_s),
        "friction_mu": float(scenario.friction_mu),
        "slope_deg": float(scenario.slope_deg),
        "step_height_m": float(scenario.step_height_m),
        "push_force_n": float(scenario.push_force_n),
        "scripted_step_count": int(2 * math.floor(scenario.duration_s / CYCLE_S)),
        "scheduled_contact_events": int(scheduled),
        "observed_contact_events": int(len(observed_events)),
        "matched_contact_events": int(matched_count),
        "skipped_contact_events": int(skipped_count),
        "max_abs_phase_offset_ms": float(1000.0 * np.max(np.abs(phase))),
        "pitch_rms_deg": float(np.sqrt(np.mean(np.rad2deg(pitch) ** 2))),
        "peak_abs_pitch_deg": float(np.max(np.abs(np.rad2deg(pitch)))),
        "min_torso_height_m": float(np.min(height)),
        "min_contacts": int(np.min(contacts)),
        "forward_displacement_m": float(forward[-1]),
        "fall_detected": bool(np.max(fall_flags) > 0.5),
    }


def run_scenario(scenario: Scenario, c2_module=None) -> dict[str, Any]:
    c2 = c2_module or load_c2_module()
    model = make_model(scenario)
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)

    joint_ids = [require_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINTS]
    qadr = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
    act_ids = map_actuators(model, joint_ids)
    torso_id = require_id(model, mujoco.mjtObj.mjOBJ_BODY, TORSO)
    root_pitch_id = require_id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_pitch_x")
    root_pitch_qadr = int(model.jnt_qposadr[root_pitch_id])
    push_body_id = require_id(model, mujoco.mjtObj.mjOBJ_BODY, PUSH_BODY)

    mujoco.mj_resetData(model, data)
    for addr, q0 in zip(qadr, q_reference(0.0)):
        data.qpos[addr] = q0
    mujoco.mj_forward(model, data)

    proxy = ActuatorProxy(6, dt)
    proxy.reset(q_reference(0.0))
    observer = c2.ContactObserver(model, data, dt)
    sync = c2.ContactPhaseSynchronizer(expected_events_for_duration(scenario.duration_s))

    rows: list[dict[str, Any]] = []
    observed_events_flat: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    nsteps = int(round(scenario.duration_s / dt)) + 1
    initial_torso_y = float(data.xpos[torso_id][1])
    fell = False

    for _ in range(nsteps):
        wall_t = min(float(data.time), scenario.duration_s)
        contact_values, site_active, leg_active, observed_events = observer.update(model, data, wall_t)
        observed_events_flat.extend(observed_events)
        event_records.extend(sync.update(wall_t, observed_events))
        reference_time = sync.reference_time(wall_t, scenario.duration_s)

        q_ref = q_reference(reference_time)
        q_proxy = proxy.update(q_ref)
        data.ctrl[:] = 0.0
        for actuator_id, command in zip(act_ids, q_proxy):
            data.ctrl[actuator_id] = command

        data.xfrc_applied[:] = 0.0
        force_active = scenario.push_start_s <= wall_t < (scenario.push_start_s + scenario.push_duration_s)
        if force_active and abs(scenario.push_force_n) > 1e-12:
            data.xfrc_applied[push_body_id, 1] = scenario.push_force_n

        mujoco.mj_step(model, data)

        torso_pos = data.xpos[torso_id].copy()
        pitch = float(data.qpos[root_pitch_qadr])
        fallen_now = bool(abs(math.degrees(pitch)) > FALL_PITCH_DEG or float(torso_pos[2]) < FALL_HEIGHT_M)
        fell = fell or fallen_now
        row = {
            "time_s": float(data.time),
            "reference_time_s": float(reference_time),
            "phase_offset_s": float(sync.offset_s),
            "left_contact_active": bool(leg_active["left"]),
            "right_contact_active": bool(leg_active["right"]),
            "ncon": int(data.ncon),
            "torso_pitch_rad": pitch,
            "torso_height_m": float(torso_pos[2]),
            "torso_y_m": float(torso_pos[1]),
            "torso_forward_displacement_m": float(torso_pos[1] - initial_torso_y),
            "matched_events": int(len(sync.matched)),
            "skipped_events": int(len(sync.skipped)),
            "push_force_N": float(scenario.push_force_n if force_active else 0.0),
            "fall_detected": int(fallen_now),
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

        if fell:
            break

    metrics = build_metrics(
        scenario,
        rows,
        observed_events_flat,
        matched_count=len(sync.matched),
        skipped_count=len(sync.skipped),
    )
    acceptance = scenario_passed(metrics)

    LOG.mkdir(parents=True, exist_ok=True)
    csv_path = LOG / f"stage9_{scenario.label}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return {
        "scenario": asdict(scenario),
        "metrics": metrics,
        "acceptance": acceptance,
        "outputs": {"csv": str(csv_path)},
        "matched_event_records": sync.matched,
        "skipped_event_records": sync.skipped,
        "event_records": event_records,
    }


def plot_matrix(table: pd.DataFrame) -> tuple[str, str]:
    FIG.mkdir(parents=True, exist_ok=True)

    colors = ["tab:green" if value else "tab:red" for value in table["robustness_pass"]]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(table["label"], [1 if value else 0 for value in table["robustness_pass"]], color=colors)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Pass flag")
    ax.set_title("Stage 9 scripted robustness matrix pass/fail")
    ax.tick_params(axis="x", rotation=65)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    pass_path = FIG / "stage9_scripted_robustness_matrix_pass_fail.png"
    fig.savefig(pass_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(table["label"], table["peak_abs_pitch_deg"], color="tab:blue")
    ax.axhline(FALL_PITCH_DEG, linestyle=":", color="black", label="fall pitch threshold")
    ax.set_ylabel("Peak abs pitch (deg)")
    ax.set_title("Stage 9 scripted robustness matrix peak pitch")
    ax.tick_params(axis="x", rotation=65)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    pitch_path = FIG / "stage9_scripted_robustness_matrix_pitch.png"
    fig.savefig(pitch_path, dpi=180)
    plt.close(fig)

    return str(pass_path), str(pitch_path)


def run_matrix(scenarios: list[Scenario]) -> dict[str, Any]:
    LOG.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    c2 = load_c2_module()

    results = []
    table_rows = []
    start = time.perf_counter()
    for index, scenario in enumerate(scenarios, start=1):
        print(f"[{index:02d}/{len(scenarios):02d}] {scenario.label} ({scenario.category})")
        result = run_scenario(scenario, c2_module=c2)
        results.append(result)
        metrics = result["metrics"]
        acceptance = result["acceptance"]
        table_rows.append(
            {
                "label": scenario.label,
                "category": scenario.category,
                "duration_s": metrics["duration_s"],
                "friction_mu": scenario.friction_mu,
                "slope_deg": scenario.slope_deg,
                "step_height_m": scenario.step_height_m,
                "push_force_n": scenario.push_force_n,
                "robustness_pass": bool(acceptance["robustness_pass"]),
                "matched_event_ratio": acceptance["matched_event_ratio"],
                "matched_contact_events": metrics["matched_contact_events"],
                "scheduled_contact_events": metrics["scheduled_contact_events"],
                "peak_abs_pitch_deg": metrics["peak_abs_pitch_deg"],
                "pitch_rms_deg": metrics["pitch_rms_deg"],
                "min_torso_height_m": metrics["min_torso_height_m"],
                "min_contacts": metrics["min_contacts"],
                "max_abs_phase_offset_ms": metrics["max_abs_phase_offset_ms"],
                "fall_detected": metrics["fall_detected"],
            }
        )
        state = "PASS" if acceptance["robustness_pass"] else "FAIL"
        print(
            f"  {state}: matched {metrics['matched_contact_events']}/{metrics['scheduled_contact_events']}, "
            f"peak pitch {metrics['peak_abs_pitch_deg']:.2f} deg, "
            f"phase {metrics['max_abs_phase_offset_ms']:.1f} ms, "
            f"min height {metrics['min_torso_height_m']:.3f} m"
        )

    table = pd.DataFrame(table_rows)
    table_path = LOG / "stage9_scripted_robustness_matrix_table.csv"
    table.to_csv(table_path, index=False)
    pass_plot, pitch_plot = plot_matrix(table)

    pass_count = int(table["robustness_pass"].sum())
    total_count = int(len(table))
    summary = {
        "stage": "stage9_scripted_periodic_robustness_matrix",
        "scope_note": (
            "Software-only robustness matrix for the scripted C2 alternating-step preview. "
            "This is not a full FDDP walking trajectory and not hardware validation."
        ),
        "acceptance_thresholds": {
            "fall_pitch_deg": FALL_PITCH_DEG,
            "fall_height_m": FALL_HEIGHT_M,
            "min_match_ratio": MIN_MATCH_RATIO,
            "max_phase_offset_ms": MAX_PHASE_OFFSET_MS,
        },
        "aggregate": {
            "scenario_count": total_count,
            "pass_count": pass_count,
            "fail_count": total_count - pass_count,
            "all_scenarios_passed": pass_count == total_count,
            "wall_clock_runtime_s": float(time.perf_counter() - start),
        },
        "outputs": {
            "table": str(table_path),
            "pass_fail_plot": pass_plot,
            "peak_pitch_plot": pitch_plot,
        },
        "scenario_results": results,
    }
    summary_path = LOG / "stage9_scripted_robustness_matrix_summary.yaml"
    summary["outputs"]["summary"] = str(summary_path)
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print(f"Saved summary: {summary_path}")
    print(f"Pass count: {pass_count}/{total_count}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 9 scripted robustness matrix.")
    parser.add_argument("--matrix", choices=("smoke", "full"), default="full")
    args = parser.parse_args()

    scenarios = build_scenarios()
    if args.matrix == "smoke":
        scenarios = [
            Scenario(label="smoke_nominal_4s", category="smoke", duration_s=4.0),
            Scenario(label="smoke_push_plus2N", category="smoke", duration_s=4.0, push_force_n=2.0),
        ]

    summary = run_matrix(scenarios)
    if not summary["aggregate"]["all_scenarios_passed"]:
        raise SystemExit("Stage 9 robustness matrix completed with one or more failed scenarios.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"STAGE 9 ROBUSTNESS MATRIX FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
