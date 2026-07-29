#!/usr/bin/env python3
"""Common-protocol C0/C1/phase-only/C2-final ten-step MuJoCo ablation."""

from __future__ import annotations

from collections import deque
import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence
from types import SimpleNamespace

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_PATH = ROOT / "mpc" / "multistep_controller.py"
REPLAY_PATH = ROOT / "simulation" / "03u_replay_fddp_multistep_mujoco.py"
PROXY_PATH = ROOT / "simulation" / "03o_replay_fddp_com_shift_mujoco.py"
PHASE_PATH = ROOT / "mpc" / "08_run_contact_phase_synchronised_fir_qp_mpc.py"
TERRAIN_PATH = ROOT / "mpc" / "13_run_scripted_robustness_matrix.py"
CONFIG_PATH = ROOT / "config" / "fddp_multistep_experiment.yaml"
XML_PATH = ROOT / "model" / "biped_planar.xml"
LOG_DIR = ROOT / "results" / "logs"
FIGURE_DIR = ROOT / "results" / "figures"
CSV_PATH = LOG_DIR / "fddp_multistep_controller_ablation.csv"
YAML_PATH = LOG_DIR / "fddp_multistep_controller_ablation.yaml"
FIGURE_PATH = FIGURE_DIR / "fddp_multistep_controller_ablation.png"

SCHEMA_VERSION = "fddp-multistep-controller-ablation-v1"
MODES = ("C0", "C1", "phase_only", "C2_final")
DISTURBANCE_TIME_S = 0.40
DISTURBANCE_PITCH_RAD = math.radians(2.0)
DISTURBANCE_RATE_RAD_S = 0.25


def _load_existing(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load existing module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


controller_runtime = _load_existing(
    CONTROLLER_PATH, "multistep_ablation_controller"
)
replay = _load_existing(REPLAY_PATH, "multistep_ablation_replay")
proxy = _load_existing(PROXY_PATH, "multistep_ablation_proxy")
phase_runtime = _load_existing(PHASE_PATH, "multistep_ablation_phase")
terrain_runtime = _load_existing(
    TERRAIN_PATH, "multistep_ablation_terrain"
)
DEFAULT_COMMAND_DELAY_S = float(proxy.COMMAND_DELAY_S)
DEFAULT_ACTUATOR_TIME_CONSTANT_S = float(
    proxy.FIRST_ORDER_TIME_CONSTANT_S
)


def _build_robustness_model(
    friction_mu: float = 0.80,
    slope_deg: float = 0.0,
    step_height_m: float = 0.0,
):
    """Reuse the existing terrain injector for the ten-step plant."""
    return terrain_runtime.make_model(SimpleNamespace(
        friction_mu=float(friction_mu),
        slope_deg=float(slope_deg),
        step_height_m=float(step_height_m),
    ))


def _controller_thresholds() -> dict[str, float]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return dict(config["controller"])


def _resolve_foot_terrain_contact_ids(model) -> dict[str, object]:
    import mujoco

    feet = {}
    for side in ("left", "right"):
        geom_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"{side}_foot_geom",
        )
        if geom_id < 0:
            raise RuntimeError(f"missing {side}_foot_geom")
        feet[side] = int(geom_id)
    terrain_ids = set()
    obstacle_ids = set()
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        )
        if name == "floor":
            terrain_ids.add(int(geom_id))
        elif name is not None and name.startswith("stage9_"):
            terrain_ids.add(int(geom_id))
            obstacle_ids.add(int(geom_id))
    if not terrain_ids:
        raise RuntimeError("no floor or step terrain geometry found")
    return {
        "feet": feet,
        "terrain": frozenset(terrain_ids),
        "obstacles": frozenset(obstacle_ids),
    }


def _geometry_leg_contacts(
    data,
    contact_ids: Mapping[str, object],
) -> dict[str, bool]:
    feet = dict(contact_ids["feet"])
    terrain = set(contact_ids["terrain"])
    active = {"left": False, "right": False}
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        for side, foot_id in feet.items():
            if int(foot_id) in pair and bool(pair & terrain):
                active[side] = True
    return active


def _obstacle_contact_snapshot(
    data,
    contact_ids: Mapping[str, object],
) -> dict[str, object]:
    feet = set(dict(contact_ids["feet"]).values())
    obstacles = set(contact_ids["obstacles"])
    contact_count = 0
    maximum_penetration_m = 0.0
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if bool(pair & feet) and bool(pair & obstacles):
            contact_count += 1
            maximum_penetration_m = max(
                maximum_penetration_m,
                max(0.0, -float(contact.dist)),
            )
    return {
        "detected": bool(contact_count),
        "contact_count": contact_count,
        "maximum_penetration_m": maximum_penetration_m,
    }


def evaluate_ablation(
    mode_metrics: Mapping[str, Mapping[str, float]]
) -> dict:
    """Apply the approved Task 6 comparison gates with inclusive boundaries."""

    missing = sorted(set(MODES) - set(mode_metrics))
    if missing:
        raise KeyError(f"missing controller modes: {missing}")
    thresholds = _controller_thresholds()
    c0 = mode_metrics["C0"]
    c1 = mode_metrics["C1"]
    c2 = mode_metrics["C2_final"]
    epsilon = 1.0e-12
    checks = {
        "c1_iae_improvement": float(c1["iae_deg_s"])
        <= float(c0["iae_deg_s"])
        * (1.0 - float(thresholds["c1_minimum_iae_improvement_fraction"]))
        + epsilon,
        "c1_peak_ratio": float(c1["peak_abs_error_deg"])
        <= float(c0["peak_abs_error_deg"])
        * float(thresholds["c1_maximum_peak_ratio"])
        + epsilon,
        "c2_iae_ratio": float(c2["iae_deg_s"])
        <= float(c1["iae_deg_s"])
        * float(thresholds["c2_maximum_iae_to_c1_ratio"])
        + epsilon,
        "c2_peak_ratio": float(c2["peak_abs_error_deg"])
        <= float(c1["peak_abs_error_deg"])
        * float(thresholds["c2_maximum_peak_to_c1_ratio"])
        + epsilon,
        "c2_contact_timing_improvement": float(
            c2["mean_abs_contact_timing_error_s"]
        )
        <= float(c1["mean_abs_contact_timing_error_s"])
        * (
            1.0
            - float(
                thresholds[
                    "c2_minimum_contact_timing_improvement_fraction"
                ]
            )
        )
        + epsilon,
    }
    numeric = [
        float(metrics[key])
        for metrics in mode_metrics.values()
        for key in (
            "iae_deg_s",
            "peak_abs_error_deg",
            "mean_abs_contact_timing_error_s",
        )
    ]
    if not np.all(np.isfinite(numeric)):
        checks["finite_metrics"] = False
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "thresholds": thresholds,
    }


def _event_metrics(mode, frame, scheduled_events, raw_matching):
    scheduled_count = int(len(scheduled_events))
    raw_metrics = {
        "raw_matched_contact_events": int(raw_matching["matched_count"]),
        "raw_scheduled_contact_events": int(raw_matching["scheduled_count"]),
        "raw_event_match_ratio": float(raw_matching["match_ratio"]),
        "raw_event_order_valid": bool(raw_matching["event_order_valid"]),
    }
    if frame.empty:
        phase_matched = 0
        phase_skipped = 0
    else:
        phase_matched = int(frame["matched_events"].iloc[-1])
        phase_skipped = int(frame["skipped_events"].iloc[-1])
    phase_matched = min(max(phase_matched, 0), scheduled_count)
    phase_skipped = min(max(phase_skipped, 0), scheduled_count)
    phase_ratio = (
        1.0 if scheduled_count == 0 else phase_matched / scheduled_count
    )
    phase_order_valid = bool(
        phase_matched + phase_skipped <= scheduled_count
    )
    phase_metrics = {
        "phase_aligned_matched_contact_events": phase_matched,
        "phase_aligned_skipped_contact_events": phase_skipped,
        "phase_aligned_event_match_ratio": float(phase_ratio),
        "phase_aligned_event_order_valid": phase_order_valid,
    }
    if mode in {"phase_only", "C2_final"}:
        primary = {
            "matched_contact_events": phase_matched,
            "scheduled_contact_events": scheduled_count,
            "event_match_ratio": float(phase_ratio),
            "event_order_valid": phase_order_valid,
        }
    else:
        primary = {
            "matched_contact_events": raw_metrics["raw_matched_contact_events"],
            "scheduled_contact_events": raw_metrics["raw_scheduled_contact_events"],
            "event_match_ratio": raw_metrics["raw_event_match_ratio"],
            "event_order_valid": raw_metrics["raw_event_order_valid"],
        }
    return {**primary, **phase_metrics, **raw_metrics}


def _contact_timing_metrics(mode, controller, raw_timing_errors):
    raw_mean = (
        float(np.mean(raw_timing_errors))
        if raw_timing_errors.size
        else np.inf
    )
    synchronizer = getattr(controller, "_synchronizers", {}).get(mode)
    matched_records = [] if synchronizer is None else synchronizer.matched
    phase_errors = np.asarray(
        [
            abs(float(record["timing_error_s"]))
            for record in matched_records
        ],
        dtype=float,
    )
    phase_mean = (
        float(np.mean(phase_errors))
        if phase_errors.size
        else np.inf
    )
    primary_mean = raw_mean
    return {
        "mean_abs_contact_timing_error_s": primary_mean,
        "phase_aligned_mean_abs_contact_timing_error_s": phase_mean,
        "raw_mean_abs_contact_timing_error_s": raw_mean,
    }
def _run_mode(
    mode: str,
    duration_s: float,
    *,
    push_force_n: float = 0.0,
    push_start_s: float = 0.80,
    push_duration_s: float = 0.12,
    seed: int = 0,
    initial_joint_noise_std_rad: float = 0.0,
    initial_pitch_noise_std_rad: float = 0.0,
    friction_mu: float = 0.80,
    slope_deg: float = 0.0,
    step_height_m: float = 0.0,
    command_delay_s: float | None = None,
    sensor_delay_s: float = 0.0,
    actuator_gain: float = 1.0,
    actuator_time_constant_s: float | None = None,
    controller_kwargs: Mapping[str, object] | None = None,
) -> dict:
    import mujoco
    import pandas as pd

    command_delay_s = float(
        DEFAULT_COMMAND_DELAY_S
        if command_delay_s is None else command_delay_s
    )
    actuator_time_constant_s = float(
        DEFAULT_ACTUATOR_TIME_CONSTANT_S
        if actuator_time_constant_s is None
        else actuator_time_constant_s
    )
    robustness_configuration = {
        "friction_mu": float(friction_mu),
        "slope_deg": float(slope_deg),
        "step_height_m": float(step_height_m),
        "command_delay_s": command_delay_s,
        "sensor_delay_s": float(sensor_delay_s),
        "actuator_gain": float(actuator_gain),
        "actuator_time_constant_s": actuator_time_constant_s,
        "initial_joint_noise_std_rad": float(
            initial_joint_noise_std_rad
        ),
        "initial_pitch_noise_std_rad": float(
            initial_pitch_noise_std_rad
        ),
    }
    if not all(
        np.isfinite(value)
        for value in robustness_configuration.values()
    ):
        raise ValueError("robustness configuration contains non-finite values")
    if (
        friction_mu <= 0.0
        or step_height_m < 0.0
        or command_delay_s < 0.0
        or sensor_delay_s < 0.0
        or actuator_gain <= 0.0
        or actuator_time_constant_s <= 0.0
        or initial_joint_noise_std_rad < 0.0
        or initial_pitch_noise_std_rad < 0.0
    ):
        raise ValueError("invalid robustness configuration")
    controller = controller_runtime.MultiStepController(
        **dict(controller_kwargs or {})
    )
    reference = controller.reference
    scheduled_events = [
        event
        for event in replay.build_reference_events(reference)
        if float(event["time_s"]) <= duration_s + 1.0e-12
    ]

    model = _build_robustness_model(
        friction_mu=friction_mu,
        slope_deg=slope_deg,
        step_height_m=step_height_m,
    )
    data = mujoco.MjData(model)
    mapping = proxy.resolve_joint_and_actuator_mapping(model)
    torso_body_id = proxy.resolve_torso_body(model)
    pitch_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "root_pitch_x"
    )
    if pitch_joint_id < 0:
        raise RuntimeError("root_pitch_x joint is missing")
    pitch_qpos_address = int(model.jnt_qposadr[pitch_joint_id])
    pitch_dof_address = int(model.jnt_dofadr[pitch_joint_id])

    mujoco.mj_resetData(model, data)
    rng = np.random.default_rng(seed)
    for address, value in zip(
        mapping["qpos_addresses"], reference.q_ref[0]
    ):
        noise = float(rng.normal(0.0, initial_joint_noise_std_rad))
        data.qpos[address] = float(value) + noise
    data.qpos[pitch_qpos_address] += float(
        rng.normal(0.0, initial_pitch_noise_std_rad)
    )
    mujoco.mj_forward(model, data)

    dt_s = float(model.opt.timestep)
    observer = phase_runtime.ContactObserver(model, data, dt_s)
    terrain_contact_ids = (
        _resolve_foot_terrain_contact_ids(model)
        if float(step_height_m) > 0.0
        else None
    )
    if terrain_contact_ids is not None:
        initial_obstacle_contact = _obstacle_contact_snapshot(
            data, terrain_contact_ids
        )
        if bool(initial_obstacle_contact["detected"]):
            raise RuntimeError(
                "step obstacle intersects a foot at initialisation"
            )
    fused_leg_active = dict(observer.leg_active)
    if terrain_contact_ids is not None:
        geometry_active = _geometry_leg_contacts(
            data, terrain_contact_ids
        )
        fused_leg_active = {
            side: bool(fused_leg_active[side] or geometry_active[side])
            for side in ("left", "right")
        }
    fusion_pending_counts = {"left": 0, "right": 0}
    proxy.COMMAND_DELAY_S = command_delay_s
    proxy.FIRST_ORDER_TIME_CONSTANT_S = actuator_time_constant_s
    actuator = proxy.PositionActuatorProxy(6, dt_s)
    actuator.reset(reference.q_ref[0])
    update_steps = max(
        1, int(round(controller_runtime.runtime.TS / dt_s))
    )
    total_steps = int(round(duration_s / dt_s))
    initial_height_m = float(data.xpos[torso_body_id][2])
    output = None
    q_command = reference.q_ref[0].copy()
    disturbed = False
    observed_events = []
    rows = []
    flight_streak_steps = 0
    maximum_flight_steps = 0
    step_obstacle_contact_sample_count = 0
    first_step_obstacle_contact_time_s = None
    maximum_step_obstacle_penetration_m = 0.0

    sensor_history = deque()
    sensor_measurement_time_s = 0.0
    for step_index in range(total_steps):
        wall_time_s = min(float(data.time), duration_s)
        push_active = bool(
            abs(float(push_force_n)) > 0.0
            and float(push_start_s) <= wall_time_s
            < float(push_start_s + push_duration_s)
        )
        data.xfrc_applied[:] = 0.0
        if push_active:
            data.xfrc_applied[torso_body_id, 1] = float(push_force_n)

        if not disturbed and wall_time_s >= DISTURBANCE_TIME_S:
            data.qpos[pitch_qpos_address] += DISTURBANCE_PITCH_RAD
            data.qvel[pitch_dof_address] += DISTURBANCE_RATE_RAD_S
            mujoco.mj_forward(model, data)
            disturbed = True

        contact_values, _, leg_active, new_events = observer.update(
            model, data, wall_time_s
        )
        if terrain_contact_ids is not None:
            geometry_active = _geometry_leg_contacts(
                data, terrain_contact_ids
            )
            raw_fused = {
                side: bool(
                    leg_active[side] or geometry_active[side]
                )
                for side in ("left", "right")
            }
            new_events = []
            for side in ("left", "right"):
                if raw_fused[side] == fused_leg_active[side]:
                    fusion_pending_counts[side] = 0
                    continue
                fusion_pending_counts[side] += 1
                if (
                    fusion_pending_counts[side]
                    >= observer.debounce_steps
                ):
                    fused_leg_active[side] = raw_fused[side]
                    fusion_pending_counts[side] = 0
                    new_events.append({
                        "time_s": float(wall_time_s),
                        "kind": (
                            f"{side}_"
                            f"{'touchdown' if raw_fused[side] else 'liftoff'}"
                        ),
                    })
            leg_active = dict(fused_leg_active)
        observed_events.extend(dict(event) for event in new_events)
        sensor_history.append({
            "time_s": wall_time_s,
            "root_pitch_rad": float(
                proxy.torso_pitch_about_x(data, torso_body_id)
            ),
            "root_pitch_rate_rad_s": float(
                data.qvel[pitch_dof_address]
            ),
            "left_contact": bool(leg_active["left"]),
            "right_contact": bool(leg_active["right"]),
        })
        sensor_target_time_s = wall_time_s - float(sensor_delay_s)
        while (
            len(sensor_history) > 1
            and sensor_history[1]["time_s"] <= sensor_target_time_s
        ):
            sensor_history.popleft()
        sensor_sample = sensor_history[0]
        sensor_measurement_time_s = float(sensor_sample["time_s"])
        flight_active = not bool(leg_active["left"] or leg_active["right"])
        if flight_active:
            flight_streak_steps += 1
            maximum_flight_steps = max(
                maximum_flight_steps, flight_streak_steps
            )
        else:
            flight_streak_steps = 0
        if step_index % update_steps == 0:
            output = controller.update(
                controller_runtime.ControllerInput(
                    wall_time_s=wall_time_s,
                    root_pitch_rad=float(sensor_sample["root_pitch_rad"]),
                    root_pitch_rate_rad_s=float(
                        sensor_sample["root_pitch_rate_rad_s"]
                    ),
                    left_toe_contact=bool(sensor_sample["left_contact"]),
                    left_heel_contact=False,
                    right_toe_contact=bool(sensor_sample["right_contact"]),
                    right_heel_contact=False,
                ),
                mode=mode,
            )
            q_command = output.q_cmd.copy()
        if output is None:
            raise RuntimeError("controller produced no initial output")

        q_proxy = actuator.update(q_command)
        q_proxy = reference.q_ref[0] + float(actuator_gain) * (
            q_proxy - reference.q_ref[0]
        )
        data.ctrl[:] = 0.0
        for actuator_id, value in zip(mapping["actuator_ids"], q_proxy):
            data.ctrl[actuator_id] = float(value)
        mujoco.mj_step(model, data)
        obstacle_contact = {
            "detected": False,
            "contact_count": 0,
            "maximum_penetration_m": 0.0,
        }
        if terrain_contact_ids is not None:
            obstacle_contact = _obstacle_contact_snapshot(
                data, terrain_contact_ids
            )
            if bool(obstacle_contact["detected"]):
                step_obstacle_contact_sample_count += 1
                if first_step_obstacle_contact_time_s is None:
                    first_step_obstacle_contact_time_s = float(data.time)
                maximum_step_obstacle_penetration_m = max(
                    maximum_step_obstacle_penetration_m,
                    float(obstacle_contact["maximum_penetration_m"]),
                )

        measured_pitch_rad = float(
            proxy.torso_pitch_about_x(data, torso_body_id)
        )
        pitch_ref_rad = controller.bundle.pitch_ref_at(
            output.reference_time_s
        )
        rows.append(
            {
                "mode": mode,
                "time_s": float(data.time),
                "sensor_measurement_time_s": sensor_measurement_time_s,
                "reference_time_s": float(output.reference_time_s),
                "command_reference_time_s": float(
                    output.command_reference_time_s
                ),
                "pitch_ref_rad": pitch_ref_rad,
                "pitch_measured_rad": measured_pitch_rad,
                "applied_push_force_n": float(push_force_n if push_active else 0.0),
                "pitch_error_deg": float(
                    np.rad2deg(measured_pitch_rad - pitch_ref_rad)
                ),
                "alpha_rad": float(output.alpha_rad),
                "controller_state": str(output.controller_state),
                "phase_offset_s": float(output.phase_offset_s),
                "matched_events": int(output.matched_events),
                "skipped_events": int(output.skipped_events),
                "limiter_active": bool(output.limiter_active),
                "saturated": bool(output.saturated),
                "ncon": int(data.ncon),
                "step_obstacle_contact_detected": bool(
                    obstacle_contact["detected"]
                ),
                "step_obstacle_contact_count": int(
                    obstacle_contact["contact_count"]
                ),
                "step_obstacle_maximum_penetration_m": float(
                    obstacle_contact["maximum_penetration_m"]
                ),
                "torso_height_m": float(data.xpos[torso_body_id][2]),
                "left_contact_active": bool(leg_active["left"]),
                "right_contact_active": bool(leg_active["right"]),
                "flight_active": flight_active,
                "unintended_flight_duration_s": replay._duration_from_steps(
                    flight_streak_steps, dt_s
                ),
                "maximum_abs_delta_q_deg": float(
                    np.max(np.abs(np.rad2deg(output.delta_q)))
                ),
                **{
                    key: float(value)
                    for key, value in contact_values.items()
                },
            }
        )

    frame = pd.DataFrame(rows)
    event_matching = replay.match_contact_events(
        scheduled_events,
        observed_events,
        tolerance_s=float(phase_runtime.MATCH_WINDOW_S),
    )
    event_metrics = _event_metrics(
        mode, frame, scheduled_events, event_matching
    )
    timing_errors = np.asarray(
        [
            abs(float(row["timing_error_s"]))
            for row in event_matching["matched"]
        ],
        dtype=float,
    )
    timing_metrics = _contact_timing_metrics(mode, controller, timing_errors)
    time_values = frame["time_s"].to_numpy(dtype=float)
    error_values = frame["pitch_error_deg"].to_numpy(dtype=float)
    disturbance_mask = time_values >= DISTURBANCE_TIME_S
    if not np.any(disturbance_mask):
        disturbance_mask = np.ones_like(time_values, dtype=bool)
    metrics = {
        "push_force_n": float(push_force_n),
        "push_start_s": float(push_start_s),
        "push_duration_s": float(push_duration_s),
        "seed": int(seed),
        "initial_joint_noise_std_rad": float(initial_joint_noise_std_rad),
        "robustness_configuration": dict(robustness_configuration),
        "controller_configuration": dict(controller.controller_configuration),
        "peak_abs_error_deg": float(
            np.max(np.abs(error_values[disturbance_mask]))
        ),
        "iae_deg_s": float(
            np.trapezoid(
                np.abs(error_values[disturbance_mask]),
                time_values[disturbance_mask],
            )
        ),
        "max_abs_alpha_deg": float(
            np.max(np.abs(np.rad2deg(frame["alpha_rad"].to_numpy())))
        ),
        "max_abs_delta_q_deg": float(
            frame["maximum_abs_delta_q_deg"].max()
        ),
        "max_abs_phase_offset_ms": float(
            1000.0 * np.max(np.abs(frame["phase_offset_s"].to_numpy()))
        ),
        "maximum_unintended_flight_s": replay._duration_from_steps(
            maximum_flight_steps, dt_s
        ),
        "minimum_contacts": int(frame["ncon"].min()),
        "minimum_height_fraction": float(
            frame["torso_height_m"].min() / initial_height_m
        ),
        "step_obstacle_contact_detected": bool(
            step_obstacle_contact_sample_count > 0
        ),
        "step_obstacle_contact_sample_count": int(
            step_obstacle_contact_sample_count
        ),
        "first_step_obstacle_contact_time_s": (
            None
            if first_step_obstacle_contact_time_s is None
            else float(first_step_obstacle_contact_time_s)
        ),
        "maximum_step_obstacle_penetration_m": float(
            maximum_step_obstacle_penetration_m
        ),
        **event_metrics,
        **timing_metrics,
        "limiter_active_fraction": float(
            frame["limiter_active"].astype(float).mean()
        ),
        "recovery_active_fraction": float(
            (frame["controller_state"] == "recovery").astype(float).mean()
        ),
        "saturation_fraction": float(
            frame["saturated"].astype(float).mean()
        ),
    }
    return {
        "frame": frame,
        "metrics": metrics,
        "observed_events": observed_events,
        "event_matching": event_matching,
    }


def _make_figure(frames: Mapping[str, object], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for mode in MODES:
        frame = frames[mode]
        axes[0].plot(
            frame["time_s"], frame["pitch_error_deg"], label=mode
        )
        axes[1].plot(
            frame["time_s"],
            1000.0 * frame["phase_offset_s"],
            label=mode,
        )
    axes[0].axvline(
        DISTURBANCE_TIME_S, color="k", linestyle="--", linewidth=1.0
    )
    axes[0].set_ylabel("Pitch error (deg)")
    axes[1].set_ylabel("Phase offset (ms)")
    axes[1].set_xlabel("Time (s)")
    for axis in axes:
        axis.grid(True)
        axis.legend(ncol=4)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def run_ablation(
    *,
    smoke_steps: int | None = None,
    write_outputs: bool = True,
) -> dict:
    import pandas as pd

    reference = replay.load_multistep_reference(
        replay.FINAL_REFERENCE_PATH
    )
    duration_s = replay.resolve_replay_duration(
        reference, smoke_steps=smoke_steps
    )
    results = {
        mode: _run_mode(mode, duration_s) for mode in MODES
    }
    mode_metrics = {
        mode: results[mode]["metrics"] for mode in MODES
    }
    decision = evaluate_ablation(mode_metrics)
    frames = {mode: results[mode]["frame"] for mode in MODES}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "fddp_multistep_shared_controller_ablation",
        "claim_boundary": "simulation_only",
        "run_kind": "full" if smoke_steps is None else "smoke",
        "smoke_steps": smoke_steps,
        "reference_duration_s": duration_s,
        "modes": list(MODES),
        "disturbance": {
            "time_s": DISTURBANCE_TIME_S,
            "pitch_rad": DISTURBANCE_PITCH_RAD,
            "pitch_deg": float(np.rad2deg(DISTURBANCE_PITCH_RAD)),
            "pitch_rate_rad_s": DISTURBANCE_RATE_RAD_S,
        },
        "inputs": {
            "reference_npz": str(replay.FINAL_REFERENCE_PATH),
            "nominal_pitch_baseline_csv": str(
                controller_runtime.BASELINE_CSV_PATH
            ),
            "fir_yaml": str(controller_runtime.FIR_YAML_PATH),
            "ar_yaml": str(controller_runtime.AR_YAML_PATH),
            "tau_ref_usage": "analysis_only_never_sent_to_ctrl",
        },
        "mode_metrics": mode_metrics,
        "acceptance": decision,
        "integrity_passed": bool(decision["passed"]),
        "failures": list(decision["failures"]),
        "outputs": {
            "csv": str(CSV_PATH),
            "yaml": str(YAML_PATH),
            "figure": str(FIGURE_PATH),
        },
    }

    if write_outputs:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        pd.concat([frames[mode] for mode in MODES], ignore_index=True).to_csv(
            CSV_PATH, index=False
        )
        _make_figure(frames, FIGURE_PATH)
        YAML_PATH.write_text(
            yaml.safe_dump(summary, sort_keys=False),
            encoding="utf-8",
        )
    return {**summary, "frames": frames}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the shared ten-step C0/C1/phase-only/C2-final ablation."
    )
    parser.add_argument("--smoke-steps", type=int, default=None)
    args = parser.parse_args(argv)
    report = run_ablation(
        smoke_steps=args.smoke_steps,
        write_outputs=True,
    )
    printable = {key: value for key, value in report.items() if key != "frames"}
    print(yaml.safe_dump(printable, sort_keys=False))
    return 0 if report["integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
