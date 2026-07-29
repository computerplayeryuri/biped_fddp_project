#!/usr/bin/env python3
"""Strict loader and MuJoCo replay contract for the accepted ten-step FDDP gait.

The actuator boundary is position command only.  ``tau_ref`` is validated as
an analysis artifact but is deliberately not exposed by ``MultiStepReference``.
This module is simulation-only and does not establish hardware walking.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
FINAL_REFERENCE_PATH = (
    ROOT / "transfer" / "trajectories" / "fddp_multistep_10step_ref.npz"
)
CONFIG_PATH = ROOT / "config" / "fddp_multistep_experiment.yaml"
XML_PATH = ROOT / "model" / "biped_planar.xml"
LOG_DIR = ROOT / "results" / "logs"
FIGURE_DIR = ROOT / "results" / "figures"

SCHEMA_VERSION = "fddp-multistep-v1"
FINAL_ACCEPTANCE_STATE = "planning_and_contact_accepted"
TAU_REF_USAGE = "analysis_only"
CLAIM_BOUNDARY = "simulation_only"
FINAL_STEP_COUNT = 10
FINAL_SAMPLE_COUNT = 1001
FINAL_RUNNING_NODES = 1000
REFERENCE_DT_S = 0.010

MAXIMUM_UNINTENDED_FLIGHT_S = 0.020
MAXIMUM_SUSTAINED_PENETRATION_S = 0.020
PENETRATION_DEPTH_THRESHOLD_M = 0.003
EVENT_MATCH_TOLERANCE_S = 0.080
REPLAY_SCHEMA_VERSION = "fddp-multistep-replay-v1"
COMMAND_DELAY_S = 0.004
FIRST_ORDER_TIME_CONSTANT_S = 0.008
MAX_COMMAND_SPEED_RAD_PER_S = 20.0
COMMAND_REFERENCE_ADVANCE_S = 0.030
FALL_HEIGHT_FRACTION = 0.50
FALL_PITCH_DEG = 45.0


class ReferenceIntegrityError(RuntimeError):
    """Raised when an input is not the promoted ten-step nominal."""


@dataclass(frozen=True)
class MultiStepReference:
    path: Path
    schema_version: str
    acceptance_state: str
    step_count: int
    t_ref: np.ndarray
    q_ref: np.ndarray
    dq_ref: np.ndarray
    state_ref: np.ndarray
    com_ref_world: np.ndarray
    phase: np.ndarray
    left_contact_active: np.ndarray
    right_contact_active: np.ndarray
    cycle_id: np.ndarray
    tau_ref_usage: str
    claim_boundary: str

    @property
    def duration_s(self) -> float:
        return float(self.t_ref[-1])


def _scalar_text(values: np.ndarray) -> str:
    return str(np.asarray(values).reshape(()).item())


def _as_bool(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind in ("U", "S", "O"):
        return np.asarray(
            [
                str(value).strip().lower() in ("1", "true", "yes", "active")
                for value in values.reshape(-1)
            ],
            dtype=bool,
        )
    return values.astype(float).reshape(-1) > 0.5


def load_multistep_reference(
    path: Path = FINAL_REFERENCE_PATH,
) -> MultiStepReference:
    """Load only the independently promoted ten-step planning/contact artifact."""

    path = Path(path)
    if not path.is_file():
        raise ReferenceIntegrityError(f"missing multi-step reference: {path}")
    required = (
        "schema_version",
        "acceptance_state",
        "step_count",
        "t_ref",
        "q_ref",
        "dq_ref",
        "state_ref",
        "tau_ref",
        "com_ref_world",
        "phase",
        "left_contact_active",
        "right_contact_active",
        "cycle_id",
        "tau_ref_hardware_usage",
        "claim_boundary",
    )
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = [name for name in required if name not in archive.files]
            if missing:
                raise ReferenceIntegrityError(
                    f"{path} missing required multi-step keys: {missing}"
                )
            schema_version = _scalar_text(archive["schema_version"])
            acceptance_state = _scalar_text(archive["acceptance_state"])
            step_count = int(np.asarray(archive["step_count"]).reshape(()).item())
            tau_ref_usage = _scalar_text(archive["tau_ref_hardware_usage"])
            claim_boundary = _scalar_text(archive["claim_boundary"])
            t_ref = np.asarray(archive["t_ref"], dtype=float).reshape(-1).copy()
            q_ref = np.asarray(archive["q_ref"], dtype=float).copy()
            dq_ref = np.asarray(archive["dq_ref"], dtype=float).copy()
            state_ref = np.asarray(archive["state_ref"], dtype=float).copy()
            tau_analysis = np.asarray(archive["tau_ref"], dtype=float)
            com_ref_world = np.asarray(
                archive["com_ref_world"], dtype=float
            ).copy()
            phase = np.asarray(archive["phase"]).astype(str).reshape(-1).copy()
            left_contact_active = _as_bool(archive["left_contact_active"])
            right_contact_active = _as_bool(archive["right_contact_active"])
            cycle_id = np.asarray(archive["cycle_id"], dtype=np.int64).reshape(-1).copy()
    except ReferenceIntegrityError:
        raise
    except Exception as error:
        raise ReferenceIntegrityError(f"cannot load {path}: {error}") from error

    failures = []
    if schema_version != SCHEMA_VERSION:
        failures.append(f"schema={schema_version!r}")
    if acceptance_state != FINAL_ACCEPTANCE_STATE:
        failures.append(f"acceptance_state={acceptance_state!r}")
    if step_count != FINAL_STEP_COUNT:
        failures.append(f"step_count={step_count}")
    if tau_ref_usage != TAU_REF_USAGE:
        failures.append(f"tau_ref_hardware_usage={tau_ref_usage!r}")
    if claim_boundary != CLAIM_BOUNDARY:
        failures.append(f"claim_boundary={claim_boundary!r}")
    if t_ref.shape != (FINAL_SAMPLE_COUNT,):
        failures.append(f"t_ref_shape={t_ref.shape}")
    if q_ref.shape != (FINAL_SAMPLE_COUNT, 6):
        failures.append(f"q_ref_shape={q_ref.shape}")
    if dq_ref.shape != (FINAL_SAMPLE_COUNT, 6):
        failures.append(f"dq_ref_shape={dq_ref.shape}")
    if state_ref.shape != (FINAL_SAMPLE_COUNT, 25):
        failures.append(f"state_ref_shape={state_ref.shape}")
    if tau_analysis.shape != (FINAL_RUNNING_NODES, 6):
        failures.append(f"tau_ref_shape={tau_analysis.shape}")
    if com_ref_world.shape != (FINAL_SAMPLE_COUNT, 3):
        failures.append(f"com_ref_world_shape={com_ref_world.shape}")
    for name, values in (
        ("phase", phase),
        ("left_contact_active", left_contact_active),
        ("right_contact_active", right_contact_active),
        ("cycle_id", cycle_id),
    ):
        if values.shape != (FINAL_SAMPLE_COUNT,):
            failures.append(f"{name}_shape={values.shape}")
    numeric = (t_ref, q_ref, dq_ref, state_ref, tau_analysis, com_ref_world)
    if not all(np.all(np.isfinite(values)) for values in numeric):
        failures.append("non-finite numeric reference values")
    if t_ref.shape == (FINAL_SAMPLE_COUNT,):
        if not np.isclose(t_ref[0], 0.0, atol=1.0e-12):
            failures.append(f"initial_time={t_ref[0]}")
        if not np.allclose(np.diff(t_ref), REFERENCE_DT_S, atol=1.0e-12):
            failures.append("reference time is not a strict 10 ms grid")
        if not np.isclose(t_ref[-1], 10.0, atol=1.0e-12):
            failures.append(f"duration={t_ref[-1]}")
    if left_contact_active.shape == right_contact_active.shape == (
        FINAL_SAMPLE_COUNT,
    ) and np.any(~(left_contact_active | right_contact_active)):
        failures.append("reference contains a flight node")
    if cycle_id.shape == (FINAL_SAMPLE_COUNT,) and set(cycle_id.tolist()) != set(
        range(5)
    ):
        failures.append(f"cycle_ids={sorted(set(cycle_id.tolist()))}")
    if failures:
        raise ReferenceIntegrityError(
            f"{path} is not the promoted ten-step reference: {failures}"
        )
    return MultiStepReference(
        path=path,
        schema_version=schema_version,
        acceptance_state=acceptance_state,
        step_count=step_count,
        t_ref=t_ref,
        q_ref=q_ref,
        dq_ref=dq_ref,
        state_ref=state_ref,
        com_ref_world=com_ref_world,
        phase=phase,
        left_contact_active=left_contact_active,
        right_contact_active=right_contact_active,
        cycle_id=cycle_id,
        tau_ref_usage=tau_ref_usage,
        claim_boundary=claim_boundary,
    )


def build_reference_events(reference: MultiStepReference) -> list[dict]:
    """Extract the declared per-leg liftoff/touchdown sequence."""

    events = []
    for side in ("left", "right"):
        active = np.asarray(
            getattr(reference, f"{side}_contact_active"), dtype=bool
        )
        for index in range(1, active.size):
            if active[index] == active[index - 1]:
                continue
            events.append(
                {
                    "time_s": float(reference.t_ref[index]),
                    "kind": f"{side}_{'touchdown' if active[index] else 'liftoff'}",
                    "index": int(index),
                    "cycle_id": int(reference.cycle_id[index]),
                    "source": f"{side}_contact_active",
                }
            )
    return sorted(events, key=lambda event: (event["time_s"], event["kind"]))


def match_contact_events(
    scheduled: Sequence[Mapping[str, object]],
    observed: Sequence[Mapping[str, object]],
    *,
    tolerance_s: float,
) -> dict:
    """Greedily match contact events while preserving declared global order."""

    if not np.isfinite(tolerance_s) or tolerance_s < 0.0:
        raise ValueError("tolerance_s must be finite and non-negative")
    scheduled = [dict(event) for event in scheduled]
    observed = [dict(event) for event in observed]
    matched = []
    skipped = []
    extra = []
    expected_index = 0
    epsilon = 1.0e-12
    for observed_index, actual in enumerate(observed):
        actual_time = float(actual["time_s"])
        while expected_index < len(scheduled):
            expected_time = float(scheduled[expected_index]["time_s"])
            if actual_time <= expected_time + tolerance_s + epsilon:
                break
            skipped.append(
                {
                    "scheduled_index": expected_index,
                    "scheduled_event": scheduled[expected_index],
                }
            )
            expected_index += 1
        if expected_index >= len(scheduled):
            extra.append({"observed_index": observed_index, "observed_event": actual})
            continue
        expected = scheduled[expected_index]
        timing_error = actual_time - float(expected["time_s"])
        if (
            str(actual["kind"]) == str(expected["kind"])
            and abs(timing_error) <= tolerance_s + epsilon
        ):
            matched.append(
                {
                    "scheduled_index": expected_index,
                    "observed_index": observed_index,
                    "scheduled_event": expected,
                    "observed_event": actual,
                    "timing_error_s": float(timing_error),
                }
            )
            expected_index += 1
        else:
            extra.append({"observed_index": observed_index, "observed_event": actual})
    while expected_index < len(scheduled):
        skipped.append(
            {
                "scheduled_index": expected_index,
                "scheduled_event": scheduled[expected_index],
            }
        )
        expected_index += 1

    scheduled_indices = [row["scheduled_index"] for row in matched]
    observed_indices = [row["observed_index"] for row in matched]
    event_order_valid = bool(
        (not scheduled or matched)
        and scheduled_indices == sorted(set(scheduled_indices))
        and observed_indices == sorted(set(observed_indices))
    )
    required_kinds = {str(event["kind"]) for event in scheduled}
    matched_kinds = {str(row["scheduled_event"]["kind"]) for row in matched}
    return {
        "scheduled_count": len(scheduled),
        "observed_count": len(observed),
        "matched_count": len(matched),
        "skipped_count": len(skipped),
        "extra_observed_count": len(extra),
        "match_ratio": (
            1.0 if not scheduled else float(len(matched) / len(scheduled))
        ),
        "event_order_valid": event_order_valid,
        "all_event_kinds_present": required_kinds <= matched_kinds,
        "matched": matched,
        "skipped": skipped,
        "extra_observed": extra,
    }


def _replay_thresholds() -> dict[str, float]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return dict(config["replay"])


def evaluate_replay(metrics: Mapping[str, object]) -> dict:
    """Apply the approved nominal replay gates with inclusive boundaries."""

    required = {
        "finite",
        "fall_detected",
        "max_joint_rmse_deg",
        "max_abs_torso_pitch_deg",
        "minimum_height_fraction",
        "maximum_unintended_flight_s",
        "maximum_sustained_penetration_s",
        "event_match_ratio",
        "event_order_valid",
        "all_event_kinds_present",
    }
    missing = sorted(required - set(metrics))
    if missing:
        raise KeyError(f"missing replay metrics: {missing}")
    thresholds = _replay_thresholds()
    checks = {
        "finite": bool(metrics["finite"]),
        "fall_detected": not bool(metrics["fall_detected"]),
        "max_joint_rmse_deg": float(metrics["max_joint_rmse_deg"])
        <= float(thresholds["maximum_joint_rmse_deg"]),
        "max_abs_torso_pitch_deg": float(metrics["max_abs_torso_pitch_deg"])
        <= float(thresholds["maximum_abs_pitch_deg"]),
        "minimum_height_fraction": float(metrics["minimum_height_fraction"])
        >= float(thresholds["minimum_height_fraction"]),
        "maximum_unintended_flight_s": float(
            metrics["maximum_unintended_flight_s"]
        )
        <= MAXIMUM_UNINTENDED_FLIGHT_S,
        "maximum_sustained_penetration_s": float(
            metrics["maximum_sustained_penetration_s"]
        )
        <= MAXIMUM_SUSTAINED_PENETRATION_S,
        "event_match_ratio": float(metrics["event_match_ratio"])
        >= float(thresholds["minimum_event_match_ratio"]),
        "event_order_valid": bool(metrics["event_order_valid"]),
        "all_event_kinds_present": bool(metrics["all_event_kinds_present"]),
    }
    numeric_keys = (
        "max_joint_rmse_deg",
        "max_abs_torso_pitch_deg",
        "minimum_height_fraction",
        "maximum_unintended_flight_s",
        "maximum_sustained_penetration_s",
        "event_match_ratio",
    )
    for name in numeric_keys:
        if not np.isfinite(float(metrics[name])):
            checks[name] = False
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "thresholds": {
            **thresholds,
            "maximum_unintended_flight_s": MAXIMUM_UNINTENDED_FLIGHT_S,
            "penetration_depth_threshold_m": PENETRATION_DEPTH_THRESHOLD_M,
            "maximum_sustained_penetration_s": MAXIMUM_SUSTAINED_PENETRATION_S,
        },
        "diagnostics": dict(metrics),
    }

def resolve_replay_duration(
    reference: MultiStepReference,
    smoke_steps: int | None = None,
) -> float:
    """Return a complete-cycle replay duration without truncating a swing."""

    if smoke_steps is None:
        return float(reference.duration_s)
    if isinstance(smoke_steps, bool) or not isinstance(smoke_steps, int):
        raise ValueError("smoke_steps must be an even integer or None")
    if smoke_steps < 2 or smoke_steps > reference.step_count or smoke_steps % 2:
        raise ValueError(
            f"smoke_steps must be even and within [2, {reference.step_count}]"
        )
    duration = reference.duration_s * float(smoke_steps) / float(reference.step_count)
    if not np.isclose(duration / REFERENCE_DT_S, round(duration / REFERENCE_DT_S)):
        raise ReferenceIntegrityError("smoke duration does not end on the reference grid")
    return float(duration)


def replay_output_paths(smoke_steps: int | None = None) -> dict[str, Path]:
    """Keep smoke diagnostics separate from the accepted full-replay paths."""

    suffix = "" if smoke_steps is None else f"_smoke_{smoke_steps}step"
    return {
        "csv": LOG_DIR / f"fddp_multistep_nominal_replay{suffix}.csv",
        "yaml": LOG_DIR / f"fddp_multistep_nominal_replay{suffix}.yaml",
        "figure": FIGURE_DIR / f"fddp_multistep_nominal_replay{suffix}.png",
    }


def _load_existing_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load existing module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _scheduled_sample(reference: MultiStepReference, time_s: float) -> dict:
    index = int(np.searchsorted(reference.t_ref, time_s, side="right") - 1)
    index = int(np.clip(index, 0, reference.t_ref.size - 1))
    return {
        "index": index,
        "phase": str(reference.phase[index]),
        "cycle_id": int(reference.cycle_id[index]),
        "left": bool(reference.left_contact_active[index]),
        "right": bool(reference.right_contact_active[index]),
    }


def _minimum_contact_distance(data) -> float:
    if int(data.ncon) <= 0:
        return 0.0
    return float(min(data.contact[index].dist for index in range(int(data.ncon))))


def _duration_from_steps(step_count: int, dt_s: float) -> float:
    return float(round(step_count * dt_s, 12))


def _make_replay_figure(frame, scheduled_events, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(frame["time_s"], frame["torso_pitch_deg"], label="torso pitch")
    axes[0].axhline(15.0, color="tab:red", linestyle="--", linewidth=1.0)
    axes[0].axhline(-15.0, color="tab:red", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Pitch (deg)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(
        frame["time_s"],
        frame["maximum_abs_joint_error_deg"],
        label="max |joint tracking error|",
    )
    axes[1].plot(
        frame["time_s"],
        1000.0 * frame["com_y_change_m"],
        label="MuJoCo CoM +Y (mm)",
    )
    axes[1].plot(
        frame["time_s"],
        1000.0 * frame["com_ref_y_change_m"],
        linestyle="--",
        label="FDDP CoM +Y (mm)",
    )
    axes[1].set_ylabel("deg / mm")
    axes[1].grid(True)
    axes[1].legend(ncol=3)

    for name in ("left_toe", "left_heel", "right_toe", "right_heel"):
        axes[2].plot(
            frame["time_s"],
            frame[f"{name}_force_N"],
            label=name,
            linewidth=1.1,
        )
    axes[2].set_ylabel("Touch force (N)")
    axes[2].grid(True)
    axes[2].legend(ncol=4)

    axes[3].plot(frame["time_s"], frame["ncon"], label="global contacts")
    axes[3].plot(
        frame["time_s"],
        1000.0 * frame["penetration_depth_m"],
        label="penetration depth (mm)",
    )
    axes[3].plot(
        frame["time_s"],
        1000.0 * frame["unintended_flight_duration_s"],
        label="flight streak (ms)",
    )
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("count / mm / ms")
    axes[3].grid(True)
    axes[3].legend(ncol=3)

    for event in scheduled_events:
        for axis in axes:
            axis.axvline(
                float(event["time_s"]),
                color="0.65",
                linestyle=":",
                linewidth=0.7,
            )
    figure.suptitle(
        "Ten-step FDDP nominal: MuJoCo position-command replay diagnostics"
    )
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def run_nominal_replay(
    *,
    smoke_steps: int | None = None,
    write_outputs: bool = True,
    reference_path: Path = FINAL_REFERENCE_PATH,
) -> dict:
    """Replay the accepted FDDP q_ref through the existing position proxy."""

    import mujoco
    import pandas as pd

    reference = load_multistep_reference(reference_path)
    duration_s = resolve_replay_duration(reference, smoke_steps=smoke_steps)
    scheduled_events = [
        event
        for event in build_reference_events(reference)
        if float(event["time_s"]) <= duration_s + 1.0e-12
    ]
    expected_event_count = 2 * int(
        round(reference.step_count * duration_s / reference.duration_s)
    )
    if len(scheduled_events) != expected_event_count:
        raise ReferenceIntegrityError(
            f"replay window has {len(scheduled_events)} events; "
            f"expected {expected_event_count}"
        )

    proxy = _load_existing_module(
        ROOT / "simulation" / "03o_replay_fddp_com_shift_mujoco.py",
        "fddp_multistep_position_proxy",
    )
    contact = _load_existing_module(
        ROOT / "mpc" / "08_run_contact_phase_synchronised_fir_qp_mpc.py",
        "fddp_multistep_contact_observer",
    )
    proxy.COMMAND_DELAY_S = COMMAND_DELAY_S
    proxy.FIRST_ORDER_TIME_CONSTANT_S = FIRST_ORDER_TIME_CONSTANT_S
    proxy.MAX_COMMAND_SPEED_RAD_PER_S = MAX_COMMAND_SPEED_RAD_PER_S

    if not XML_PATH.is_file():
        raise FileNotFoundError(XML_PATH)
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    dt_s = float(model.opt.timestep)
    if dt_s <= 0.0:
        raise RuntimeError("MuJoCo timestep must be positive")

    mapping = proxy.resolve_joint_and_actuator_mapping(model)
    torso_body_id = proxy.resolve_torso_body(model)
    mujoco.mj_resetData(model, data)
    for address, value in zip(mapping["qpos_addresses"], reference.q_ref[0]):
        data.qpos[address] = float(value)
    for address in mapping["dof_addresses"]:
        data.qvel[address] = 0.0
    mujoco.mj_forward(model, data)

    initial_torso_height_m = float(data.xpos[torso_body_id][2])
    initial_com_y_m = float(proxy.total_mass_com(model, data)[1])
    if not math.isfinite(initial_torso_height_m) or initial_torso_height_m <= 0.0:
        raise RuntimeError(f"invalid initial torso height: {initial_torso_height_m}")

    observer = contact.ContactObserver(model, data, dt_s)
    actuator = proxy.PositionActuatorProxy(6, dt_s)
    actuator.reset(reference.q_ref[0])

    records: list[dict] = []
    observed_events: list[dict] = []
    finite = True
    flight_streak_steps = 0
    maximum_flight_steps = 0
    penetration_streak_steps = 0
    maximum_penetration_steps = 0
    total_steps = int(round(duration_s / dt_s))

    for _ in range(total_steps):
        command_reference_time_s = min(
            float(data.time) + COMMAND_REFERENCE_ADVANCE_S,
            duration_s,
        )
        q_command = proxy.interpolate_reference(
            command_reference_time_s, reference.t_ref, reference.q_ref
        )
        q_proxy = actuator.update(q_command)

        data.ctrl[:] = 0.0
        for actuator_id, value in zip(mapping["actuator_ids"], q_proxy):
            data.ctrl[actuator_id] = float(value)
        mujoco.mj_step(model, data)

        sample_time_s = float(data.time)
        reference_time_s = min(sample_time_s, duration_s)
        contact_values, site_active, leg_active, new_events = observer.update(
            model, data, sample_time_s
        )
        observed_events.extend(dict(event) for event in new_events)

        q_reference = proxy.interpolate_reference(
            reference_time_s, reference.t_ref, reference.q_ref
        )
        dq_reference = proxy.interpolate_reference(
            reference_time_s, reference.t_ref, reference.dq_ref
        )
        com_reference = proxy.interpolate_reference(
            reference_time_s, reference.t_ref, reference.com_ref_world
        )
        q_measured = np.asarray(
            [data.qpos[address] for address in mapping["qpos_addresses"]],
            dtype=float,
        )
        dq_measured = np.asarray(
            [data.qvel[address] for address in mapping["dof_addresses"]],
            dtype=float,
        )
        torso_height_m = float(data.xpos[torso_body_id][2])
        torso_pitch_rad = float(proxy.torso_pitch_about_x(data, torso_body_id))
        com_world = np.asarray(proxy.total_mass_com(model, data), dtype=float)
        scheduled = _scheduled_sample(reference, reference_time_s)

        minimum_contact_distance_m = _minimum_contact_distance(data)
        penetration_depth_m = max(0.0, -minimum_contact_distance_m)
        penetration_active = penetration_depth_m > PENETRATION_DEPTH_THRESHOLD_M
        if penetration_active:
            penetration_streak_steps += 1
            maximum_penetration_steps = max(
                maximum_penetration_steps, penetration_streak_steps
            )
        else:
            penetration_streak_steps = 0

        flight_active = not bool(leg_active["left"] or leg_active["right"])
        if flight_active:
            flight_streak_steps += 1
            maximum_flight_steps = max(maximum_flight_steps, flight_streak_steps)
        else:
            flight_streak_steps = 0

        contact_force_array = np.asarray(list(contact_values.values()), dtype=float)
        if not (
            np.all(np.isfinite(q_reference))
            and np.all(np.isfinite(dq_reference))
            and np.all(np.isfinite(q_proxy))
            and np.all(np.isfinite(q_measured))
            and np.all(np.isfinite(dq_measured))
            and np.all(np.isfinite(com_world))
            and np.all(np.isfinite(com_reference))
            and np.all(np.isfinite(contact_force_array))
            and math.isfinite(torso_height_m)
            and math.isfinite(torso_pitch_rad)
            and math.isfinite(penetration_depth_m)
        ):
            finite = False
            break

        joint_error_deg = np.rad2deg(q_measured - q_reference)
        row = {
            "time_s": sample_time_s,
            "reference_time_s": reference_time_s,
            "command_reference_time_s": command_reference_time_s,
            "phase_offset_s": 0.0,
            "phase": scheduled["phase"],
            "cycle_id": scheduled["cycle_id"],
            "torso_height_m": torso_height_m,
            "torso_pitch_rad": torso_pitch_rad,
            "torso_pitch_deg": float(np.rad2deg(torso_pitch_rad)),
            "com_y_m": float(com_world[1]),
            "com_y_change_m": float(com_world[1] - initial_com_y_m),
            "com_ref_y_m": float(com_reference[1]),
            "com_ref_y_change_m": float(
                com_reference[1] - reference.com_ref_world[0, 1]
            ),
            "ncon": int(data.ncon),
            "scheduled_left_contact": bool(scheduled["left"]),
            "scheduled_right_contact": bool(scheduled["right"]),
            "left_contact_observed": bool(leg_active["left"]),
            "right_contact_observed": bool(leg_active["right"]),
            "flight_active": flight_active,
            "unintended_flight_duration_s": _duration_from_steps(
                flight_streak_steps, dt_s
            ),
            "minimum_contact_distance_m": minimum_contact_distance_m,
            "penetration_depth_m": penetration_depth_m,
            "penetration_violation_active": penetration_active,
            "sustained_penetration_duration_s": _duration_from_steps(
                penetration_streak_steps, dt_s
            ),
            "observed_event_count": len(observed_events),
            "maximum_abs_joint_error_deg": float(np.max(np.abs(joint_error_deg))),
        }
        for key, value in contact_values.items():
            row[key] = float(value)
        for key, value in site_active.items():
            row[f"{key}_active"] = bool(value)
        for index, joint_name in enumerate(proxy.JOINT_ORDER):
            row[f"{joint_name}_ref_rad"] = float(q_reference[index])
            row[f"{joint_name}_proxy_cmd_rad"] = float(q_proxy[index])
            row[f"{joint_name}_meas_rad"] = float(q_measured[index])
            row[f"{joint_name}_ref_vel_rad_s"] = float(dq_reference[index])
            row[f"{joint_name}_meas_vel_rad_s"] = float(dq_measured[index])
        records.append(row)

    if not records:
        raise RuntimeError("multi-step replay produced no finite records")
    frame = pd.DataFrame(records)
    event_matching = match_contact_events(
        scheduled_events,
        observed_events,
        tolerance_s=EVENT_MATCH_TOLERANCE_S,
    )

    q_errors = np.column_stack(
        [
            frame[f"{joint_name}_meas_rad"].to_numpy()
            - frame[f"{joint_name}_ref_rad"].to_numpy()
            for joint_name in proxy.JOINT_ORDER
        ]
    )
    joint_rmse_deg = np.rad2deg(np.sqrt(np.mean(q_errors**2, axis=0)))
    minimum_torso_height_m = float(frame["torso_height_m"].min())
    maximum_abs_pitch_deg = float(frame["torso_pitch_deg"].abs().max())
    minimum_height_fraction = minimum_torso_height_m / initial_torso_height_m
    fall_detected = bool(
        minimum_height_fraction < FALL_HEIGHT_FRACTION
        or maximum_abs_pitch_deg > FALL_PITCH_DEG
    )
    metrics = {
        "finite": bool(finite and len(records) == total_steps),
        "fall_detected": fall_detected,
        "max_joint_rmse_deg": float(np.max(joint_rmse_deg)),
        "max_abs_torso_pitch_deg": maximum_abs_pitch_deg,
        "minimum_height_fraction": float(minimum_height_fraction),
        "maximum_unintended_flight_s": _duration_from_steps(
            maximum_flight_steps, dt_s
        ),
        "maximum_sustained_penetration_s": _duration_from_steps(
            maximum_penetration_steps, dt_s
        ),
        "event_match_ratio": float(event_matching["match_ratio"]),
        "event_order_valid": bool(event_matching["event_order_valid"]),
        "all_event_kinds_present": bool(
            event_matching["all_event_kinds_present"]
        ),
        "initial_torso_height_m": initial_torso_height_m,
        "minimum_torso_height_m": minimum_torso_height_m,
        "minimum_global_contact_count": int(frame["ncon"].min()),
        "maximum_penetration_depth_m": float(
            frame["penetration_depth_m"].max()
        ),
        "scheduled_contact_events": int(len(scheduled_events)),
        "observed_contact_events": int(len(observed_events)),
        "matched_contact_events": int(event_matching["matched_count"]),
        "skipped_contact_events": int(event_matching["skipped_count"]),
        "extra_observed_contact_events": int(
            event_matching["extra_observed_count"]
        ),
        "mujoco_final_com_y_shift_m": float(frame["com_y_change_m"].iloc[-1]),
        "fddp_final_com_y_shift_m": float(
            np.interp(
                duration_s,
                reference.t_ref,
                reference.com_ref_world[:, 1],
            )
            - reference.com_ref_world[0, 1]
        ),
    }
    acceptance = evaluate_replay(metrics)
    output_paths = replay_output_paths(smoke_steps=smoke_steps)
    summary = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "stage": "fddp_multistep_mujoco_nominal_position_replay",
        "purpose": (
            "MuJoCo position-command replay of the independently promoted "
            "multi-step FDDP nominal; not hardware walking."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "run_kind": "full" if smoke_steps is None else "smoke",
        "smoke_steps": smoke_steps,
        "reference_duration_s": duration_s,
        "control_source": "q_ref_position_proxy_only",
        "tau_ref_sent_to_ctrl": False,
        "tau_ref_usage": "analysis_only_never_sent_to_ctrl",
        "inputs": {
            "reference_npz": str(reference.path),
            "mujoco_xml": str(XML_PATH),
            "reference_schema_version": reference.schema_version,
            "reference_acceptance_state": reference.acceptance_state,
            "mujoco_timestep_s": dt_s,
        },
        "actuator_proxy": {
            "command_delay_s": COMMAND_DELAY_S,
            "first_order_time_constant_s": FIRST_ORDER_TIME_CONSTANT_S,
            "maximum_command_speed_rad_per_s": MAX_COMMAND_SPEED_RAD_PER_S,
            "command_reference_advance_s": COMMAND_REFERENCE_ADVANCE_S,
        },
        "contact_monitor": {
            "event_match_tolerance_s": EVENT_MATCH_TOLERANCE_S,
            "penetration_depth_threshold_m": PENETRATION_DEPTH_THRESHOLD_M,
            "maximum_unintended_flight_s": MAXIMUM_UNINTENDED_FLIGHT_S,
            "maximum_sustained_penetration_s": MAXIMUM_SUSTAINED_PENETRATION_S,
        },
        "metrics": metrics,
        "joint_rmse_deg": {
            name: float(value)
            for name, value in zip(proxy.JOINT_ORDER, joint_rmse_deg)
        },
        "scheduled_events": scheduled_events,
        "observed_events": observed_events,
        "event_matching": event_matching,
        "acceptance": acceptance,
        "integrity_passed": bool(acceptance["passed"]),
        "failures": list(acceptance["failures"]),
        "outputs": {name: str(path) for name, path in output_paths.items()},
    }

    if write_outputs:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_paths["csv"], index=False)
        _make_replay_figure(frame, scheduled_events, output_paths["figure"])
        output_paths["yaml"].write_text(
            yaml.safe_dump(summary, sort_keys=False),
            encoding="utf-8",
        )

    return {
        **summary,
        "frame": frame,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the promoted ten-step FDDP reference through MuJoCo "
            "position actuators."
        )
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=None,
        help="Replay an even complete-step prefix without overwriting full outputs.",
    )
    args = parser.parse_args(argv)
    result = run_nominal_replay(
        smoke_steps=args.smoke_steps,
        write_outputs=True,
    )
    printable = {key: value for key, value in result.items() if key != "frame"}
    print(yaml.safe_dump(printable, sort_keys=False))
    return 0 if result["integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
