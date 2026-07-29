"""Pure contracts for the gated multi-step FDDP simulation pipeline.

This module deliberately has no Pinocchio, Crocoddyl, or MuJoCo dependency so
that schedules, schemas, and acceptance boundaries can be tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "fddp_multistep_experiment.yaml"

SCHEMA_VERSION = "fddp-multistep-v1"
CONFIG_SCHEMA_VERSION = "fddp-multistep-config-v1"
DT = 0.010
DS_NODES = 20
SS_NODES = 70
STEP_FORWARD_Y_M = 0.010
SWING_CLEARANCE_Z_M = 0.020

DEFAULT_THRESHOLDS = {
    "solver_stop": 1.0e-7,
    "shooting_gap_norm": 1.0e-7,
    "active_support_translation_error_m": 5.0e-4,
    "active_support_rotation_error_rad": np.deg2rad(0.5),
    "landing_translation_error_m": 1.0e-3,
    "landing_rotation_error_rad": np.deg2rad(1.0),
    "swing_clearance_m": 1.0e-2,
    "joint_limit_margin_rad": 1.0e-2,
    "forward_displacement_m": 4.0e-2,
    "duration_s": 10.0,
}

_MAXIMUM_GATES = {
    "solver_stop",
    "shooting_gap_norm",
    "active_support_translation_error_m",
    "active_support_rotation_error_rad",
    "landing_translation_error_m",
    "landing_rotation_error_rad",
}


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    mode: str
    node_count: int
    swing_side: str | None
    cycle_id: int

    @property
    def left_contact_active(self) -> bool:
        return self.mode != "RIGHT_SS"

    @property
    def right_contact_active(self) -> bool:
        return self.mode != "LEFT_SS"


def _validate_step_count(step_count: int, require_final: bool) -> None:
    if isinstance(step_count, bool) or not isinstance(step_count, int):
        raise TypeError("step_count must be an integer")
    if step_count <= 0 or step_count % 2:
        raise ValueError("step_count must be a positive even integer")
    if require_final and step_count < 10:
        raise ValueError("the final multi-step run requires at least ten steps")


def build_step_schedule(
    step_count: int,
    *,
    require_final: bool = False,
) -> list[PhaseSpec]:
    """Build complete 200-node left/right periodic cycles.

    Consecutive DS phases at cycle boundaries are intentionally not merged.
    Five cycles therefore contain 10 steps, 1000 nodes, and 10 seconds.
    """

    _validate_step_count(step_count, require_final)
    schedule: list[PhaseSpec] = []
    template = (
        ("double_support_start", "DS", DS_NODES, None),
        ("right_stance_left_swing", "RIGHT_SS", SS_NODES, "left"),
        ("double_support_mid", "DS", DS_NODES, None),
        ("left_stance_right_swing", "LEFT_SS", SS_NODES, "right"),
        ("double_support_landing", "DS", DS_NODES, None),
    )
    for cycle_id in range(step_count // 2):
        for suffix, mode, node_count, swing_side in template:
            schedule.append(
                PhaseSpec(
                    name=f"cycle_{cycle_id:02d}_{suffix}",
                    mode=mode,
                    node_count=node_count,
                    swing_side=swing_side,
                    cycle_id=cycle_id,
                )
            )
    return schedule


def phase_arrays(schedule: list[PhaseSpec]) -> dict[str, np.ndarray]:
    if not schedule:
        raise ValueError("schedule must not be empty")

    rows = {
        "phase_name": [],
        "mode": [],
        "swing_side": [],
        "cycle_id": [],
        "left_contact_active": [],
        "right_contact_active": [],
    }
    for phase in schedule:
        if phase.node_count <= 0:
            raise ValueError("every phase must contain at least one node")
        rows["phase_name"].extend([phase.name] * phase.node_count)
        rows["mode"].extend([phase.mode] * phase.node_count)
        rows["swing_side"].extend([(phase.swing_side or "none")] * phase.node_count)
        rows["cycle_id"].extend([phase.cycle_id] * phase.node_count)
        rows["left_contact_active"].extend(
            [phase.left_contact_active] * phase.node_count
        )
        rows["right_contact_active"].extend(
            [phase.right_contact_active] * phase.node_count
        )

    return {
        "phase_name": np.asarray(rows["phase_name"], dtype=str),
        "mode": np.asarray(rows["mode"], dtype=str),
        "swing_side": np.asarray(rows["swing_side"], dtype=str),
        "cycle_id": np.asarray(rows["cycle_id"], dtype=int),
        "left_contact_active": np.asarray(
            rows["left_contact_active"], dtype=bool
        ),
        "right_contact_active": np.asarray(
            rows["right_contact_active"], dtype=bool
        ),
    }


def planned_forward_displacement(schedule: list[PhaseSpec]) -> float:
    if not schedule:
        raise ValueError("schedule must not be empty")
    cycle_ids = {phase.cycle_id for phase in schedule}
    if cycle_ids != set(range(len(cycle_ids))):
        raise ValueError("cycle identifiers must be contiguous and zero based")
    return len(cycle_ids) * STEP_FORWARD_Y_M


def export_metadata(step_count: int, horizon_nodes: int) -> dict:
    _validate_step_count(step_count, require_final=False)
    if isinstance(horizon_nodes, bool) or not isinstance(horizon_nodes, int):
        raise TypeError("horizon_nodes must be an integer")
    if horizon_nodes <= 0:
        raise ValueError("horizon_nodes must be positive")
    return {
        "schema_version": SCHEMA_VERSION,
        "step_count": step_count,
        "horizon_nodes": horizon_nodes,
        "dt_s": DT,
        "duration_s": horizon_nodes * DT,
        "command_interface": "q_cmd=q_ref+delta_q",
        "tau_ref_hardware_usage": "analysis_only",
        "claim_boundary": "simulation_only",
    }


def validate_thresholds(thresholds: Mapping[str, float]) -> None:
    required = set(DEFAULT_THRESHOLDS)
    supplied = set(thresholds)
    if supplied != required:
        missing = sorted(required - supplied)
        unknown = sorted(supplied - required)
        raise KeyError(f"threshold schema mismatch: missing={missing}, unknown={unknown}")
    for name, value in thresholds.items():
        numeric = float(value)
        if not np.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"threshold {name!r} must be finite and positive")


def evaluate_gate(
    metrics: Mapping[str, float],
    thresholds: Mapping[str, float],
) -> dict:
    validate_thresholds(thresholds)
    missing = sorted(set(thresholds) - set(metrics))
    if missing:
        raise KeyError(f"missing gate metrics: {missing}")

    checks: dict[str, bool] = {}
    diagnostics: dict[str, float] = {}
    for name, threshold in thresholds.items():
        value = float(metrics[name])
        if not np.isfinite(value):
            raise ValueError(f"metric {name!r} must be finite")
        diagnostics[name] = value
        if name in _MAXIMUM_GATES:
            checks[name] = value <= float(threshold)
        else:
            checks[name] = value >= float(threshold)

    failures = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "diagnostics": diagnostics,
    }


def load_experiment_config(path: Path = CONFIG_PATH) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("experiment config must contain a mapping")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported multi-step config schema")

    planning = config.get("planning")
    if not isinstance(planning, dict):
        raise ValueError("experiment config is missing planning settings")
    thresholds = planning.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("experiment config is missing planning thresholds")
    thresholds = {name: float(value) for name, value in thresholds.items()}
    validate_thresholds(thresholds)
    planning["thresholds"] = thresholds

    final_steps = planning.get("final_step_count")
    schedule = build_step_schedule(final_steps, require_final=True)
    expected_nodes = sum(phase.node_count for phase in schedule)
    if planning.get("final_horizon_nodes") != expected_nodes:
        raise ValueError("final_horizon_nodes disagrees with the periodic schedule")
    return config

