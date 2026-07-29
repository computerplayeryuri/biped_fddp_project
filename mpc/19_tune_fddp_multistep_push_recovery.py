#!/usr/bin/env python3
"""Bounded staged recovery search for the accepted ten-step FDDP replay.

The executable stages reuse the existing multi-step FIR-QP controller,
contact-phase synchroniser, safety limiter, and MuJoCo ablation runner.  A
foothold stage is declared but deliberately blocked until an adapter can
regenerate reference/contact-feasibility evidence; metadata-only offsets are
not treated as recovery control.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "mpc" / "18_validate_fddp_multistep_c2_final.py"
CONFIG_PATH = ROOT / "config" / "fddp_multistep_experiment.yaml"
LOG_DIR = ROOT / "results" / "logs"
FIGURE_DIR = ROOT / "results" / "figures"
TRIAL_CSV_PATH = LOG_DIR / "fddp_multistep_recovery_sweep.csv"
SELECTION_YAML_PATH = LOG_DIR / "fddp_multistep_recovery_selection.yaml"
FIGURE_PATH = FIGURE_DIR / "fddp_multistep_recovery_sweep.png"


@dataclass(frozen=True)
class RecoveryLimits:
    max_alpha_rad: float = math.radians(5.0)
    max_foothold_y_m: float = 0.015
    maximum_unintended_flight_s: float = 0.020
    minimum_height_fraction: float = 0.65
    maximum_pitch_error_deg: float = 15.0
    minimum_event_match_ratio: float = 0.90
    maximum_saturation_fraction: float = 0.0
    minimum_success_rate: float = 0.90


@dataclass(frozen=True)
class RecoveryTerms:
    alpha_rad: float
    foothold_y_m: float


@dataclass(frozen=True)
class RecoveryCandidate:
    name: str
    stage: str
    fir_qp_candidate: dict[str, float] = field(default_factory=dict)
    phase_sync: dict[str, float] = field(default_factory=dict)
    foothold_y_m: float = 0.0
    executable: bool = True
    blocked_reason: str | None = None


def bound_recovery(
    raw_alpha_rad: float,
    raw_foothold_y_m: float,
    limits: RecoveryLimits | None = None,
) -> RecoveryTerms:
    """Clip recovery terms to the approved position-command authority."""
    limits = RecoveryLimits() if limits is None else limits
    return RecoveryTerms(
        alpha_rad=float(np.clip(
            raw_alpha_rad, -limits.max_alpha_rad, limits.max_alpha_rad
        )),
        foothold_y_m=float(np.clip(
            raw_foothold_y_m,
            -limits.max_foothold_y_m,
            limits.max_foothold_y_m,
        )),
    )


def recovery_terms_respect_limits(
    terms: RecoveryTerms,
    limits: RecoveryLimits | None = None,
) -> bool:
    limits = RecoveryLimits() if limits is None else limits
    values = (terms.alpha_rad, terms.foothold_y_m)
    return bool(
        np.all(np.isfinite(values))
        and abs(terms.alpha_rad) <= limits.max_alpha_rad + 1.0e-12
        and abs(terms.foothold_y_m) <= limits.max_foothold_y_m + 1.0e-12
    )


def build_recovery_candidates() -> list[RecoveryCandidate]:
    """Return staged candidates in the approved least-authority-first order."""
    fast_fir = {
        "name": "recovery_fast",
        "q": 2500.0,
        "r": 5.0,
        "slew": 10.0,
    }
    fast_phase = {
        "max_offset_s": 0.100,
        "max_step_s": 0.050,
        "correction_gain": 1.0,
    }
    return [
        RecoveryCandidate(
            name="fir_fast",
            stage="fir_qp_retune",
            fir_qp_candidate=dict(fast_fir),
        ),
        RecoveryCandidate(
            name="fir_fast_phase_1p0",
            stage="phase_gain_scheduling",
            fir_qp_candidate=dict(fast_fir),
            phase_sync=dict(fast_phase),
        ),
        RecoveryCandidate(
            name="bounded_foothold_requires_replan",
            stage="bounded_foothold",
            fir_qp_candidate=dict(fast_fir),
            phase_sync=dict(fast_phase),
            foothold_y_m=0.015,
            executable=False,
            blocked_reason=(
                "No existing online foothold adapter regenerates both the "
                "FDDP reference metadata and contact-feasibility report."
            ),
        ),
    ]


def select_first_accepted(
    rows: Sequence[dict[str, Any]],
    minimum_rate: float = 0.90,
) -> dict[str, Any] | None:
    """Select the first candidate meeting the two-direction success gate."""
    for row in rows:
        forward = row.get("forward_success_rate")
        backward = row.get("backward_success_rate")
        if forward is None or backward is None:
            continue
        if float(forward) >= minimum_rate and float(backward) >= minimum_rate:
            return row
    return None


def evaluate_trial(
    metrics: dict[str, Any],
    limits: RecoveryLimits | None = None,
) -> dict[str, Any]:
    """Apply fixed stability, contact, event, and command-safety gates."""
    limits = RecoveryLimits() if limits is None else limits
    failures: list[str] = []
    if float(metrics["maximum_unintended_flight_s"]) > (
        limits.maximum_unintended_flight_s + 1.0e-12
    ):
        failures.append("unintended_flight_above_limit")
    if float(metrics["minimum_height_fraction"]) < limits.minimum_height_fraction:
        failures.append("torso_height_below_limit")
    if float(metrics["peak_abs_error_deg"]) > limits.maximum_pitch_error_deg:
        failures.append("pitch_error_above_limit")
    if float(metrics["event_match_ratio"]) < limits.minimum_event_match_ratio:
        failures.append("contact_event_match_below_limit")
    if not bool(metrics["event_order_valid"]):
        failures.append("contact_event_order_invalid")
    if float(metrics["max_abs_alpha_deg"]) > math.degrees(limits.max_alpha_rad) + 1.0e-12:
        failures.append("alpha_above_limit")
    if float(metrics["saturation_fraction"]) > limits.maximum_saturation_fraction + 1.0e-12:
        failures.append("joint_saturation_detected")
    return {"passed": not failures, "failures": failures}


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "fddp_multistep_recovery_runner", RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load existing runner: {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _controller_kwargs(candidate: RecoveryCandidate) -> dict[str, object]:
    controller_config = yaml.safe_load(
        CONFIG_PATH.read_text(encoding="utf-8")
    )["controller"]
    kwargs: dict[str, object] = {
        "recovery_fir_qp_candidate": dict(candidate.fir_qp_candidate),
        "recovery_activation_pitch_deg": float(
            controller_config["recovery_activation_pitch_deg"]
        ),
    }
    if candidate.phase_sync:
        kwargs["phase_sync"] = dict(candidate.phase_sync)
    return kwargs


def _run_candidate(
    candidate: RecoveryCandidate,
    seeds: Sequence[int],
    limits: RecoveryLimits,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runner = _load_runner()
    rows: list[dict[str, Any]] = []
    direction_for_force = {-4.0: "backward", 4.0: "forward"}
    for force_n, direction in direction_for_force.items():
        for seed in seeds:
            result = runner._run_mode(
                "C2_final",
                10.0,
                push_force_n=force_n,
                push_start_s=0.80,
                push_duration_s=0.12,
                seed=int(seed),
                initial_joint_noise_std_rad=0.001,
                controller_kwargs=_controller_kwargs(candidate),
            )
            metrics = result["metrics"]
            decision = evaluate_trial(metrics, limits)
            rows.append({
                "candidate": candidate.name,
                "stage": candidate.stage,
                "status": "executed",
                "direction": direction,
                "push_force_n": force_n,
                "seed": int(seed),
                "passed": bool(decision["passed"]),
                "failures": ";".join(decision["failures"]),
                "minimum_contacts": int(metrics["minimum_contacts"]),
                "maximum_unintended_flight_s": float(
                    metrics["maximum_unintended_flight_s"]
                ),
                "minimum_height_fraction": float(metrics["minimum_height_fraction"]),
                "peak_abs_error_deg": float(metrics["peak_abs_error_deg"]),
                "event_match_ratio": float(metrics["event_match_ratio"]),
                "event_order_valid": bool(metrics["event_order_valid"]),
                "max_abs_alpha_deg": float(metrics["max_abs_alpha_deg"]),
                "max_abs_delta_q_deg": float(metrics["max_abs_delta_q_deg"]),
                "max_abs_phase_offset_ms": float(metrics["max_abs_phase_offset_ms"]),
                "recovery_active_fraction": float(
                    metrics["recovery_active_fraction"]
                ),
                "saturation_fraction": float(metrics["saturation_fraction"]),
                "foothold_y_m": float(candidate.foothold_y_m),
            })
    summary: dict[str, Any] = {
        "candidate": candidate.name,
        "stage": candidate.stage,
        "status": "executed",
    }
    for direction in ("forward", "backward"):
        selected = [row for row in rows if row["direction"] == direction]
        summary[f"{direction}_success_count"] = sum(
            bool(row["passed"]) for row in selected
        )
        summary[f"{direction}_trial_count"] = len(selected)
        summary[f"{direction}_success_rate"] = (
            summary[f"{direction}_success_count"] / max(len(selected), 1)
        )
    summary["accepted"] = bool(
        summary["forward_success_rate"] >= limits.minimum_success_rate
        and summary["backward_success_rate"] >= limits.minimum_success_rate
    )
    return summary, rows


def _write_figure(candidate_rows: Sequence[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    executed = [row for row in candidate_rows if row.get("status") == "executed"]
    labels: list[str] = []
    values: list[float] = []
    for row in executed:
        for direction in ("forward", "backward"):
            labels.append(f"{row['candidate']}\n{direction}")
            values.append(float(row[f"{direction}_success_rate"]))
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.bar(range(len(values)), values)
    axis.axhline(0.90, color="black", linestyle="--", linewidth=1.0)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("success rate")
    axis.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    axis.grid(axis="y")
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def run_recovery_search(
    seeds: Sequence[int],
    write_outputs: bool = True,
) -> dict[str, Any]:
    import pandas as pd
    import yaml

    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("at least one seed is required")
    limits = RecoveryLimits()
    candidate_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    blocked_reason: str | None = None
    candidates = build_recovery_candidates()
    for index, candidate in enumerate(candidates):
        if not candidate.executable:
            blocked_reason = candidate.blocked_reason
            candidate_rows.append({
                "candidate": candidate.name,
                "stage": candidate.stage,
                "status": "blocked",
                "forward_success_rate": None,
                "backward_success_rate": None,
                "accepted": False,
                "blocked_reason": blocked_reason,
            })
            trial_rows.append({
                "candidate": candidate.name,
                "stage": candidate.stage,
                "status": "blocked",
                "direction": "not_run",
                "seed": -1,
                "passed": False,
                "failures": blocked_reason,
                "foothold_y_m": candidate.foothold_y_m,
            })
            break
        summary, rows = _run_candidate(candidate, seeds, limits)
        candidate_rows.append(summary)
        trial_rows.extend(rows)
        selected = select_first_accepted(
            candidate_rows, minimum_rate=limits.minimum_success_rate
        )
        if selected is not None:
            for later in candidates[index + 1:]:
                candidate_rows.append({
                    "candidate": later.name,
                    "stage": later.stage,
                    "status": "not_run_earlier_stage_accepted",
                    "forward_success_rate": None,
                    "backward_success_rate": None,
                    "accepted": False,
                })
            break
    report = {
        "stage": "fddp_multistep_bounded_push_recovery",
        "claim_boundary": "simulation_only_position_command",
        "seeds": list(seeds),
        "push": {"forces_n": [-4.0, 4.0], "start_s": 0.80, "duration_s": 0.12},
        "limits": asdict(limits),
        "candidate_results": candidate_rows,
        "selected_candidate": None if selected is None else selected["candidate"],
        "accepted": selected is not None,
        "blocked_reason": blocked_reason if selected is None else None,
        "foothold_claim": (
            "not_executed_without_reference_and_contact_feasibility_regeneration"
        ),
        "outputs": {
            "trials_csv": str(TRIAL_CSV_PATH),
            "selection_yaml": str(SELECTION_YAML_PATH),
            "figure": str(FIGURE_PATH),
        },
    }
    if write_outputs:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(trial_rows).to_csv(TRIAL_CSV_PATH, index=False)
        SELECTION_YAML_PATH.write_text(
            yaml.safe_dump(report, sort_keys=False), encoding="utf-8"
        )
        _write_figure(candidate_rows)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seeds", nargs="+", type=int)
    group.add_argument("--seed-count", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.seeds is not None:
        seeds = args.seeds
    else:
        if args.seed_count <= 0:
            raise ValueError("--seed-count must be positive")
        seeds = list(range(args.seed_count))
    report = run_recovery_search(seeds)
    print(f"selected_candidate={report['selected_candidate']}")
    print(f"accepted={report['accepted']}")
    if report["blocked_reason"]:
        print(f"blocked_reason={report['blocked_reason']}")
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
