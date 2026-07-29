#!/usr/bin/env python3
"""Thirty-seed multi-scenario robustness matrix for ten-step C2-final."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
STATS_PATH = ROOT / "mpc" / "robustness_statistics.py"
RUNNER_PATH = ROOT / "mpc" / "18_validate_fddp_multistep_c2_final.py"
RECOVERY_PATH = ROOT / "mpc" / "19_tune_fddp_multistep_push_recovery.py"
LOG_DIR = ROOT / "results" / "logs"
FIGURE_DIR = ROOT / "results" / "figures"
TRIAL_CSV_PATH = LOG_DIR / "fddp_multistep_robustness_trials.csv"
AGGREGATE_CSV_PATH = LOG_DIR / "fddp_multistep_robustness_aggregate.csv"
SUMMARY_YAML_PATH = LOG_DIR / "fddp_multistep_robustness_summary.yaml"
FIGURE_PATH = FIGURE_DIR / "fddp_multistep_robustness_matrix.png"
SCHEMA_VERSION = "fddp-multistep-robustness-v1"


def _load_existing(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load existing module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


statistics = _load_existing(STATS_PATH, "multistep_robustness_statistics")
runner = _load_existing(RUNNER_PATH, "multistep_robustness_runner")
recovery = _load_existing(RECOVERY_PATH, "multistep_robustness_recovery_gate")


def _scenario_kwargs(scenario) -> dict[str, Any]:
    return {
        "push_force_n": float(scenario.push_force_n),
        "friction_mu": float(scenario.friction_mu),
        "slope_deg": float(scenario.slope_deg),
        "step_height_m": float(scenario.step_height_m),
        "command_delay_s": float(scenario.command_delay_s),
        "sensor_delay_s": float(scenario.sensor_delay_s),
        "actuator_gain": float(scenario.actuator_gain),
        "actuator_time_constant_s": float(
            scenario.actuator_time_constant_s
        ),
        "initial_joint_noise_std_rad": float(
            scenario.initial_joint_noise_std_rad
        ),
        "initial_pitch_noise_std_rad": float(
            scenario.initial_pitch_noise_std_rad
        ),
    }


def _scenario_specific_failures(scenario, metrics) -> list[str]:
    if (
        float(scenario.step_height_m) > 0.0
        and not bool(metrics.get("step_obstacle_contact_detected", False))
    ):
        return ["step_obstacle_not_encountered"]
    return []


def run_trial(
    scenario,
    seed: int,
    mode: str = "C2_final",
    duration_s: float = 10.0,
) -> dict[str, Any]:
    try:
        result = runner._run_mode(
            mode,
            float(duration_s),
            seed=int(seed),
            **_scenario_kwargs(scenario),
        )
        metrics = result["metrics"]
        decision = recovery.evaluate_trial(metrics)
        failures = list(decision["failures"])
        failures.extend(_scenario_specific_failures(scenario, metrics))
        return {
            "scenario": scenario.label,
            "category": scenario.category,
            "critical": bool(scenario.critical),
            "mode": mode,
            "seed": int(seed),
            "passed": not failures,
            "failure_category": "" if not failures else failures[0],
            "failures": ";".join(failures),
            "peak_abs_error_deg": float(metrics["peak_abs_error_deg"]),
            "minimum_height_fraction": float(
                metrics["minimum_height_fraction"]
            ),
            "maximum_unintended_flight_s": float(
                metrics["maximum_unintended_flight_s"]
            ),
            "event_match_ratio": float(metrics["event_match_ratio"]),
            "event_order_valid": bool(metrics["event_order_valid"]),
            "max_abs_alpha_deg": float(metrics["max_abs_alpha_deg"]),
            "max_abs_delta_q_deg": float(metrics["max_abs_delta_q_deg"]),
            "max_abs_phase_offset_ms": float(
                metrics["max_abs_phase_offset_ms"]
            ),
            "recovery_active_fraction": float(
                metrics["recovery_active_fraction"]
            ),
            "saturation_fraction": float(metrics["saturation_fraction"]),
            "step_obstacle_contact_detected": bool(
                metrics["step_obstacle_contact_detected"]
            ),
            "step_obstacle_contact_sample_count": int(
                metrics["step_obstacle_contact_sample_count"]
            ),
            "first_step_obstacle_contact_time_s": metrics[
                "first_step_obstacle_contact_time_s"
            ],
            "maximum_step_obstacle_penetration_m": float(
                metrics["maximum_step_obstacle_penetration_m"]
            ),
            "runtime_exception": "",
        }
    except Exception as exc:
        return {
            "scenario": scenario.label,
            "category": scenario.category,
            "critical": bool(scenario.critical),
            "mode": mode,
            "seed": int(seed),
            "passed": False,
            "failure_category": "runtime_exception",
            "failures": f"{type(exc).__name__}: {exc}",
            "runtime_exception": f"{type(exc).__name__}: {exc}",
        }


def _write_figure(aggregates: Sequence[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["scenario"]) for row in aggregates]
    values = np.asarray(
        [float(row["success_rate"]) for row in aggregates], dtype=float
    )
    lower = values - np.asarray(
        [float(row["wilson_95_low"]) for row in aggregates], dtype=float
    )
    upper = np.asarray(
        [float(row["wilson_95_high"]) for row in aggregates], dtype=float
    ) - values
    figure, axis = plt.subplots(figsize=(16, 6))
    colors = [
        "#d95f02" if bool(row["critical"]) else "#1b9e77"
        for row in aggregates
    ]
    axis.bar(
        range(len(values)),
        values,
        yerr=np.vstack([lower, upper]),
        color=colors,
        capsize=2,
    )
    axis.axhline(0.90, color="black", linestyle="--", linewidth=1)
    axis.axhline(0.95, color="black", linestyle=":", linewidth=1)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("success rate")
    axis.set_xticks(range(len(labels)), labels, rotation=65, ha="right")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def run_matrix(
    scenarios: Sequence[Any] | None = None,
    seed_count: int = 30,
    duration_s: float = 10.0,
    write_outputs: bool = True,
) -> dict[str, Any]:
    scenarios = (
        statistics.build_scenarios()
        if scenarios is None else list(scenarios)
    )
    if not scenarios:
        raise ValueError("at least one scenario is required")
    seed_count = int(seed_count)
    if seed_count <= 0:
        raise ValueError("seed_count must be positive")
    trial_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        seeds = statistics.scenario_seeds(scenario, seed_count)
        for mode in scenario.modes:
            for seed in seeds:
                trial_rows.append(
                    run_trial(
                        scenario,
                        seed=seed,
                        mode=mode,
                        duration_s=duration_s,
                    )
                )
    aggregates = statistics.aggregate_trials(trial_rows)
    registered_labels = {
        item.label for item in statistics.build_scenarios()
    }
    complete_matrix = (
        {item.label for item in scenarios} == registered_labels
        and all(len(statistics.scenario_seeds(item, seed_count)) >= 30
                for item in scenarios)
    )
    acceptance = (
        statistics.evaluate_matrix(aggregates)
        if complete_matrix
        else {
            "passed": False,
            "checks": {"complete_30_seed_matrix": False},
            "failures": ["complete_30_seed_matrix"],
        }
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "stage": "fddp_multistep_statistical_robustness",
        "claim_boundary": "simulation_only_position_command",
        "seed_count": seed_count,
        "duration_s": float(duration_s),
        "scenario_count": len(scenarios),
        "trial_count": len(trial_rows),
        "terrain_obstacle_geometry": (
            runner.terrain_runtime.step_obstacle_geometry()
        ),
        "scenarios": [
            {
                **asdict(item),
                "modes": list(item.modes),
            }
            for item in scenarios
        ],
        "trial_rows": trial_rows,
        "aggregates": aggregates,
        "acceptance": acceptance,
        "outputs": {
            "trials_csv": str(TRIAL_CSV_PATH),
            "aggregate_csv": str(AGGREGATE_CSV_PATH),
            "summary_yaml": str(SUMMARY_YAML_PATH),
            "figure": str(FIGURE_PATH),
        },
    }
    if write_outputs:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(trial_rows).to_csv(TRIAL_CSV_PATH, index=False)
        pd.DataFrame(aggregates).to_csv(AGGREGATE_CSV_PATH, index=False)
        SUMMARY_YAML_PATH.write_text(
            yaml.safe_dump(report, sort_keys=False), encoding="utf-8"
        )
        _write_figure(aggregates)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-count", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seed_count = (
        int(args.seed_count)
        if args.seed_count is not None
        else (3 if args.smoke else 30)
    )
    report = run_matrix(seed_count=seed_count, write_outputs=True)
    print(
        f"scenarios={report['scenario_count']} "
        f"trials={report['trial_count']} "
        f"accepted={report['acceptance']['passed']}"
    )
    if not args.smoke and not report["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
