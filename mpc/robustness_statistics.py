#!/usr/bin/env python3
"""Pure scenario registry and statistics for ten-step robustness evidence."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class Scenario:
    label: str
    category: str
    critical: bool = False
    seed_count: int = 30
    modes: tuple[str, ...] = ("C2_final",)
    friction_mu: float = 0.80
    slope_deg: float = 0.0
    step_height_m: float = 0.0
    push_force_n: float = 0.0
    command_delay_s: float = 0.020
    sensor_delay_s: float = 0.0
    actuator_gain: float = 1.0
    actuator_time_constant_s: float = 0.030
    initial_joint_noise_std_rad: float = 0.0
    initial_pitch_noise_std_rad: float = 0.0


def build_scenarios() -> list[Scenario]:
    scenarios = [Scenario("nominal", "nominal", critical=True)]
    scenarios.extend([
        Scenario("push_minus_2N", "push", push_force_n=-2.0),
        Scenario("push_plus_2N", "push", push_force_n=2.0),
        Scenario("push_minus_4N", "push", critical=True, push_force_n=-4.0),
        Scenario("push_plus_4N", "push", critical=True, push_force_n=4.0),
    ])
    scenarios.extend([
        Scenario(
            f"friction_mu_{mu:.2f}".replace(".", "p"),
            "friction",
            critical=mu == 0.40,
            friction_mu=mu,
        )
        for mu in (0.40, 0.50, 0.80)
    ])
    scenarios.extend([
        Scenario(
            f"slope_{'plus' if value > 0 else 'minus'}_{abs(value):g}deg",
            "slope",
            critical=abs(value) == 5.0,
            slope_deg=value,
        )
        for value in (-5.0, -3.0, 3.0, 5.0)
    ])
    scenarios.extend([
        Scenario(
            f"step_{millimetres}mm",
            "step",
            critical=millimetres == 6,
            step_height_m=millimetres / 1000.0,
        )
        for millimetres in (2, 4, 6)
    ])
    scenarios.extend([
        Scenario(
            f"command_delay_{milliseconds}ms",
            "command_delay",
            critical=milliseconds == 20,
            command_delay_s=milliseconds / 1000.0,
        )
        for milliseconds in (4, 12, 20)
    ])
    scenarios.extend([
        Scenario(
            f"sensor_delay_{milliseconds}ms",
            "sensor_delay",
            critical=milliseconds == 40,
            sensor_delay_s=milliseconds / 1000.0,
        )
        for milliseconds in (0, 20, 40)
    ])
    scenarios.extend([
        Scenario(
            "actuator_gain_minus_20pct",
            "actuator_gain",
            critical=True,
            actuator_gain=0.80,
        ),
        Scenario(
            "actuator_gain_plus_20pct",
            "actuator_gain",
            critical=True,
            actuator_gain=1.20,
        ),
    ])
    scenarios.extend([
        Scenario(
            f"actuator_tau_{milliseconds}ms",
            "actuator_time_constant",
            critical=milliseconds in (20, 50),
            actuator_time_constant_s=milliseconds / 1000.0,
        )
        for milliseconds in (20, 30, 40, 50)
    ])
    scenarios.extend([
        Scenario(
            "joint_noise_0p002rad",
            "joint_noise",
            critical=True,
            initial_joint_noise_std_rad=0.002,
        ),
        Scenario(
            "pitch_noise_0p5deg",
            "pitch_noise",
            critical=True,
            initial_pitch_noise_std_rad=math.radians(0.5),
        ),
        Scenario(
            "combined_A",
            "combined",
            critical=True,
            friction_mu=0.50,
            slope_deg=3.0,
            command_delay_s=0.012,
        ),
        Scenario(
            "combined_B",
            "combined",
            critical=True,
            step_height_m=0.004,
            sensor_delay_s=0.020,
            push_force_n=2.0,
        ),
        Scenario(
            "combined_C",
            "combined",
            critical=True,
            friction_mu=0.40,
            slope_deg=-3.0,
            push_force_n=4.0,
        ),
    ])
    labels = [item.label for item in scenarios]
    if len(labels) != len(set(labels)):
        raise RuntimeError("robustness scenario labels must be unique")
    return scenarios


def scenario_seeds(scenario: Scenario, count: int | None = None) -> tuple[int, ...]:
    count = scenario.seed_count if count is None else int(count)
    if count <= 0:
        raise ValueError("seed count must be positive")
    base = int.from_bytes(
        hashlib.sha256(scenario.label.encode("utf-8")).digest()[:4], "big"
    ) % 1_000_000
    return tuple(base + index for index in range(count))


def wilson_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    successes = int(successes)
    trials = int(trials)
    confidence = float(confidence)
    if trials <= 0:
        raise ValueError("trials must be positive")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be within [0, trials]")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be within (0, 1)")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def aggregate_trials(
    rows: Sequence[dict[str, Any]],
    confidence: float = 0.95,
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("trial rows must not be empty")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if not bool(row["passed"]) and not str(
            row.get("failure_category", "")
        ).strip():
            raise ValueError("every failed trial requires a failure_category")
        key = (str(row["scenario"]), str(row["mode"]))
        groups.setdefault(key, []).append(dict(row))
    scenario_map = {item.label: item for item in build_scenarios()}
    aggregates = []
    for (scenario_label, mode), selected in sorted(groups.items()):
        successes = sum(bool(row["passed"]) for row in selected)
        low, high = wilson_interval(successes, len(selected), confidence)
        event_values = [
            float(row["event_match_ratio"])
            for row in selected
            if "event_match_ratio" in row
        ]
        scenario = scenario_map.get(scenario_label)
        aggregates.append({
            "scenario": scenario_label,
            "mode": mode,
            "critical": bool(scenario.critical) if scenario else False,
            "trial_count": len(selected),
            "success_count": successes,
            "failure_count": len(selected) - successes,
            "success_rate": successes / len(selected),
            "wilson_95_low": low,
            "wilson_95_high": high,
            "mean_event_match_ratio": (
                sum(event_values) / len(event_values)
                if event_values else float("nan")
            ),
        })
    return aggregates


def evaluate_matrix(
    aggregates: Iterable[dict[str, Any]],
    critical_minimum: float = 0.90,
    aggregate_minimum: float = 0.95,
) -> dict[str, Any]:
    rows = [
        dict(row) for row in aggregates
        if str(row.get("mode")) == "C2_final"
    ]
    if not rows:
        raise ValueError("C2_final aggregates are required")
    by_name = {str(row["scenario"]): row for row in rows}
    nominal_ok = (
        "nominal" in by_name
        and float(by_name["nominal"]["success_rate"]) == 1.0
    )
    critical_rows = [row for row in rows if bool(row.get("critical", False))]
    critical_ok = bool(critical_rows) and all(
        float(row["success_rate"]) >= critical_minimum
        for row in critical_rows
    )
    pushes_ok = all(
        label in by_name
        and float(by_name[label]["success_rate"]) >= critical_minimum
        for label in ("push_minus_4N", "push_plus_4N")
    )
    trials = sum(int(row["trial_count"]) for row in rows)
    successes = sum(int(row["success_count"]) for row in rows)
    overall_rate = successes / max(trials, 1)
    aggregate_ok = overall_rate >= aggregate_minimum
    checks = {
        "nominal_30_of_30": nominal_ok,
        "critical_scenarios": critical_ok,
        "push_4n_both_directions": pushes_ok,
        "aggregate_c2_final": aggregate_ok,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
        "aggregate_success_rate": overall_rate,
        "aggregate_success_count": successes,
        "aggregate_trial_count": trials,
    }
