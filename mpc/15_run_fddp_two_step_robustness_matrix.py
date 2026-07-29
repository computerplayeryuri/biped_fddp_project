#!/usr/bin/env python3
"""Robustness schema for the independent two-step FDDP simulation study."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Scenario:
    label: str
    category: str
    repeat_count: int = 5
    friction_mu: float = 0.8
    slope_deg: float = 0.0
    step_height_m: float = 0.0
    push_force_n: float = 0.0
    command_delay_s: float = 0.004
    sensor_delay_s: float = 0.0


def build_scenarios() -> list[Scenario]:
    """Return the minimum non-scripted FDDP two-step robustness matrix."""
    return [
        Scenario("nominal", "nominal"),
        Scenario("push_plus_4N", "push", push_force_n=4.0),
        Scenario("friction_mu_0p50", "friction", friction_mu=0.50),
        Scenario("slope_plus_3deg", "slope", slope_deg=3.0),
        Scenario("step_4mm", "step", step_height_m=0.004),
        Scenario("command_delay_12ms", "command_delay", command_delay_s=0.012),
        Scenario("sensor_delay_20ms", "sensor_delay", sensor_delay_s=0.020),
    ]


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson 95 percent confidence interval."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be within [0, trials]")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)



def summarize_trials(trials: list[dict]) -> dict:
    """Aggregate explicit trial pass/fail and scheduled-event evidence."""
    if not trials:
        raise ValueError("trials must not be empty")
    trial_count = len(trials)
    success_count = sum(bool(trial["passed"]) for trial in trials)
    matched = sum(int(trial["matched_contact_events"]) for trial in trials)
    scheduled = sum(int(trial["scheduled_contact_events"]) for trial in trials)
    return {
        "trial_count": trial_count,
        "success_count": success_count,
        "success_rate": success_count / trial_count,
        "success_rate_wilson_95": wilson_interval(success_count, trial_count),
        "event_match_rate": matched / max(scheduled, 1),
    }



def trial_passed(metrics: dict) -> dict:
    """Apply explicit stability and natural-event evidence gates to one trial."""
    failures: list[str] = []
    if metrics["min_torso_height_m"] < 0.65 * metrics["initial_torso_height_m"]:
        failures.append("torso height below stability threshold")
    if metrics["min_global_contact_count"] < 1:
        failures.append("lost ground contact")
    if metrics["max_abs_torso_pitch_deg"] > 30.0:
        failures.append("torso pitch exceeds fall threshold")
    if metrics["scheduled_contact_events"] != 4:
        failures.append("unexpected scheduled contact event count")
    if metrics.get("mode", "c2") != "c0" and metrics["matched_contact_events"] < 1:
        failures.append("no natural contact event matched")
    return {"passed": not failures, "failures": failures, "metrics": dict(metrics)}


from collections import deque
import importlib.util
from pathlib import Path
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_trial(scenario: Scenario, mode: str = "c2", seed: int = 0) -> dict:
    """Run one FDDP-two-step MuJoCo trial with C0 or natural-event C2 clock."""
    if mode not in ("c0", "c2"):
        raise ValueError("mode must be c0 or c2")
    c2 = _load_module(ROOT / "mpc" / "14_validate_fddp_two_step_natural_contact_c2.py", "trial_c2")
    stage9 = _load_module(ROOT / "mpc" / "13_run_scripted_robustness_matrix.py", "terrain_injector")
    proxy = _load_module(ROOT / "simulation" / "03o_replay_fddp_com_shift_mujoco.py", "trial_proxy")
    proxy.COMMAND_DELAY_S = float(scenario.command_delay_s)
    reference = c2.load_reference()
    scheduled = c2.build_reference_events(reference)
    model = stage9.make_model(scenario)
    data = mujoco.MjData(model)
    mapping = proxy.resolve_joint_and_actuator_mapping(model)
    torso = proxy.resolve_torso_body(model)
    rng = np.random.default_rng(seed)
    for address, value in zip(mapping["qpos_addresses"], reference["q_ref"][0]):
        data.qpos[address] = value + float(rng.normal(0.0, 1.0e-3))
    mujoco.mj_forward(model, data)
    h0 = float(data.xpos[torso][2])
    observer = _load_module(
        ROOT / "mpc" / "08_run_contact_phase_synchronised_fir_qp_mpc.py",
        "trial_c2_common",
    ).ContactObserver(model, data, float(model.opt.timestep))
    sync = _load_module(
        ROOT / "mpc" / "08_run_contact_phase_synchronised_fir_qp_mpc.py",
        "trial_c2_sync",
    ).ContactPhaseSynchronizer(scheduled)
    actuator = proxy.PositionActuatorProxy(6, float(model.opt.timestep))
    actuator.reset(reference["q_ref"][0])
    duration = float(reference["t_ref"][-1])
    observed_count = 0
    max_pitch = 0.0
    min_height = h0
    min_contacts = int(data.ncon)
    delayed_events = deque()
    while data.time <= duration:
        wall = min(float(data.time), duration)
        values, _, _, observed = observer.update(model, data, wall)
        for event in observed:
            delayed_events.append((wall + scenario.sensor_delay_s, event))
        ready = []
        while delayed_events and delayed_events[0][0] <= wall:
            ready.append(delayed_events.popleft()[1])
        observed_count += len(ready)
        if mode == "c2":
            sync.update(wall, ready)
            ref_time = sync.reference_time(wall, duration)
        else:
            ref_time = wall
        q_command = proxy.interpolate_reference(ref_time, reference["t_ref"], reference["q_ref"])
        q_proxy = actuator.update(q_command)
        if abs(scenario.push_force_n) > 0.0 and 0.8 <= wall < 0.92:
            data.xfrc_applied[torso, 1] = scenario.push_force_n
        data.ctrl[:] = 0.0
        for aid, value in zip(mapping["actuator_ids"], q_proxy):
            data.ctrl[aid] = value
        mujoco.mj_step(model, data)
        max_pitch = max(max_pitch, abs(float(np.rad2deg(proxy.torso_pitch_about_x(data, torso)))))
        min_height = min(min_height, float(data.xpos[torso][2]))
        min_contacts = min(min_contacts, int(data.ncon))
    metrics = {
        "mode": mode,
        "scenario_label": scenario.label,
        "seed": int(seed),
        "min_torso_height_m": min_height,
        "initial_torso_height_m": h0,
        "min_global_contact_count": min_contacts,
        "max_abs_torso_pitch_deg": max_pitch,
        "scheduled_contact_events": len(scheduled),
        "matched_contact_events": len(sync.matched) if mode == "c2" else 0,
        "natural_observed_contact_events": observed_count,
    }
    return trial_passed(metrics)



def run_matrix(
    scenarios: list[Scenario] | None = None,
    modes: tuple[str, ...] = ("c0", "c2"),
) -> dict:
    """Execute scenario x seed x mode trials and write aggregate evidence."""
    import pandas as pd
    import yaml

    scenarios = build_scenarios() if scenarios is None else scenarios
    rows: list[dict] = []
    aggregates: list[dict] = []
    for scenario in scenarios:
        for mode in modes:
            trials: list[dict] = []
            for seed in range(scenario.repeat_count):
                decision = run_trial(scenario, mode=mode, seed=seed)
                trial = {**decision["metrics"], "passed": decision["passed"],
                         "failures": "; ".join(decision["failures"])}
                rows.append(trial)
                trials.append(trial)
            aggregate = summarize_trials(trials)
            aggregates.append({
                "scenario_label": scenario.label, "category": scenario.category,
                "mode": mode, **aggregate,
            })
    log_dir = ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    trial_csv = log_dir / "fddp_two_step_robustness_trials.csv"
    aggregate_csv = log_dir / "fddp_two_step_robustness_aggregate.csv"
    summary_yaml = log_dir / "fddp_two_step_robustness_summary.yaml"
    pd.DataFrame(rows).to_csv(trial_csv, index=False)
    aggregate_frame = pd.DataFrame(aggregates)
    aggregate_frame.to_csv(aggregate_csv, index=False)
    import matplotlib.pyplot as plt
    figure_path = ROOT / "results" / "figures" / "fddp_two_step_robustness_success_rates.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 4))
    labels = ["{0}\n{1}".format(row["scenario_label"], row["mode"]) for row in aggregates]
    values = [row["success_rate"] for row in aggregates]
    axis.bar(range(len(values)), values)
    axis.set_ylim(0.0, 1.05); axis.set_ylabel("success rate")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.grid(axis="y"); figure.tight_layout()
    figure.savefig(figure_path, dpi=180); plt.close(figure)
    summary = {
        "stage": "fddp_two_step_c0_c2_robustness_batch",
        "claim_boundary": "Simulation-only FDDP nominal replay; not hardware walking.",
        "inputs": {"reference": "fddp_two_step_alternating_ref.npz",
                   "tau_ref_usage": "analysis_only"},
        "trial_count": len(rows),
        "aggregates": aggregates,
        "outputs": {"trials_csv": str(trial_csv),
                    "aggregate_csv": str(aggregate_csv), "figure": str(figure_path)},
    }
    summary_yaml.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    summary["outputs"]["summary_yaml"] = str(summary_yaml)
    return summary


def main() -> int:
    summary = run_matrix()
    print(f"Completed {summary['trial_count']} FDDP robustness trials.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
