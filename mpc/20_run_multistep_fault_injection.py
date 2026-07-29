#!/usr/bin/env python3
"""Deterministic software fault campaign for the multi-step controller boundary."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mpc.runtime_safety import Fault, RuntimeSafetyMachine

LOG_DIR = ROOT / "results" / "logs"
CSV_PATH = LOG_DIR / "fddp_multistep_fault_injection.csv"
YAML_PATH = LOG_DIR / "fddp_multistep_fault_injection.yaml"
SCHEMA_VERSION = "fddp-multistep-fault-injection-v1"


@dataclass(frozen=True)
class FaultScenario:
    label: str
    fault: Fault
    category: str


def build_fault_scenarios() -> list[FaultScenario]:
    return [
        FaultScenario("stale_pitch", Fault.STALE_MEASUREMENT, "measurement"),
        FaultScenario("nan_pitch", Fault.NAN_MEASUREMENT, "measurement"),
        FaultScenario("inf_pitch", Fault.INF_MEASUREMENT, "measurement"),
        FaultScenario("contact_stuck_on", Fault.STUCK_CONTACT, "contact_sensor"),
        FaultScenario("contact_stuck_off", Fault.STUCK_CONTACT, "contact_sensor"),
        FaultScenario(
            "missing_contact_event", Fault.MISSING_CONTACT_EVENT, "contact_event"
        ),
        FaultScenario(
            "out_of_order_contact_event",
            Fault.OUT_OF_ORDER_CONTACT_EVENT,
            "contact_event",
        ),
        FaultScenario("invalid_qp", Fault.INVALID_QP, "online_controller"),
        FaultScenario("deadline_miss", Fault.DEADLINE_MISS, "timing"),
        FaultScenario(
            "command_saturation", Fault.COMMAND_SATURATION, "command"
        ),
        FaultScenario("rate_saturation", Fault.RATE_SATURATION, "command"),
        FaultScenario("actuator_delay", Fault.ACTUATOR_DELAY, "actuator"),
        FaultScenario(
            "missing_reference", Fault.MISSING_REFERENCE, "startup_integrity"
        ),
        FaultScenario(
            "truncated_reference", Fault.TRUNCATED_REFERENCE, "startup_integrity"
        ),
        FaultScenario(
            "invalid_reference_schema",
            Fault.INVALID_REFERENCE_SCHEMA,
            "startup_integrity",
        ),
    ]


def _run_scenario(scenario: FaultScenario, index: int) -> dict[str, Any]:
    q_hold = np.zeros(6, dtype=float)
    machine = RuntimeSafetyMachine(q_hold=q_hold)
    now_s = 0.02 * index
    uncaught_exception = False
    error = ""
    command = q_hold.copy()
    try:
        machine.handle(scenario.fault, now_s=now_s)
        command = machine.safe_command(
            q_last_valid=np.zeros(6), q_hold=q_hold, dt_s=0.020
        )
    except Exception as exc:
        uncaught_exception = True
        error = f"{type(exc).__name__}: {exc}"
    command = np.asarray(command, dtype=float).reshape(6)
    finite = bool(np.all(np.isfinite(command)))
    low = machine._limiter.low
    high = machine._limiter.high
    within_limits = bool(
        finite and np.all(command >= low - 1e-12)
        and np.all(command <= high + 1e-12)
    )
    rate_bounded = bool(
        finite and np.all(np.abs(command) <= machine.max_step_rad + 1e-12)
    )
    transition = machine.transitions[-1] if machine.transitions else {
        "time_s": now_s,
        "reason": scenario.fault.name,
        "previous_state": "NORMAL",
        "next_state": machine.state.name,
        "recovery_outcome": "exception",
    }
    return {
        "scenario": scenario.label,
        "category": scenario.category,
        "fault": scenario.fault.name,
        **transition,
        "uncaught_exception": uncaught_exception,
        "error": error,
        "command_finite": finite,
        "command_within_limits": within_limits,
        "command_rate_bounded": rate_bounded,
        "maximum_abs_command_rad": float(np.max(np.abs(command))),
    }


def run_fault_campaign(write_outputs: bool = True) -> dict[str, Any]:
    rows = [
        _run_scenario(scenario, index)
        for index, scenario in enumerate(build_fault_scenarios())
    ]
    accepted = bool(all(
        not row["uncaught_exception"]
        and row["command_finite"]
        and row["command_within_limits"]
        and row["command_rate_bounded"]
        for row in rows
    ))
    report = {
        "schema_version": SCHEMA_VERSION,
        "stage": "fddp_multistep_runtime_fault_injection",
        "claim_boundary": "software_only_position_command",
        "scenario_count": len(rows),
        "accepted": accepted,
        "failures": [
            row["scenario"] for row in rows
            if not (
                not row["uncaught_exception"]
                and row["command_finite"]
                and row["command_within_limits"]
                and row["command_rate_bounded"]
            )
        ],
        "scenarios": [asdict(item) | {"fault": item.fault.name}
                      for item in build_fault_scenarios()],
        "rows": rows,
        "outputs": {"csv": str(CSV_PATH), "yaml": str(YAML_PATH)},
    }
    if write_outputs:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
        YAML_PATH.write_text(
            yaml.safe_dump(report, sort_keys=False), encoding="utf-8"
        )
    return report


def main() -> None:
    report = run_fault_campaign(write_outputs=True)
    print(yaml.safe_dump(report, sort_keys=False))
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
