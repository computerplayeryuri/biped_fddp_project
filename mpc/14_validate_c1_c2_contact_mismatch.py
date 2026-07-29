#!/usr/bin/env python3
"""E5 event-layer comparison of fixed-phase C1 and phase-synchronised C2."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
C2_PATH = ROOT / "mpc" / "08_run_contact_phase_synchronised_fir_qp_mpc.py"
LOG_DIR = ROOT / "results" / "logs"
FIG_DIR = ROOT / "results" / "figures"
EXPECTED_EVENTS = [
    {"time_s": 0.36, "kind": "left_liftoff"},
    {"time_s": 1.00, "kind": "left_touchdown"},
    {"time_s": 1.30, "kind": "right_liftoff"},
    {"time_s": 1.95, "kind": "right_touchdown"},
]
SCENARIOS = ("early_touchdown_30ms", "late_touchdown_30ms", "sensor_delay_30ms", "short_contact_loss_40ms")


def load_c2():
    spec = importlib.util.spec_from_file_location("c2_phase_sync", C2_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def inject_event_mismatch(events, scenario: str):
    result = [dict(event) for event in events]
    if scenario == "early_touchdown_30ms":
        result[1]["time_s"] -= 0.030
    elif scenario in ("late_touchdown_30ms", "sensor_delay_30ms"):
        result[1]["time_s"] += 0.030
    elif scenario == "short_contact_loss_40ms":
        result[2]["time_s"] += 0.040
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return result


def phase_offset_for_mode(mode: str, offset_s: float) -> float:
    if mode == "C1_fixed_phase":
        return 0.0
    if mode == "C2_phase_sync":
        return float(offset_s)
    raise ValueError(f"Unknown mode: {mode}")


def run_mode(c2, mode: str, scenario: str):
    observed = inject_event_mismatch(EXPECTED_EVENTS, scenario)
    sync = c2.ContactPhaseSynchronizer([dict(event) for event in EXPECTED_EVENTS])
    rows = []
    for index, event in enumerate(observed):
        records = [] if mode == "C1_fixed_phase" else sync.update(event["time_s"], [event])
        matching = next((record for record in records if record.get("status") == "matched"), None)
        expected = EXPECTED_EVENTS[index]
        offset = phase_offset_for_mode(mode, sync.offset_s)
        rows.append({
            "mode": mode, "scenario": scenario, "event_index": index,
            "event_kind": event["kind"], "scheduled_event_time_s": expected["time_s"],
            "observed_event_time_s": event["time_s"],
            "event_timing_error_ms": (event["time_s"] - expected["time_s"]) * 1000.0,
            "phase_offset_ms": offset * 1000.0,
            "reference_time_error_ms": offset * 1000.0,
            "matched": bool(matching is not None),
        })
    return rows


def main():
    c2 = load_c2()
    rows = [row for scenario in SCENARIOS for mode in ("C1_fixed_phase", "C2_phase_sync") for row in run_mode(c2, mode, scenario)]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(rows)
    csv_path = LOG_DIR / "e5_c1_c2_contact_mismatch.csv"
    dataframe.to_csv(csv_path, index=False)
    summary = dataframe.groupby(["mode", "scenario"], as_index=False).agg(
        matched_event_ratio=("matched", "mean"),
        max_abs_phase_offset_ms=("phase_offset_ms", lambda values: float(abs(values).max())),
        max_abs_reference_time_error_ms=("reference_time_error_ms", lambda values: float(abs(values).max())),
    )
    summary_path = LOG_DIR / "e5_c1_c2_contact_mismatch_summary.yaml"
    summary_path.write_text(yaml.safe_dump({"stage": "E5_event_layer_c1_vs_c2", "limitation": "Event-layer injection only; not terrain dynamics.", "results": summary.to_dict(orient="records"), "outputs": {"csv": str(csv_path)}}, sort_keys=False), encoding="utf-8")
    figure, axis = plt.subplots(figsize=(11, 5))
    for (mode, scenario), group in dataframe.groupby(["mode", "scenario"], sort=False):
        axis.plot(group["event_index"], group["phase_offset_ms"], marker="o", label=f"{mode}: {scenario}")
    axis.set_xlabel("Event index"); axis.set_ylabel("Phase offset (ms)"); axis.grid(True); axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure_path = FIG_DIR / "e5_c1_c2_phase_error_comparison.png"
    figure.savefig(figure_path, dpi=180); plt.close(figure)
    print(f"E5 summaries: {len(summary)}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
