#!/usr/bin/env python3
"""
C2 phase synchronizer logic validation with synthetic contact events.

This script does not claim walking validation. It checks the event-matching and
bounded phase-offset update logic used by
08_run_contact_phase_synchronised_fir_qp_mpc.py under controlled synthetic
contact timing errors.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
C2_PATH = ROOT / "mpc" / "08_run_contact_phase_synchronised_fir_qp_mpc.py"
LOG = ROOT / "results" / "logs"
FIG = ROOT / "results" / "figures"

REFERENCE_EVENTS = [
    {"time_s": 0.20, "kind": "left_liftoff"},
    {"time_s": 0.45, "kind": "left_touchdown"},
    {"time_s": 0.70, "kind": "right_liftoff"},
    {"time_s": 0.95, "kind": "right_touchdown"},
]

SCENARIOS = {
    "observed_late_30ms": [0.030, 0.030, 0.030, 0.030],
    "observed_early_25ms": [-0.025, -0.025, -0.025, -0.025],
    "mixed_jitter": [0.020, -0.015, 0.010, -0.005],
}


def load_c2_module():
    spec = importlib.util.spec_from_file_location("c2_phase_sync", C2_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_scenario(c2, name, timing_offsets):
    sync = c2.ContactPhaseSynchronizer([dict(event) for event in REFERENCE_EVENTS])
    rows = []
    for index, (reference_event, timing_offset) in enumerate(zip(REFERENCE_EVENTS, timing_offsets)):
        observed_time = float(reference_event["time_s"] + timing_offset)
        observed = {"time_s": observed_time, "kind": reference_event["kind"]}
        records = sync.update(observed_time, [observed])
        matched = [record for record in records if record.get("status") == "matched"]
        status = matched[0]["status"] if matched else records[0].get("status", "missing_record")
        rows.append({
            "scenario": name,
            "event_index": index,
            "event_kind": reference_event["kind"],
            "reference_time_s": float(reference_event["time_s"]),
            "observed_time_s": observed_time,
            "input_timing_error_s": float(timing_offset),
            "match_status": status,
            "phase_offset_ms": float(1000.0 * sync.offset_s),
            "matched_count": int(len(sync.matched)),
            "skipped_count": int(len(sync.skipped)),
        })
    return rows


def plot_results(df):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for scenario, group in df.groupby("scenario"):
        ax.plot(group["event_index"], group["phase_offset_ms"], marker="o", label=scenario)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Synthetic contact event index")
    ax.set_ylabel("Phase offset after event (ms)")
    ax.set_title("C2 phase synchronizer response to synthetic contact timing errors")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    path = FIG / "c2_phase_sync_synthetic_event_validation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    c2 = load_c2_module()
    rows = []
    for name, offsets in SCENARIOS.items():
        rows.extend(run_scenario(c2, name, offsets))
    df = pd.DataFrame(rows)

    LOG.mkdir(parents=True, exist_ok=True)
    csv_path = LOG / "c2_phase_sync_synthetic_event_validation.csv"
    df.to_csv(csv_path, index=False)
    fig_path = plot_results(df)

    scenario_summary = {}
    for scenario, group in df.groupby("scenario"):
        scenario_summary[scenario] = {
            "events": int(len(group)),
            "matched_events": int((group["match_status"] == "matched").sum()),
            "final_phase_offset_ms": float(group["phase_offset_ms"].iloc[-1]),
            "max_abs_phase_offset_ms": float(group["phase_offset_ms"].abs().max()),
        }

    all_matched = bool((df["match_status"] == "matched").all())
    bounded = bool(df["phase_offset_ms"].abs().max() <= 1000.0 * c2.MAX_PHASE_OFFSET_S + 1e-9)
    summary = {
        "stage": "C2_synthetic_phase_synchronizer_logic_validation",
        "scope_note": "Synthetic event timing validation only; not a walking or hardware validation.",
        "reference_events": REFERENCE_EVENTS,
        "scenario_summary": scenario_summary,
        "acceptance": {
            "all_events_matched": all_matched,
            "phase_offset_bound_respected": bounded,
            "max_phase_offset_s": float(c2.MAX_PHASE_OFFSET_S),
        },
        "outputs": {"csv": str(csv_path), "figure": str(fig_path)},
    }
    yaml_path = LOG / "c2_phase_sync_synthetic_event_validation.yaml"
    yaml_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")

    print("C2 synthetic phase synchronizer validation")
    print(df.to_string(index=False))
    print(f"Saved CSV: {csv_path}")
    print(f"Saved YAML: {yaml_path}")
    print(f"Saved figure: {fig_path}")
    if not (all_matched and bounded):
        raise SystemExit("C2 synthetic phase synchronizer validation failed acceptance checks.")


if __name__ == "__main__":
    main()
