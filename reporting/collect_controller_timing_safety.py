#!/usr/bin/env python3
"""E6 timing and available-safety evidence from C1/C2 simulation logs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "results" / "logs"
FIG_DIR = ROOT / "results" / "figures"
CONTROLLER_PERIOD_S = 0.020


def summarize_timing(dataframe: pd.DataFrame, controller_period_s: float) -> dict:
    values = dataframe.loc[dataframe["qp_solve_time_us"] > 0.0, "qp_solve_time_us"]
    deadline_us = controller_period_s * 1e6
    return {
        "controller_period_s": float(controller_period_s),
        "update_count": int(len(values)),
        "p50_qp_solve_time_us": None if values.empty else float(values.quantile(0.50)),
        "p95_qp_solve_time_us": None if values.empty else float(values.quantile(0.95)),
        "max_qp_solve_time_us": None if values.empty else float(values.max()),
        "deadline_miss_count": int((values > deadline_us).sum()),
    }


def summarize_available_safety(dataframe: pd.DataFrame) -> dict:
    correction_columns = [column for column in dataframe.columns if column.endswith("_delta_rad")]
    maximum = float(dataframe[correction_columns].abs().to_numpy().max()) if correction_columns else 0.0
    return {
        "max_abs_joint_correction_rad": maximum,
        "position_clip_available": False,
        "speed_clip_available": False,
        "increment_clip_available": False,
        "fallback_available": False,
        "note": "Current C1/C2 logs contain applied corrections but not explicit clip or fallback counters.",
    }


def main():
    sources = {
        "C1_forward": LOG_DIR / "c1_qp2_balanced_forward.csv",
        "C1_backward": LOG_DIR / "c1_qp2_balanced_backward.csv",
        "C2_forward": LOG_DIR / "c2_phase_sync_forward.csv",
        "C2_backward": LOG_DIR / "c2_phase_sync_backward.csv",
    }
    rows = []
    for label, path in sources.items():
        dataframe = pd.read_csv(path)
        rows.append({"controller_run": label, "source_csv": str(path), **summarize_timing(dataframe, CONTROLLER_PERIOD_S), **summarize_available_safety(dataframe)})
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(rows)
    csv_path = LOG_DIR / "e6_controller_timing_safety.csv"
    summary_df.to_csv(csv_path, index=False)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(summary_df["controller_run"], summary_df["p95_qp_solve_time_us"])
    axis.axhline(CONTROLLER_PERIOD_S * 1e6, color="red", linestyle="--", label="20 ms deadline")
    axis.set_ylabel("QP solve time (us)"); axis.grid(True, axis="y"); axis.legend(); figure.tight_layout()
    figure_path = FIG_DIR / "e6_controller_timing_distribution.png"
    figure.savefig(figure_path, dpi=180); plt.close(figure)
    summary_path = LOG_DIR / "e6_controller_timing_safety_summary.yaml"
    summary_path.write_text(yaml.safe_dump({"stage": "E6_online_timing_and_available_safety", "results": rows, "outputs": {"csv": str(csv_path), "figure": str(figure_path)}}, sort_keys=False), encoding="utf-8")
    print(f"E6 runs: {len(rows)}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
