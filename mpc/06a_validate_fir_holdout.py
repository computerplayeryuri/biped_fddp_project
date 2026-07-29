#!/usr/bin/env python3
"""E4: sign-held-out validation for the empirical C1 FIR model."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER_PATH = ROOT / "mpc" / "06_identify_qp_mpc_fir_model.py"
LOG_DIR = ROOT / "results" / "logs"
FIG_DIR = ROOT / "results" / "figures"


def load_identifier():
    spec = importlib.util.spec_from_file_location("fir_identifier", IDENTIFIER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def prepare_direction(identifier, direction: str):
    baseline_path = LOG_DIR / f"c1_qp0_baseline_{direction}.csv"
    plus_path = LOG_DIR / f"c1_qp0_plus_pulse_{direction}.csv"
    minus_path = LOG_DIR / f"c1_qp0_minus_pulse_{direction}.csv"
    baseline_df = identifier.load_csv(baseline_path)
    plus_df = identifier.load_csv(plus_path)
    minus_df = identifier.load_csv(minus_path)
    t_end = min(
        baseline_df["time_s"].max(),
        plus_df["time_s"].max(),
        minus_df["time_s"].max(),
    )
    t_grid = np.arange(0.0, t_end + 1e-12, identifier.SAMPLE_TIME_S)
    baseline = identifier.resample_run(baseline_df, t_grid)
    plus = identifier.resample_run(plus_df, t_grid)
    minus = identifier.resample_run(minus_df, t_grid)
    return {
        "time_s": t_grid,
        "plus": {
            "source": str(plus_path),
            "u": plus["alpha_rad"],
            "y": plus["root_pitch_rad"] - baseline["root_pitch_rad"],
        },
        "minus": {
            "source": str(minus_path),
            "u": minus["alpha_rad"],
            "y": minus["root_pitch_rad"] - baseline["root_pitch_rad"],
        },
    }


def validate_direction(identifier, direction: str):
    prepared = prepare_direction(identifier, direction)
    rows, summaries = [], []
    splits = (("train_plus_validate_minus", "plus", "minus"), ("train_minus_validate_plus", "minus", "plus"))
    for split, train_name, validation_name in splits:
        train = prepared[train_name]
        validation = prepared[validation_name]
        result = identifier.holdout_metrics(
            train["u"], train["y"], validation["u"], validation["y"],
            sample_time_s=identifier.SAMPLE_TIME_S,
            fir_length=identifier.FIR_LENGTH,
        )
        summary = {
            "direction": direction,
            "split": split,
            "train_source": train["source"],
            "validation_source": validation["source"],
            **{key: value for key, value in result.items() if key != "prediction_rad"},
        }
        summaries.append(summary)
        for index, (time_s, u, measured, predicted) in enumerate(zip(
            prepared["time_s"], validation["u"], validation["y"], result["prediction_rad"],
        )):
            rows.append({
                "direction": direction,
                "split": split,
                "sample_index": int(index),
                "time_s": float(time_s),
                "alpha_input_rad": float(u),
                "measured_pitch_deviation_rad": float(measured),
                "predicted_pitch_deviation_rad": float(predicted),
                "prediction_error_rad": float(measured - predicted),
            })
    return rows, summaries


def plot_holdout(rows: pd.DataFrame, output: Path):
    directions = list(rows["direction"].drop_duplicates())
    figure, axes = plt.subplots(len(directions), 1, figsize=(12, 5 * len(directions)), squeeze=False)
    for axis, direction in zip(axes[:, 0], directions):
        data = rows.loc[rows["direction"] == direction]
        for split, group in data.groupby("split", sort=False):
            axis.plot(group["time_s"], np.rad2deg(group["measured_pitch_deviation_rad"]), label=f"{split}: measured")
            axis.plot(group["time_s"], np.rad2deg(group["predicted_pitch_deviation_rad"]), "--", label=f"{split}: predicted")
        axis.set_title(f"E4 FIR sign-held-out validation - {direction}")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Pitch deviation (deg)")
        axis.grid(True)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=("forward", "backward", "both"), default="both")
    args = parser.parse_args()
    identifier = load_identifier()
    directions = ("forward", "backward") if args.direction == "both" else (args.direction,)
    all_rows, all_summaries = [], []
    for direction in directions:
        rows, summaries = validate_direction(identifier, direction)
        all_rows.extend(rows)
        all_summaries.extend(summaries)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(all_rows)
    csv_path = LOG_DIR / "e4_fir_holdout_predictions.csv"
    dataframe.to_csv(csv_path, index=False)
    figure_path = FIG_DIR / "e4_fir_holdout_measured_vs_predicted.png"
    plot_holdout(dataframe, figure_path)
    summary_path = LOG_DIR / "e4_fir_holdout_summary.yaml"
    summary = {
        "stage": "E4_fir_sign_held_out_validation",
        "limitation": "One positive and one negative calibration pulse are available per direction; sign-held-out validation is used.",
        "splits": all_summaries,
        "outputs": {"prediction_csv": str(csv_path), "figure": str(figure_path)},
    }
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print(f"E4 splits: {len(all_summaries)}")
    for item in all_summaries:
        print(f"{item['direction']} {item['split']}: RMSE={np.rad2deg(item['multi_step_rmse_rad']):.4f} deg")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
