#!/usr/bin/env python3
from __future__ import annotations

"""
C1-QP1 — Empirical FIR identification for MuJoCo pitch-response model.

Reads C1-QP0 calibration CSVs and identifies:
    y[k] ≈ Σ h[i] alpha[k-i]
at the 20 ms QP update rate.
"""

from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "results" / "logs"
FIG_DIR = ROOT / "results" / "figures"

SAMPLE_TIME_S = 0.020
FIR_LENGTH = 15


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def interp_series(time_old: np.ndarray, values_old: np.ndarray, time_new: np.ndarray):
    return np.interp(time_new, time_old, values_old)


def resample_run(df: pd.DataFrame, t_grid: np.ndarray):
    t = df["time_s"].to_numpy(dtype=float)
    y = df["root_pitch_rad"].to_numpy(dtype=float)
    u = df["alpha_pulse_rad"].to_numpy(dtype=float) if "alpha_pulse_rad" in df.columns else np.zeros_like(t)
    return {
        "time_s": t_grid.copy(),
        "root_pitch_rad": interp_series(t, y, t_grid),
        "alpha_rad": interp_series(t, u, t_grid),
    }


def build_fir_regression(u: np.ndarray, y: np.ndarray, M: int):
    N = len(u)
    Phi = np.zeros((N, M), dtype=float)
    for k in range(N):
        for i in range(M):
            if k - i >= 0:
                Phi[k, i] = u[k - i]
    return Phi, y.copy()


def identify_fir(u_list: list[np.ndarray], y_list: list[np.ndarray], M: int):
    Phi_blocks = []
    y_blocks = []
    for u, y in zip(u_list, y_list):
        Phi, yy = build_fir_regression(u, y, M)
        Phi_blocks.append(Phi)
        y_blocks.append(yy)
    Phi_all = np.vstack(Phi_blocks)
    y_all = np.concatenate(y_blocks)
    reg = 1e-8 * np.eye(M)
    h = np.linalg.solve(Phi_all.T @ Phi_all + reg, Phi_all.T @ y_all)
    predictions = [build_fir_regression(u, y, M)[0] @ h for u, y in zip(u_list, y_list)]
    return h, predictions


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def holdout_metrics(
    train_u: np.ndarray,
    train_y: np.ndarray,
    validation_u: np.ndarray,
    validation_y: np.ndarray,
    *,
    sample_time_s: float,
    fir_length: int,
) -> dict:
    """Fit an FIR on one pulse and evaluate it on a disjoint pulse."""
    h, _ = identify_fir([train_u], [train_y], fir_length)
    validation_phi, _ = build_fir_regression(validation_u, validation_y, fir_length)
    prediction = validation_phi @ h
    validation_range = float(np.ptp(validation_y))
    error = validation_y - prediction
    threshold = 0.05 * float(np.max(np.abs(h))) if len(h) else 0.0
    significant = np.where(np.abs(h) >= threshold)[0] if threshold > 0.0 else np.array([], dtype=int)
    return {
        "fir_coefficients_rad_per_rad": [float(value) for value in h],
        "prediction_rad": np.asarray(prediction, dtype=float),
        "one_step_rmse_rad": float(abs(error[0])) if len(error) else 0.0,
        "multi_step_rmse_rad": rmse(validation_y, prediction),
        "nrmse": None if validation_range <= 1e-12 else float(rmse(validation_y, prediction) / validation_range),
        "estimated_delay_s": None if len(significant) == 0 else float(significant[0] * sample_time_s),
        "prediction_horizon_samples": int(len(validation_y)),
    }


def analyze_direction(direction: str):
    baseline_path = LOG_DIR / f"c1_qp0_baseline_{direction}.csv"
    plus_path = LOG_DIR / f"c1_qp0_plus_pulse_{direction}.csv"
    minus_path = LOG_DIR / f"c1_qp0_minus_pulse_{direction}.csv"

    baseline_df = load_csv(baseline_path)
    plus_df = load_csv(plus_path)
    minus_df = load_csv(minus_path)

    t_end = min(
        baseline_df["time_s"].max(),
        plus_df["time_s"].max(),
        minus_df["time_s"].max(),
    )
    t_grid = np.arange(0.0, t_end + 1e-12, SAMPLE_TIME_S)

    baseline = resample_run(baseline_df, t_grid)
    plus = resample_run(plus_df, t_grid)
    minus = resample_run(minus_df, t_grid)

    y_plus = plus["root_pitch_rad"] - baseline["root_pitch_rad"]
    y_minus = minus["root_pitch_rad"] - baseline["root_pitch_rad"]
    u_plus = plus["alpha_rad"]
    u_minus = minus["alpha_rad"]

    h, predictions = identify_fir([u_plus, u_minus], [y_plus, y_minus], FIR_LENGTH)
    yhat_plus, yhat_minus = predictions

    fit_plus_rmse = rmse(y_plus, yhat_plus)
    fit_minus_rmse = rmse(y_minus, yhat_minus)

    impulse = h.copy()
    step = np.cumsum(h)

    max_abs_idx = int(np.argmax(np.abs(impulse)))
    peak_impulse = float(impulse[max_abs_idx])
    peak_time = float(max_abs_idx * SAMPLE_TIME_S)
    threshold = 0.05 * np.max(np.abs(impulse))
    above = np.where(np.abs(impulse) >= threshold)[0]
    effective_delay = float(above[0] * SAMPLE_TIME_S) if len(above) else None

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t_grid, np.rad2deg(y_plus), label="measured +pulse")
    ax.plot(t_grid, np.rad2deg(yhat_plus), "--", label="predicted +pulse")
    ax.plot(t_grid, np.rad2deg(y_minus), label="measured -pulse")
    ax.plot(t_grid, np.rad2deg(yhat_minus), "--", label="predicted -pulse")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch deviation from baseline (deg)")
    ax.set_title(f"C1-QP1 FIR fit — {direction}")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fit_plot = FIG_DIR / f"c1_qp1_fir_fit_{direction}.png"
    fig.savefig(fit_plot, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    k = np.arange(FIR_LENGTH)
    markerline, stemlines, baseline_line = ax.stem(k * SAMPLE_TIME_S, impulse)
    ax.set_xlabel("Lag time (s)")
    ax.set_ylabel("Impulse response h[i] (rad/rad)")
    ax.set_title(f"C1-QP1 identified FIR impulse response — {direction}")
    ax.grid(True)
    fig.tight_layout()
    impulse_plot = FIG_DIR / f"c1_qp1_impulse_{direction}.png"
    fig.savefig(impulse_plot, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(k * SAMPLE_TIME_S, step, marker="o")
    ax.set_xlabel("Lag time (s)")
    ax.set_ylabel("Step response cumulative sum")
    ax.set_title(f"C1-QP1 identified FIR step response — {direction}")
    ax.grid(True)
    fig.tight_layout()
    step_plot = FIG_DIR / f"c1_qp1_step_{direction}.png"
    fig.savefig(step_plot, dpi=180)
    plt.close(fig)

    return {
        "direction": direction,
        "sample_time_s": SAMPLE_TIME_S,
        "fir_length": FIR_LENGTH,
        "fir_coefficients_rad_per_rad": [float(v) for v in impulse],
        "step_response_cumsum": [float(v) for v in step],
        "fit_metrics": {
            "plus_pulse_rmse_rad": fit_plus_rmse,
            "plus_pulse_rmse_deg": float(np.rad2deg(fit_plus_rmse)),
            "minus_pulse_rmse_rad": fit_minus_rmse,
            "minus_pulse_rmse_deg": float(np.rad2deg(fit_minus_rmse)),
        },
        "identified_features": {
            "peak_impulse_value_rad_per_rad": peak_impulse,
            "peak_impulse_time_s": peak_time,
            "effective_delay_s": effective_delay,
            "dc_gain_sum_h": float(np.sum(impulse)),
            "note": (
                "Use this FIR kernel in the lifted QP prediction model for C1-QP v2. "
                "If coefficients change sign, the online model must retain the full FIR "
                "sequence rather than collapsing it to a single gain."
            ),
        },
        "outputs": {
            "fit_plot": str(fit_plot),
            "impulse_plot": str(impulse_plot),
            "step_plot": str(step_plot),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Identify empirical FIR pitch-response model for C1-QP."
    )
    parser.add_argument(
        "--direction",
        choices=("forward", "backward", "both"),
        default="both",
    )
    args = parser.parse_args()
    directions = ("forward", "backward") if args.direction == "both" else (args.direction,)

    print("===== C1-QP1: EMPIRICAL FIR IDENTIFICATION =====")
    print(f"QP sample time: {SAMPLE_TIME_S:.3f} s")
    print(f"FIR length: {FIR_LENGTH} samples = {FIR_LENGTH * SAMPLE_TIME_S:.3f} s")

    report = {
        "stage": "C1_QP1_empirical_FIR_identification",
        "purpose": (
            "Identify a discrete-time FIR model from alpha correction to MuJoCo "
            "root-pitch deviation using the C1-QP0 calibration pulses."
        ),
        "directions": {},
    }

    for direction in directions:
        result = analyze_direction(direction)
        report["directions"][direction] = result
        print()
        print(f"----- {direction.upper()} -----")
        print(
            "Peak impulse h[i]: "
            f"{result['identified_features']['peak_impulse_value_rad_per_rad']:+.6f} "
            f"at {result['identified_features']['peak_impulse_time_s']:.3f}s"
        )
        print(
            "Effective delay: "
            f"{result['identified_features']['effective_delay_s']}"
        )
        print(
            "Fit RMSE (+/- pulse) deg: "
            f"{result['fit_metrics']['plus_pulse_rmse_deg']:.4f} / "
            f"{result['fit_metrics']['minus_pulse_rmse_deg']:.4f}"
        )
        print(
            "DC gain sum(h): "
            f"{result['identified_features']['dc_gain_sum_h']:+.6f}"
        )

    summary_path = LOG_DIR / "c1_qp1_fir_identification.yaml"
    summary_path.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")

    print()
    print("C1-QP1 IDENTIFICATION COMPLETE")
    print(f"Saved summary:\n  {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\\nC1-QP1 IDENTIFICATION FAILED: {error}", file=sys.stderr)
        sys.exit(1)
