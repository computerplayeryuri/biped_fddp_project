from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd

from common import FIG_DIR, LOG_DIR, ensure_output_dirs

SUMMARY_PATH = LOG_DIR / "baseline_pitch_sweep_summary.csv"


def main() -> None:
    ensure_output_dirs()
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SUMMARY_PATH}. Run 08c_baseline_pitch_sweep.py first."
        )

    summary = pd.read_csv(SUMMARY_PATH)

    # Figure 1: all pitch responses.
    fig, ax = plt.subplots(figsize=(11, 6))
    for _, row in summary.iterrows():
        log_path = LOG_DIR / row["log_file"]
        data = pd.read_csv(log_path)
        label = f"{row['initial_pitch_deg']:+.0f} deg"
        ax.plot(data["time_s"], data["torso_pitch_deg"], label=label)

    ax.axhline(1.0, linestyle="--")
    ax.axhline(-1.0, linestyle="--")
    ax.set_title("C0 baseline: initial-pitch sweep")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torso pitch (deg)")
    ax.grid(True)
    ax.legend(ncol=3)
    fig.tight_layout()
    pitch_path = FIG_DIR / "baseline_pitch_sweep_response.png"
    fig.savefig(pitch_path, dpi=160)
    plt.close(fig)

    # Figure 2: settling time and peak tracking error.
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    x = summary["initial_pitch_deg"]
    settling = summary["settling_time_s"].copy()
    settling = settling.fillna(6.0)

    axes[0].bar(x, settling, width=1.5)
    axes[0].set_ylabel("Settling time (s)")
    axes[0].set_title("C0 baseline: recovery and tracking summary")
    axes[0].grid(True, axis="y")

    axes[1].bar(x, summary["peak_joint_rmse_deg"], width=1.5)
    axes[1].set_xlabel("Initial torso pitch (deg)")
    axes[1].set_ylabel("Peak joint RMSE (deg)")
    axes[1].grid(True, axis="y")

    fig.tight_layout()
    summary_path = FIG_DIR / "baseline_pitch_sweep_summary.png"
    fig.savefig(summary_path, dpi=160)
    plt.close(fig)

    print(f"Saved: {pitch_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
