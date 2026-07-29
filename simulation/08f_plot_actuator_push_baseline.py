from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from common import FIG_DIR, LOG_DIR, ensure_output_dirs

SUMMARY_PATH = LOG_DIR / "actuator_push_baseline_summary.csv"


def main() -> None:
    ensure_output_dirs()
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SUMMARY_PATH}. Run 08e_actuator_push_baseline.py first."
        )

    summary = pd.read_csv(SUMMARY_PATH).sort_values("push_force_N")

    # 1. Pitch response overlay.
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for _, row in summary.iterrows():
        data = pd.read_csv(LOG_DIR / row["log_file"])
        ax.plot(data["time_s"], data["torso_pitch_deg"], label=f"{row['push_force_N']:+.0f} N")
    ax.axvspan(1.00, 1.12, alpha=0.15, label="push window")
    ax.axhline(1.0, linestyle="--")
    ax.axhline(-1.0, linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torso pitch (deg)")
    ax.set_title("C0 actuator-aware baseline: response to short horizontal pushes")
    ax.grid(True)
    ax.legend(ncol=2)
    fig.tight_layout()
    out_pitch = FIG_DIR / "actuator_push_baseline_pitch_response.png"
    fig.savefig(out_pitch, dpi=160)
    plt.close(fig)

    # 2. Summary metrics.
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 9.0), sharex=True)
    x = summary["push_force_N"]
    axes[0].bar(x, summary["peak_abs_pitch_deg"], width=0.9)
    axes[0].set_ylabel("Peak |pitch| (deg)")
    axes[0].grid(True, axis="y")

    settling = summary["settling_time_s"].fillna(4.0)
    axes[1].bar(x, settling, width=0.9)
    axes[1].set_ylabel("Settling time (s)")
    axes[1].grid(True, axis="y")

    axes[2].bar(x, summary["peak_forward_displacement_mm"], width=0.9)
    axes[2].set_xlabel("Applied horizontal force pulse (N)")
    axes[2].set_ylabel("Peak forward shift (mm)")
    axes[2].grid(True, axis="y")

    fig.suptitle("C0 actuator-aware push baseline: recovery summary")
    fig.tight_layout()
    out_summary = FIG_DIR / "actuator_push_baseline_summary.png"
    fig.savefig(out_summary, dpi=160)
    plt.close(fig)

    print(f"Saved: {out_pitch}")
    print(f"Saved: {out_summary}")


if __name__ == "__main__":
    main()
