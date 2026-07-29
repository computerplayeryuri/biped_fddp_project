from __future__ import annotations

"""Plot Stage-5 C0 initial-pitch baseline results."""

import matplotlib.pyplot as plt
import pandas as pd

from common import FIG_DIR, LOG_DIR, ensure_output_dirs

ensure_output_dirs()

files = [
    ("0 deg", LOG_DIR / "baseline_initial_pitch_zero0deg.csv"),
    ("+5 deg", LOG_DIR / "baseline_initial_pitch_plus5deg.csv"),
    ("-5 deg", LOG_DIR / "baseline_initial_pitch_minus5deg.csv"),
]

loaded = []
for label, path in files:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run 08a_initial_pitch_baseline.py first."
        )
    loaded.append((label, pd.read_csv(path)))

fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.8), sharex=True)
for label, df in loaded:
    axes[0].plot(df["time_s"], df["torso_pitch_deg"], label=label)
    axes[1].plot(df["time_s"], df["hip_height_m"], label=label)

axes[0].axhline(0.0, linewidth=1.0)
axes[0].axhline(+1.0, linewidth=0.8, linestyle="--")
axes[0].axhline(-1.0, linewidth=0.8, linestyle="--")
axes[0].set_ylabel("Torso pitch (deg)")
axes[0].set_title("C0 baseline: response to initial base-pitch offsets")
axes[0].grid(True)
axes[0].legend(loc="best")

axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Hip height (m)")
axes[1].grid(True)
axes[1].legend(loc="best")

fig.tight_layout()
out_main = FIG_DIR / "baseline_initial_pitch_response.png"
fig.savefig(out_main, dpi=160, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.5), sharex=True)
for label, df in loaded:
    axes[0].step(df["time_s"], df["left_full_support"], where="post", label=label)
    axes[1].step(df["time_s"], df["right_full_support"], where="post", label=label)
    axes[2].plot(df["time_s"], df["joint_rmse_to_cmd_deg"], label=label)

axes[0].set_ylabel("Left full\nsupport")
axes[1].set_ylabel("Right full\nsupport")
axes[2].set_ylabel("Joint RMSE\n(deg)")
axes[2].set_xlabel("Time (s)")
axes[0].set_title("C0 baseline: support state and joint tracking")
for ax in axes:
    ax.grid(True)
    ax.legend(loc="best")

fig.tight_layout()
out_support = FIG_DIR / "baseline_initial_pitch_support.png"
fig.savefig(out_support, dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {out_main}")
print(f"Saved: {out_support}")
