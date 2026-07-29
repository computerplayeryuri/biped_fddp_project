from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from common import FIG_DIR, LOG_DIR, ensure_output_dirs

ensure_output_dirs()
path = LOG_DIR / "actuator_squat_test.csv"
if not path.exists():
    raise FileNotFoundError(
        f"Run 07a_actuator_squat_test.py first. Missing: {path}"
    )

df = pd.read_csv(path)
if df.empty:
    raise RuntimeError("Actuator squat log is empty.")

# The three left-leg traces are enough for this symmetric test.
fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.5), sharex=True)
labels = ["Hip", "Knee", "Ankle"]
for i, ax in enumerate(axes):
    ax.plot(df["time_s"], df[f"q_nom_{i}"], "--", label="nominal reference")
    ax.plot(df["time_s"], df[f"q_cmd_{i}"], label="filtered command")
    ax.plot(df["time_s"], df[f"q_{i}"], label="measured joint")
    ax.set_ylabel(f"{labels[i]} (rad)")
    ax.grid(True)
    ax.legend(loc="best")
axes[-1].set_xlabel("Time (s)")
fig.suptitle("Actuator-aware squat test: left-leg joint tracking")
fig.tight_layout()
out_tracking = FIG_DIR / "actuator_squat_joint_tracking.png"
fig.savefig(out_tracking, dpi=160, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True)
axes[0].plot(df["time_s"], df["torso_pitch_deg"])
axes[0].set_ylabel("Torso pitch (deg)")
axes[0].set_title("Actuator-aware squat test: body response")
axes[0].grid(True)
axes[1].plot(df["time_s"], df["joint_rmse_to_cmd_deg"])
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Joint RMSE to command (deg)")
axes[1].grid(True)
fig.tight_layout()
out_body = FIG_DIR / "actuator_squat_body_response.png"
fig.savefig(out_body, dpi=160, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 4.5))
for column in [
    "left_toe_switch",
    "left_heel_switch",
    "right_toe_switch",
    "right_heel_switch",
]:
    ax.step(df["time_s"], df[column], where="post", label=column)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Digital contact switch")
ax.set_ylim(-0.1, 1.1)
ax.set_title("Actuator-aware squat test: foot contacts")
ax.grid(True)
ax.legend(loc="best")
out_contact = FIG_DIR / "actuator_squat_contacts.png"
fig.savefig(out_contact, dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {out_tracking}")
print(f"Saved: {out_body}")
print(f"Saved: {out_contact}")
