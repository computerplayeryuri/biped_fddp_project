from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import FIG_DIR, LOG_DIR, ensure_output_dirs

ensure_output_dirs()
path = LOG_DIR / "standing_position_control.csv"
if not path.exists():
    raise FileNotFoundError(
        f"Run 05_standing_position_control.py first. Missing: {path}"
    )

df = pd.read_csv(path)
if df.empty:
    raise RuntimeError("Standing log is empty.")

# Figure 1: torso pitch.
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(df["time_s"], df["torso_pitch_deg"])
ax.set_xlabel("Time (s)")
ax.set_ylabel("Torso pitch (deg)")
ax.set_title("Standing position-control: torso pitch")
ax.grid(True)
out_pitch = FIG_DIR / "standing_torso_pitch.png"
fig.savefig(out_pitch, dpi=160, bbox_inches="tight")
plt.close(fig)

# Figure 2: toe/heel digital switches. 0 = no contact, 1 = contact.
fig, ax = plt.subplots(figsize=(8, 4.5))
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
ax.set_title("Standing position-control: toe/heel contact switches")
ax.grid(True)
ax.legend(loc="best")
out_contact = FIG_DIR / "standing_contact_switches.png"
fig.savefig(out_contact, dpi=160, bbox_inches="tight")
plt.close(fig)

# Figure 3: actual joint angle tracking versus the zero reference.
fig, ax = plt.subplots(figsize=(8, 4.5))
for i in range(6):
    ax.plot(df["time_s"], df[f"q_{i}"], label=f"q_{i}")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Joint angle (rad)")
ax.set_title("Standing position-control: measured joint angles")
ax.grid(True)
ax.legend(loc="best", ncol=2)
out_joint = FIG_DIR / "standing_joint_angles.png"
fig.savefig(out_joint, dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {out_pitch}")
print(f"Saved: {out_contact}")
print(f"Saved: {out_joint}")
