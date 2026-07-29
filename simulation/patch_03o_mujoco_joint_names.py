#!/usr/bin/env python3
"""
Patch Stage 7C MuJoCo replay script for the actual joint names in
model/biped_planar.xml.

FDDP/URDF names:
  left_hip_pitch, left_knee_pitch, ...

MuJoCo XML names:
  left_hip_pitch_joint, left_knee_pitch_joint, ...
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = PROJECT_ROOT / "simulation" / "03o_replay_fddp_com_shift_mujoco.py"
BACKUP = TARGET.with_suffix(TARGET.suffix + ".pre_joint_name_patch.bak")

OLD_BLOCK = """JOINT_ORDER = [
    "left_hip_pitch",
    "left_knee_pitch",
    "left_ankle_pitch",
    "right_hip_pitch",
    "right_knee_pitch",
    "right_ankle_pitch",
]"""

NEW_BLOCK = """JOINT_ORDER = [
    "left_hip_pitch_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
]"""


def main():
    if not TARGET.exists():
        raise FileNotFoundError(
            f"Replay script not found: {TARGET}\n"
            "Copy 03o_replay_fddp_com_shift_mujoco.py into simulation first."
        )

    text = TARGET.read_text(encoding="utf-8")

    if NEW_BLOCK in text:
        print("PATCH STATUS: already applied.")
        print(f"Target : {TARGET}")
        return

    if OLD_BLOCK not in text:
        raise RuntimeError(
            "Could not find the expected JOINT_ORDER block.\n"
            "The replay script may be a different version. Do not edit it blindly."
        )

    BACKUP.write_text(text, encoding="utf-8")
    TARGET.write_text(text.replace(OLD_BLOCK, NEW_BLOCK, 1), encoding="utf-8")

    print("PATCH STATUS: applied successfully.")
    print(f"Patched: {TARGET}")
    print(f"Backup : {BACKUP}")
    print("Updated MuJoCo joint lookup names:")
    for name in [
        "left_hip_pitch_joint",
        "left_knee_pitch_joint",
        "left_ankle_pitch_joint",
        "right_hip_pitch_joint",
        "right_knee_pitch_joint",
        "right_ankle_pitch_joint",
    ]:
        print(f"  - {name}")
    print("Next: re-run python simulation/03o_replay_fddp_com_shift_mujoco.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nJOINT NAME PATCH FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
