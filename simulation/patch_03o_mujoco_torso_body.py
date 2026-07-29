#!/usr/bin/env python3
"""
Patch Stage 7C MuJoCo replay script to use hip_plate_link as the torso body.

The current MuJoCo model uses:
    hip_plate_link
as the rigid body carrying the floating-base pose. It is the appropriate
body for torso height and pitch measurements in the replay script.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = PROJECT_ROOT / "simulation" / "03o_replay_fddp_com_shift_mujoco.py"
BACKUP = TARGET.with_suffix(TARGET.suffix + ".pre_torso_body_patch.bak")

OLD_BLOCK = """TORSO_BODY_CANDIDATES = [
    "torso",
    "pelvis",
    "base",
    "body",
    "top_electronics",
    "hip_bridge",
]"""

NEW_BLOCK = """TORSO_BODY_CANDIDATES = [
    "hip_plate_link",
    "electronics_lump_link",
    "torso",
    "pelvis",
    "base",
    "body",
    "top_electronics",
    "hip_bridge",
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
            "Could not find the expected TORSO_BODY_CANDIDATES block.\n"
            "The replay script may be a different version. Do not edit it blindly."
        )

    BACKUP.write_text(text, encoding="utf-8")
    TARGET.write_text(text.replace(OLD_BLOCK, NEW_BLOCK, 1), encoding="utf-8")

    print("PATCH STATUS: applied successfully.")
    print(f"Patched: {TARGET}")
    print(f"Backup : {BACKUP}")
    print("Replay body metrics will now use: hip_plate_link")
    print("Next: re-run python simulation/03o_replay_fddp_com_shift_mujoco.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nTORSO BODY PATCH FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
