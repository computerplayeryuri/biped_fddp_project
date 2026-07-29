#!/usr/bin/env python3
"""
Fix Stage 7B exported q_ref dimensions.

The solver state is [q(13), v(12)].  For a free-flyer model:
  q[0:7]          = root position + quaternion
  q[7:model.nq]   = six actuated joint positions

The previous slice xs[:, 7:] mistakenly also exported all velocity states.
"""

from pathlib import Path
import shutil
import sys

TARGET_NAME = "02b_solve_double_support_fddp_stand.py"
BACKUP_SUFFIX = ".pre_qref_export_fix.bak"

OLD_LINE = "    q_ref = xs[:, 7:]"
NEW_LINE = "    q_ref = xs[:, 7:model.nq]"


def main():
    target = Path(__file__).resolve().parent / TARGET_NAME

    if not target.exists():
        raise FileNotFoundError(
            f"Could not find {target}. Put this patch file in the same fddp folder "
            f"as {TARGET_NAME} and run it again."
        )

    source = target.read_text(encoding="utf-8")

    if NEW_LINE in source:
        print("PATCH STATUS: q_ref export is already correct; no changes made.")
        return

    if OLD_LINE not in source:
        raise RuntimeError(
            "Could not find the expected q_ref export line. "
            "The Stage 7B script is not the expected revision; stop here and send "
            "the output of: grep -n \"q_ref =\" fddp/02b_solve_double_support_fddp_stand.py"
        )

    backup = target.with_name(target.name + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(target, backup)

    patched = source.replace(OLD_LINE, NEW_LINE, 1)
    target.write_text(patched, encoding="utf-8")

    print("PATCH STATUS: q_ref export fix applied successfully.")
    print(f"Patched: {target}")
    print(f"Backup : {backup}")
    print("Expected exported q_ref shape after re-running Stage 7B: [41, 6].")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"PATCH FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
