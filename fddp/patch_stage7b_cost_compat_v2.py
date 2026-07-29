#!/usr/bin/env python3
"""Patch the Stage 7B FDDP solver for Crocoddyl versions without problem.cost."""

from pathlib import Path
import shutil
import sys

TARGET_NAME = "02b_solve_double_support_fddp_stand.py"
BACKUP_SUFFIX = ".pre_cost_compat.bak"

OLD_INITIAL = """    problem.calc(xs_init, us_init)\n    initial_cost = float(problem.cost)\n"""
NEW_INITIAL = """    problem.calc(xs_init, us_init)\n    initial_cost = (\n        sum(float(data.cost) for data in problem.runningDatas)\n        + float(problem.terminalData.cost)\n    )\n"""

OLD_FINAL = """    problem.calc([x.copy() for x in xs], [u.copy() for u in us])\n    final_cost = float(problem.cost)\n"""
NEW_FINAL = """    problem.calc([x.copy() for x in xs], [u.copy() for u in us])\n    final_cost = (\n        sum(float(data.cost) for data in problem.runningDatas)\n        + float(problem.terminalData.cost)\n    )\n"""


def main() -> None:
    fddp_dir = Path(__file__).resolve().parent
    target = fddp_dir / TARGET_NAME

    if not target.exists():
        raise FileNotFoundError(
            f"Could not find {target}. Copy this patch script into the same fddp folder "
            f"as {TARGET_NAME}, then run it again."
        )

    source = target.read_text(encoding="utf-8")

    if "sum(float(data.cost) for data in problem.runningDatas)" in source:
        print("PATCH STATUS: already applied; no changes made.")
        return

    if OLD_INITIAL not in source:
        raise RuntimeError(
            "Could not find the expected initial-cost block. "
            "The Stage 7B solver revision differs from the expected file."
        )

    if OLD_FINAL not in source:
        raise RuntimeError(
            "Could not find the expected final-cost block. "
            "The Stage 7B solver revision differs from the expected file."
        )

    backup = target.with_name(target.name + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(target, backup)

    patched = source.replace(OLD_INITIAL, NEW_INITIAL, 1)
    patched = patched.replace(OLD_FINAL, NEW_FINAL, 1)
    target.write_text(patched, encoding="utf-8")

    print("PATCH STATUS: applied successfully.")
    print(f"Patched: {target}")
    print(f"Backup : {backup}")
    print("Next: run python fddp/02b_solve_double_support_fddp_stand.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"PATCH FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
