#!/usr/bin/env python3
"""
Patch Stage 7B FDDP stand solver for Crocoddyl builds where
ShootingProblem does not expose a .cost property.

This script:
  1) makes a timestamp-free backup beside the target script;
  2) inserts a helper that sums running and terminal data.cost values;
  3) replaces the two incompatible problem.cost reads;
  4) is safe to re-run: if already patched, it exits without changes.
"""

from pathlib import Path
import shutil
import sys

TARGET_NAME = "02b_solve_double_support_fddp_stand.py"
BACKUP_SUFFIX = ".pre_cost_compat.bak"

HELPER = r