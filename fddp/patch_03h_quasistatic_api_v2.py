#!/usr/bin/env python3
"""Patch Stage 7C dynamic-seed audit for Crocoddyl quasiStatic(xs) bindings.

The installed binding accepts only ShootingProblem.quasiStatic(xs) and returns
controls. This patch updates the existing audit script accordingly and replaces
its rollout helper with a binding-independent explicit dynamics rollout.
"""
from pathlib import Path
import re
import shutil
import sys

TARGET_NAME = "03h_audit_fddp_dynamic_initialisation.py"
BACKUP_SUFFIX = ".pre_quasistatic_api_v2.bak"

OLD_QUASI = '''    us_quasi = [np.zeros(nu, dtype=float) for _ in range(HORIZON_NODES)]
    try:
        problem.quasiStatic(us_quasi, xs_seed)
    except Exception as exc:
        return {
            "contact_gains_Kp_Kd": [float(v) for v in gains],
            "zero_control": {
                "cost": float(zero_cost),
                "shooting_gaps": zero_gaps,
            },
            "quasi_static": {
                "available": False,
                "error": str(exc),
            },
        }

    us_quasi = [np.asarray(u, dtype=float).copy() for u in us_quasi]
'''

NEW_QUASI = '''    try:
        # This Crocoddyl Python binding exposes quasiStatic(xs) and returns the
        # T control vectors directly. Different builds accept either T states
        # (running nodes only) or T+1 states, so try the standard T-state call
        # first and retain the successful mode in the report.
        try:
            returned_controls = problem.quasiStatic(
                [np.asarray(x, dtype=float).copy() for x in xs_seed[:-1]]
            )
            quasi_static_state_list_mode = "running_nodes_only"
        except Exception:
            returned_controls = problem.quasiStatic(
                [np.asarray(x, dtype=float).copy() for x in xs_seed]
            )
            quasi_static_state_list_mode = "full_state_sequence"

        us_quasi = [
            np.asarray(u, dtype=float).reshape(-1).copy()
            for u in returned_controls
        ]
        if len(us_quasi) != HORIZON_NODES:
            raise RuntimeError(
                f"quasiStatic returned {len(us_quasi)} controls; "
                f"expected {HORIZON_NODES}."
            )
        if any(u.shape != (nu,) for u in us_quasi):
            raise RuntimeError(
                "quasiStatic returned a control vector with an unexpected shape."
            )
    except Exception as exc:
        return {
            "contact_gains_Kp_Kd": [float(v) for v in gains],
            "zero_control": {
                "cost": float(zero_cost),
                "shooting_gaps": zero_gaps,
            },
            "quasi_static": {
                "available": False,
                "error": str(exc),
            },
        }
'''

OLD_ROLLOUT_RE = r'''def rollout_controls\(problem, us\):\n    """Use whichever rollout binding is present in the installed Crocoddyl build\."""\n    try:\n        output = problem\.rollout_us\(us\)\n        return \[np\.asarray\(x, dtype=float\)\.copy\(\) for x in output\]\n    except AttributeError:\n        xs = \[np\.asarray\(problem\.x0, dtype=float\)\.copy\(\) for _ in range\(HORIZON_NODES \+ 1\)\]\n        problem\.rollout\(us, xs\)\n        return \[np\.asarray\(x, dtype=float\)\.copy\(\) for x in xs\]\n\n'''

NEW_ROLLOUT = '''def rollout_controls(problem, us):
    """Binding-independent forward rollout through Crocoddyl running models."""
    xs = [np.asarray(problem.x0, dtype=float).copy()]
    for k, u in enumerate(us):
        running_model = problem.runningModels[k]
        running_data = problem.runningDatas[k]
        running_model.calc(running_data, xs[-1], u)
        xs.append(np.asarray(running_data.xnext, dtype=float).copy())
    return xs

'''


def main():
    target = Path(__file__).resolve().parent / TARGET_NAME
    if not target.exists():
        raise FileNotFoundError(
            f"Could not find {target}. Put this patch in the same fddp folder as {TARGET_NAME}."
        )

    source = target.read_text(encoding="utf-8")
    if "quasi_static_state_list_mode" in source:
        print("PATCH STATUS: quasiStatic API fix already applied; no changes made.")
        return

    if OLD_QUASI not in source:
        raise RuntimeError(
            "Could not find the expected quasiStatic block. "
            "Do not edit manually; send the output of: "
            "nl -ba fddp/03h_audit_fddp_dynamic_initialisation.py | sed -n '220,330p'"
        )

    patched = source.replace(OLD_QUASI, NEW_QUASI, 1)
    patched, replacement_count = re.subn(OLD_ROLLOUT_RE, NEW_ROLLOUT, patched, count=1)
    if replacement_count != 1:
        raise RuntimeError(
            "Could not replace the rollout helper. The source revision differs; "
            "do not edit manually."
        )

    backup = target.with_name(target.name + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(target, backup)
    target.write_text(patched, encoding="utf-8")

    print("PATCH STATUS: applied successfully.")
    print(f"Patched: {target}")
    print(f"Backup : {backup}")
    print("Next: re-run python fddp/03h_audit_fddp_dynamic_initialisation.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"PATCH FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
