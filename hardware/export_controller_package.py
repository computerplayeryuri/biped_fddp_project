#!/usr/bin/env python3
"""Export a hardware-side controller data package for C0/C1/C2.

The export is intended for a PC/ROS2-side hardware bridge. It contains the joint
order, safety limits, controller constants, FIR coefficients and time-indexed
joint references. ST3215 servos still receive position commands, not torques.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
TRAJ = ROOT / "transfer" / "trajectories"
LOG = ROOT / "results" / "logs"
CONFIG = ROOT / "config"
OUT = ROOT / "hardware" / "deploy"

JOINT_ORDER = [
    "left_hip_pitch_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
]
G = [0.27, 0.062, -0.228, 0.27, 0.062, -0.228]


def as_builtin(value):
    if isinstance(value, dict):
        return {k: as_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [as_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [as_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def export_reference(direction: str) -> dict:
    src = TRAJ / f"com_shift_{direction}_ref.npz"
    with np.load(src) as z:
        t = np.asarray(z["t_ref"], dtype=float)
        q = np.asarray(z["q_ref"], dtype=float)
        phase = np.asarray(z["phase"]).astype(str) if "phase" in z.files else np.asarray(["unknown"] * len(t))
        left = np.asarray(z["left_contact_active"], dtype=bool) if "left_contact_active" in z.files else np.ones(len(t), dtype=bool)
        right = np.asarray(z["right_contact_active"], dtype=bool) if "right_contact_active" in z.files else np.ones(len(t), dtype=bool)
    rows = {"time_s": t, "phase": phase, "left_contact_active": left.astype(int), "right_contact_active": right.astype(int)}
    for i, name in enumerate(JOINT_ORDER):
        rows[f"{name}_q_ref_rad"] = q[:, i]
    out = OUT / f"trajectory_{direction}_q_ref.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return {"source_npz": str(src), "export_csv": str(out), "samples": int(len(t)), "duration_s": float(t[-1])}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    robot_limits = yaml.safe_load((CONFIG / "robot_limits.yaml").read_text())
    actuator_model = yaml.safe_load((CONFIG / "actuator_model.yaml").read_text())
    fir = yaml.safe_load((LOG / "c1_qp1_fir_identification.yaml").read_text())
    c1_summary = yaml.safe_load((LOG / "c1_qp2_fir_mpc_tuning_summary.yaml").read_text())
    selected = c1_summary["selection"]["selected_candidate"]
    candidate = None
    for item in c1_summary["candidate_results"]:
        if item["candidate"]["name"] == selected:
            candidate = item["candidate"]
            break
    if candidate is None:
        raise RuntimeError("Selected C1 candidate not found in summary.")

    references = {direction: export_reference(direction) for direction in ("forward", "backward")}
    package = {
        "package_name": "biped_fddp_c0_c1_c2_hardware_controller_package",
        "generated_for": "PC/ROS2-side ST3215 joint-position command bridge",
        "command_type": "joint_position_rad",
        "torque_command_allowed": False,
        "control_update_period_s": 0.020,
        "joint_order": JOINT_ORDER,
        "joint_limits_rad": robot_limits["joint_limits_rad"],
        "actuator_model": actuator_model,
        "controllers": {
            "C0": {"enabled": True, "uses_reference": True, "uses_fir_qp": False, "uses_contact_phase_sync": False},
            "C1": {"enabled": True, "uses_reference": True, "uses_fir_qp": True, "uses_contact_phase_sync": False, "candidate": candidate, "G_rad_per_rad": G},
            "C2": {"enabled": True, "uses_reference": True, "uses_fir_qp": True, "uses_contact_phase_sync": True, "candidate": candidate, "G_rad_per_rad": G},
        },
        "fir_identification": {
            direction: {
                "coefficients_rad_per_rad": fir["directions"][direction]["fir_coefficients_rad_per_rad"],
                "source": str(LOG / "c1_qp1_fir_identification.yaml"),
            }
            for direction in ("forward", "backward")
        },
        "references": references,
        "runtime": {
            "python_module": "hardware.controller_runtime",
            "factory": "make_controller(controller_name, direction)",
            "manifest": str(ROOT / "hardware" / "controller_package.yaml"),
        },
        "safety_notes": [
            "Verify servo ID mapping and zero offsets before enabling motor power.",
            "Start with C0 at low speed; only enable C1/C2 after encoder, IMU and contact switch logs are stable.",
            "Use emergency stop and communication timeout in the hardware bridge.",
            "Do not send FDDP tau_ref to ST3215 servos; send q_cmd only.",
        ],
    }
    package_path = OUT / "controller_package.json"
    package_path.write_text(json.dumps(as_builtin(package), indent=2), encoding="utf-8")
    print(f"Exported hardware controller package: {package_path}")
    for direction, info in references.items():
        print(f"{direction}: {info['export_csv']}")


if __name__ == "__main__":
    main()
