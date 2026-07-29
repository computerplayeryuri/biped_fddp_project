#!/usr/bin/env python3
"""Natural toe/heel contact C2 phase-retiming validation for two-step FDDP.

This adapter consumes only the gated two-step nominal reference.  It will use
MuJoCo touch sensors for observed events; tau_ref_usage is analysis_only and
is never a position-actuator control command.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NPZ_PATH = ROOT / "transfer" / "trajectories" / "fddp_two_step_alternating_ref.npz"


def _as_bool(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind in ("U", "S", "O"):
        return np.asarray(
            [str(value).lower() in ("1", "true", "yes", "active") for value in values],
            dtype=bool,
        )
    return values.astype(float) > 0.5


def load_reference() -> dict:
    """Load the gated two-step nominal and explicit contact schedule."""
    with np.load(NPZ_PATH) as raw:
        required = (
            "t_ref", "q_ref", "phase", "left_contact_active",
            "right_contact_active", "tau_ref_hardware_usage",
        )
        missing = [name for name in required if name not in raw.files]
        if missing:
            raise RuntimeError(f"{NPZ_PATH} missing {missing}")
        reference = {
            "path": str(NPZ_PATH),
            "t_ref": np.asarray(raw["t_ref"], dtype=float).reshape(-1),
            "q_ref": np.asarray(raw["q_ref"], dtype=float),
            "phase": np.asarray(raw["phase"]).astype(str),
            "left_contact_active": _as_bool(raw["left_contact_active"]),
            "right_contact_active": _as_bool(raw["right_contact_active"]),
            "tau_ref_usage": str(np.asarray(raw["tau_ref_hardware_usage"]).item()),
        }
    size = reference["t_ref"].size
    if reference["q_ref"].shape != (size, 6):
        raise RuntimeError("two-step q_ref shape mismatch")
    for key in ("phase", "left_contact_active", "right_contact_active"):
        if np.asarray(reference[key]).size != size:
            raise RuntimeError(f"two-step {key} size mismatch")
    if reference["tau_ref_usage"] != "analysis_only":
        raise RuntimeError("tau_ref usage boundary is invalid")
    return reference


def build_reference_events(reference: dict) -> list[dict]:
    """Extract only liftoff/touchdown events from explicit per-leg schedule."""
    events: list[dict] = []
    for leg in ("left", "right"):
        active = np.asarray(reference[f"{leg}_contact_active"], dtype=bool)
        for index in range(1, active.size):
            if active[index] == active[index - 1]:
                continue
            events.append(
                {
                    "time_s": float(reference["t_ref"][index]),
                    "kind": f"{leg}_{'touchdown' if active[index] else 'liftoff'}",
                    "index": int(index),
                    "source": f"{leg}_contact_active",
                }
            )
    return sorted(events, key=lambda event: event["time_s"])


import importlib.util
import sys

import matplotlib.pyplot as plt
import mujoco
import pandas as pd
import yaml

XML_PATH = ROOT / "model" / "biped_planar.xml"
LOG_DIR = ROOT / "results" / "logs"
FIG_DIR = ROOT / "results" / "figures"
CSV_PATH = LOG_DIR / "fddp_two_step_natural_contact_c2.csv"
YAML_PATH = LOG_DIR / "fddp_two_step_natural_contact_c2.yaml"
FIG_PATH = FIG_DIR / "fddp_two_step_natural_contact_c2.png"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_natural_contact_c2() -> dict:
    """Replay the two-step nominal with natural toe/heel event retiming."""
    c2 = _load_module(
        ROOT / "mpc" / "08_run_contact_phase_synchronised_fir_qp_mpc.py",
        "fddp_two_step_c2_common",
    )
    proxy = _load_module(
        ROOT / "simulation" / "03o_replay_fddp_com_shift_mujoco.py",
        "fddp_two_step_c2_proxy",
    )
    reference = load_reference()
    scheduled = build_reference_events(reference)
    t_ref = reference["t_ref"]
    q_ref = reference["q_ref"]
    duration = float(t_ref[-1])

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    mapping = proxy.resolve_joint_and_actuator_mapping(model)
    torso = proxy.resolve_torso_body(model)
    mujoco.mj_resetData(model, data)
    for address, value in zip(mapping["qpos_addresses"], q_ref[0]):
        data.qpos[address] = value
    mujoco.mj_forward(model, data)

    observer = c2.ContactObserver(model, data, float(model.opt.timestep))
    synchronizer = c2.ContactPhaseSynchronizer(scheduled)
    actuator = proxy.PositionActuatorProxy(6, float(model.opt.timestep))
    actuator.reset(q_ref[0])
    rows: list[dict] = []
    event_records: list[dict] = []
    observed_count = 0
    initial_height = float(data.xpos[torso][2])

    while data.time <= duration:
        wall_time = min(float(data.time), duration)
        contact_values, site_active, leg_active, observed = observer.update(
            model, data, wall_time
        )
        observed_count += len(observed)
        event_records.extend(synchronizer.update(wall_time, observed))
        phase_time = synchronizer.reference_time(wall_time, duration)
        q_command = proxy.interpolate_reference(phase_time, t_ref, q_ref)
        q_proxy = actuator.update(q_command)
        data.ctrl[:] = 0.0
        for actuator_id, value in zip(mapping["actuator_ids"], q_proxy):
            data.ctrl[actuator_id] = value
        mujoco.mj_step(model, data)
        row = {
            "wall_time_s": float(data.time),
            "reference_time_s": synchronizer.reference_time(
                min(float(data.time), duration), duration
            ),
            "phase_offset_s": float(synchronizer.offset_s),
            "ncon": int(data.ncon),
            "torso_height_m": float(data.xpos[torso][2]),
            "left_contact_active": bool(leg_active["left"]),
            "right_contact_active": bool(leg_active["right"]),
            "matched_contact_events": int(len(synchronizer.matched)),
            "skipped_contact_events": int(len(synchronizer.skipped)),
        }
        for key, value in contact_values.items():
            row[key] = float(value)
        for key, value in site_active.items():
            row[f"{key}_active"] = bool(value)
        rows.append(row)

    frame = pd.DataFrame(rows)
    matched_count = len(synchronizer.matched)
    metrics = {
        "scheduled_contact_events": len(scheduled),
        "natural_observed_contact_events": int(observed_count),
        "matched_contact_events": int(matched_count),
        "skipped_contact_events": int(len(synchronizer.skipped)),
        "max_abs_phase_offset_ms": float(
            1000.0 * np.max(np.abs(frame["phase_offset_s"].to_numpy()))
        ),
        "min_global_contact_count": int(frame["ncon"].min()),
        "min_torso_height_m": float(frame["torso_height_m"].min()),
        "initial_torso_height_m": initial_height,
    }
    integrity_passed = bool(
        metrics["scheduled_contact_events"] == 4
        and metrics["natural_observed_contact_events"] > 0
        and metrics["matched_contact_events"] > 0
        and metrics["min_global_contact_count"] > 0
    )
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CSV_PATH, index=False)
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(
        frame["wall_time_s"], 1000.0 * frame["phase_offset_s"],
        label="C2 phase offset (ms)",
    )
    axes[0].grid(True); axes[0].legend()
    for name in ("left_toe", "left_heel", "right_toe", "right_heel"):
        axes[1].plot(
            frame["wall_time_s"], frame[f"{name}_force_N"], label=name
        )
    axes[1].set_xlabel("wall time (s)")
    axes[1].set_ylabel("touch force (N)")
    axes[1].grid(True); axes[1].legend(ncol=2)
    figure.tight_layout(); figure.savefig(FIG_PATH, dpi=180); plt.close(figure)
    summary = {
        "stage": "two_step_fddp_natural_toe_heel_c2",
        "purpose": (
            "MuJoCo natural toe/heel contact-event C2 phase retiming around "
            "two-step FDDP position-command replay, not hardware walking."
        ),
        "inputs": {
            "reference_npz": str(NPZ_PATH),
            "mujoco_xml": str(XML_PATH),
            "tau_ref_usage": "analysis_only_never_sent_to_ctrl",
        },
        "metrics": metrics,
        "scheduled_events": scheduled,
        "event_records": event_records,
        "integrity_passed": integrity_passed,
        "outputs": {"csv": str(CSV_PATH), "figure": str(FIG_PATH)},
    }
    YAML_PATH.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(summary, sort_keys=False))
    return summary


def main() -> int:
    return 0 if run_natural_contact_c2()["integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
