#!/usr/bin/env python3
"""
C2 MuJoCo contact-switching mechanism demo.

This is a mechanism validation for contact-event phase synchronisation. It uses
an intentionally conservative scripted joint-position reference that lifts and
replaces the left foot, so the MuJoCo toe/heel touch sensors produce real
liftoff/touchdown events. It is not an FDDP walking trajectory and should not be
presented as walking validation.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
C2_PATH = ROOT / "mpc" / "08_run_contact_phase_synchronised_fir_qp_mpc.py"
XML = ROOT / "model" / "biped_planar.xml"
LOG = ROOT / "results" / "logs"
FIG = ROOT / "results" / "figures"

JOINTS = [
    "left_hip_pitch_joint", "left_knee_pitch_joint", "left_ankle_pitch_joint",
    "right_hip_pitch_joint", "right_knee_pitch_joint", "right_ankle_pitch_joint",
]
TORSO = "hip_plate_link"

DURATION_S = 1.35
LIFT_START_S = 0.25
LIFT_FULL_S = 0.45
LOWER_START_S = 0.78
TOUCHDOWN_REF_S = 1.00
LIFTOFF_EVENT_REF_S = 0.36
DT_LOGIC_S = 0.002
TOUCH_THRESHOLD_N = 0.25

# Same actuator proxy assumptions used in C0/C1/C2 validation.
COMMAND_DELAY_S = 0.020
ACTUATOR_TAU_S = 0.030
MAX_COMMAND_SPEED_RAD_S = 8.0

EXPECTED_EVENTS = [
    {"time_s": LIFTOFF_EVENT_REF_S, "kind": "left_liftoff", "source": "scripted_reference"},
    {"time_s": TOUCHDOWN_REF_S, "kind": "left_touchdown", "source": "scripted_reference"},
]


def load_c2_module():
    spec = importlib.util.spec_from_file_location("c2_phase_sync", C2_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def smoothstep(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def lift_profile(t):
    if t < LIFT_START_S:
        return 0.0
    if t < LIFT_FULL_S:
        return smoothstep((t - LIFT_START_S) / (LIFT_FULL_S - LIFT_START_S))
    if t < LOWER_START_S:
        return 1.0
    if t < TOUCHDOWN_REF_S:
        return 1.0 - smoothstep((t - LOWER_START_S) / (TOUCHDOWN_REF_S - LOWER_START_S))
    return 0.0


def q_reference(t):
    s = lift_profile(t)
    q = np.zeros(6, dtype=float)
    # Left foot lift: hip forward, knee flexed, ankle compensates toward a flatter foot.
    q[0] = math.radians(25.0) * s
    q[1] = math.radians(-54.0) * s
    q[2] = math.radians(25.0) * s
    return q


class ActuatorProxy:
    def __init__(self, n, dt):
        self.n = int(n)
        self.dt = float(dt)
        self.delay = int(round(COMMAND_DELAY_S / self.dt))
        self.hist = deque(maxlen=self.delay + 1)
        self.out = np.zeros(self.n)

    def reset(self, q):
        q = np.asarray(q, dtype=float).reshape(self.n)
        self.hist.clear()
        for _ in range(self.delay + 1):
            self.hist.append(q.copy())
        self.out = q.copy()

    def update(self, q):
        q = np.asarray(q, dtype=float).reshape(self.n)
        self.hist.append(q.copy())
        delayed = self.hist[0]
        gain = min(1.0, self.dt / ACTUATOR_TAU_S)
        delta = gain * (delayed - self.out)
        limit = MAX_COMMAND_SPEED_RAD_S * self.dt
        self.out += np.clip(delta, -limit, limit)
        return self.out.copy()


def require_id(model, kind, name):
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise RuntimeError(f"Required MuJoCo name not found: {name}")
    return int(value)


def map_actuators(model, joint_ids):
    result = []
    for jid in joint_ids:
        candidates = [i for i in range(model.nu) if int(model.actuator_trnid[i, 0]) == jid]
        if len(candidates) != 1:
            raise RuntimeError(f"Joint actuator mapping failed for {jid}: {candidates}")
        result.append(candidates[0])
    return result


def simulate(enable_phase_sync=True):
    c2 = load_c2_module()
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)

    joint_ids = [require_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINTS]
    qadr = [int(model.jnt_qposadr[jid]) for jid in joint_ids]
    act_ids = map_actuators(model, joint_ids)
    torso_id = require_id(model, mujoco.mjtObj.mjOBJ_BODY, TORSO)

    mujoco.mj_resetData(model, data)
    for addr, q0 in zip(qadr, q_reference(0.0)):
        data.qpos[addr] = q0
    mujoco.mj_forward(model, data)

    proxy = ActuatorProxy(6, dt)
    proxy.reset(q_reference(0.0))
    observer = c2.ContactObserver(model, data, dt)
    sync = c2.ContactPhaseSynchronizer([dict(event) for event in EXPECTED_EVENTS])

    rows = []
    event_records = []
    observed_events_flat = []
    nsteps = int(round(DURATION_S / dt)) + 1

    for _ in range(nsteps):
        wall_t = min(float(data.time), DURATION_S)
        contact_values, site_active, leg_active, observed_events = observer.update(model, data, wall_t)
        if observed_events:
            observed_events_flat.extend(observed_events)
        sync_records = sync.update(wall_t, observed_events) if enable_phase_sync else []
        event_records.extend(sync_records)
        reference_time = sync.reference_time(wall_t, DURATION_S) if enable_phase_sync else wall_t

        q_ref = q_reference(reference_time)
        q_proxy = proxy.update(q_ref)
        data.ctrl[:] = 0.0
        for aid, command in zip(act_ids, q_proxy):
            data.ctrl[aid] = command
        mujoco.mj_step(model, data)

        row = {
            "time_s": float(data.time),
            "reference_time_s": float(reference_time),
            "phase_offset_s": float(sync.offset_s if enable_phase_sync else 0.0),
            "enable_phase_sync": bool(enable_phase_sync),
            "left_contact_active": bool(leg_active["left"]),
            "right_contact_active": bool(leg_active["right"]),
            "ncon": int(data.ncon),
            "torso_height_m": float(data.xpos[torso_id][2]),
            "matched_events": int(len(sync.matched)),
            "skipped_events": int(len(sync.skipped)),
        }
        for name, value in contact_values.items():
            row[name] = float(value)
        for name, active in site_active.items():
            row[f"{name}_active"] = bool(active)
        q_meas = [float(data.qpos[addr]) for addr in qadr]
        for index, name in enumerate(JOINTS):
            row[f"{name}_ref_rad"] = float(q_ref[index])
            row[f"{name}_meas_rad"] = q_meas[index]
            row[f"{name}_cmd_rad"] = float(q_proxy[index])
        rows.append(row)

    return pd.DataFrame(rows), sync, event_records, observed_events_flat


def event_time(events, kind):
    for event in events:
        if event.get("kind") == kind:
            return float(event.get("time_s"))
    return None


def plot_demo(fixed_df, c2_df):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    for prefix, label in (("left_toe", "left toe"), ("left_heel", "left heel"), ("right_toe", "right toe"), ("right_heel", "right heel")):
        ax.plot(c2_df.time_s, c2_df[f"{prefix}_force_N"], label=label)
    for event in EXPECTED_EVENTS:
        ax.axvline(event["time_s"], linestyle="--", linewidth=1.0, label=f"expected {event['kind']}")
    ax.axhline(TOUCH_THRESHOLD_N, linestyle=":", color="black", label="touch threshold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Touch sensor force (N)")
    ax.set_title("C2 contact-switching demo touch sensors")
    ax.grid(True)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    contacts_path = FIG / "c2_contact_switching_demo_contacts.png"
    fig.savefig(contacts_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(c2_df.time_s, 1000.0 * c2_df.phase_offset_s, label="C2 phase offset")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Phase offset (ms)")
    ax.set_title("C2 contact-switching demo phase correction")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    phase_path = FIG / "c2_contact_switching_demo_phase_offset.png"
    fig.savefig(phase_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(fixed_df.time_s, np.rad2deg(fixed_df.left_knee_pitch_joint_ref_rad), label="fixed-clock left knee ref")
    ax.plot(c2_df.time_s, np.rad2deg(c2_df.left_knee_pitch_joint_ref_rad), label="C2 left knee ref")
    ax.plot(c2_df.time_s, np.rad2deg(c2_df.left_knee_pitch_joint_meas_rad), label="C2 left knee measured")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Left knee angle (deg)")
    ax.set_title("C2 contact-switching demo reference retiming")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    joint_path = FIG / "c2_contact_switching_demo_joint_retiming.png"
    fig.savefig(joint_path, dpi=180)
    plt.close(fig)
    return contacts_path, phase_path, joint_path


def main():
    LOG.mkdir(parents=True, exist_ok=True)
    fixed_df, fixed_sync, fixed_records, fixed_observed = simulate(enable_phase_sync=False)
    c2_df, c2_sync, c2_records, c2_observed = simulate(enable_phase_sync=True)

    fixed_csv = LOG / "c2_contact_switching_demo_fixed_clock.csv"
    c2_csv = LOG / "c2_contact_switching_demo_phase_sync.csv"
    fixed_df.to_csv(fixed_csv, index=False)
    c2_df.to_csv(c2_csv, index=False)
    contacts_path, phase_path, joint_path = plot_demo(fixed_df, c2_df)

    observed_liftoff = event_time(c2_observed, "left_liftoff")
    observed_touchdown = event_time(c2_observed, "left_touchdown")
    matched = len(c2_sync.matched)
    summary = {
        "stage": "C2_mujoco_contact_switching_mechanism_demo",
        "scope_note": "Scripted joint-position contact-switching mechanism validation only; not FDDP walking, not hardware validation.",
        "scripted_reference": {
            "duration_s": DURATION_S,
            "lift_start_s": LIFT_START_S,
            "lift_full_s": LIFT_FULL_S,
            "lower_start_s": LOWER_START_S,
            "liftoff_event_reference_s": LIFTOFF_EVENT_REF_S,
            "touchdown_reference_s": TOUCHDOWN_REF_S,
            "maximum_left_hip_deg": 25.0,
            "maximum_left_knee_deg": -54.0,
            "maximum_left_ankle_deg": 25.0,
        },
        "expected_events": EXPECTED_EVENTS,
        "observed_events": c2_observed,
        "matched_event_records": c2_sync.matched,
        "skipped_event_records": c2_sync.skipped,
        "metrics": {
            "observed_left_liftoff_s": observed_liftoff,
            "observed_left_touchdown_s": observed_touchdown,
            "matched_events": int(matched),
            "scheduled_events": int(len(EXPECTED_EVENTS)),
            "max_abs_phase_offset_ms": float(1000.0 * np.max(np.abs(c2_df.phase_offset_s.to_numpy()))),
            "final_phase_offset_ms": float(1000.0 * c2_df.phase_offset_s.iloc[-1]),
            "min_contacts": int(c2_df.ncon.min()),
            "min_torso_height_m": float(c2_df.torso_height_m.min()),
            "left_contact_was_lost": bool((~c2_df.left_contact_active.astype(bool)).any()),
            "left_contact_was_restored": bool(c2_df.left_contact_active.astype(bool).iloc[-1]),
        },
        "outputs": {
            "fixed_clock_csv": str(fixed_csv),
            "phase_sync_csv": str(c2_csv),
            "contacts_plot": str(contacts_path),
            "phase_offset_plot": str(phase_path),
            "joint_retiming_plot": str(joint_path),
        },
    }
    summary["acceptance"] = {
        "left_liftoff_detected": observed_liftoff is not None,
        "left_touchdown_detected": observed_touchdown is not None,
        "all_expected_events_matched": matched == len(EXPECTED_EVENTS),
        "phase_offset_bound_respected": summary["metrics"]["max_abs_phase_offset_ms"] <= 60.0 + 1e-9,
        "support_contact_retained": summary["metrics"]["min_contacts"] >= 1,
    }
    summary_path = LOG / "c2_contact_switching_demo_summary.yaml"
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")

    print("C2 MuJoCo contact-switching mechanism demo")
    print(f"Observed liftoff: {observed_liftoff}")
    print(f"Observed touchdown: {observed_touchdown}")
    print(f"Matched events: {matched}/{len(EXPECTED_EVENTS)}")
    print(f"Max abs phase offset: {summary['metrics']['max_abs_phase_offset_ms']:.3f} ms")
    print(f"Saved summary: {summary_path}")
    if not all(summary["acceptance"].values()):
        print(summary["acceptance"])
        raise SystemExit("C2 contact-switching mechanism demo failed acceptance checks.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"C2 CONTACT-SWITCHING DEMO FAILED: {exc}", file=sys.stderr)
        sys.exit(1)



