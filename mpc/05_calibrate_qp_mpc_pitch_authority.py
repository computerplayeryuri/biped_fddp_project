#!/usr/bin/env python3
"""
C1-QP0 — MuJoCo pitch-authority and reference-alignment calibration.

Why this stage is necessary
----------------------------
The first C1-QP trial showed stable execution and sub-millisecond QP solve
times, but C1 worsened final pitch tracking. That result means the controller
must not yet be considered validated.

The two quantities that must be measured in MuJoCo, rather than assumed from
the Pinocchio/FDDP model, are:
  1) the fixed-phase nominal root-pitch trajectory realised by the C0 MuJoCo
     replay through the actuator proxy;
  2) the signed pitch response caused by the FDDP-guided symmetric six-joint
     correction distribution G.

This script:
  - runs a C0 nominal MuJoCo baseline;
  - applies a positive and a negative bounded correction pulse at t=0.40 s;
  - measures the root-pitch response relative to the C0 MuJoCo baseline;
  - estimates signed pitch authority (rad pitch / rad alpha);
  - writes values needed by the reference-aligned C1-QP v2 controller.

No disturbance is injected in this script. It is a system-identification
calibration, not a performance comparison.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import argparse
import math
import sys

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
XML_PATH = ROOT / "model" / "biped_planar.xml"
TRAJ_DIR = ROOT / "transfer" / "trajectories"
LOG_DIR = ROOT / "results" / "logs"
FIG_DIR = ROOT / "results" / "figures"

JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
]
ROOT_PITCH_JOINT = "root_pitch_x"

COMMAND_DELAY_S = 0.020
ACTUATOR_TIME_CONSTANT_S = 0.030
MAX_COMMAND_SPEED_RAD_S = 8.0

# FDDP-guided symmetric distribution from the successful C1-A local replan.
G = np.array([0.27, 0.062, -0.228, 0.27, 0.062, -0.228], dtype=float)

PULSE_TIME_S = 0.400
PULSE_DURATION_S = 0.100
PULSE_ALPHA_DEG = 2.0

# Probe response times after the pulse begins.
PROBE_DELAYS_S = [0.040, 0.080, 0.120, 0.160, 0.240]


class PositionActuatorProxy:
    def __init__(self, count: int, dt: float):
        self.count = int(count)
        self.dt = float(dt)
        self.delay_steps = int(round(COMMAND_DELAY_S / self.dt))
        self.history = deque(maxlen=self.delay_steps + 1)
        self.output = np.zeros(self.count, dtype=float)

    def reset(self, q0: np.ndarray) -> None:
        q0 = np.asarray(q0, dtype=float).reshape(self.count)
        self.history.clear()
        for _ in range(self.delay_steps + 1):
            self.history.append(q0.copy())
        self.output = q0.copy()

    def update(self, command: np.ndarray) -> np.ndarray:
        command = np.asarray(command, dtype=float).reshape(self.count)
        self.history.append(command.copy())
        delayed = self.history[0]
        alpha = min(1.0, self.dt / ACTUATOR_TIME_CONSTANT_S)
        step = alpha * (delayed - self.output)
        limit = MAX_COMMAND_SPEED_RAD_S * self.dt
        step = np.clip(step, -limit, limit)
        self.output = self.output + step
        return self.output.copy()


def require_name(model, object_type, name: str) -> int:
    value = mujoco.mj_name2id(model, object_type, name)
    if value < 0:
        values = [
            mujoco.mj_id2name(model, object_type, idx)
            for idx in range(
                model.njnt if object_type == mujoco.mjtObj.mjOBJ_JOINT else model.nbody
            )
        ]
        raise RuntimeError(f"Required object '{name}' not found. Available: {values}")
    return int(value)


def actuator_ids_for_joints(model, joint_ids: list[int]) -> list[int]:
    ids = []
    for joint_id in joint_ids:
        candidates = [
            index for index in range(model.nu)
            if int(model.actuator_trnid[index, 0]) == joint_id
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Cannot map joint id {joint_id} to exactly one actuator; got {candidates}."
            )
        ids.append(candidates[0])
    return ids


def interpolate(t: float, t_ref: np.ndarray, values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return np.asarray(np.interp(t, t_ref, values), dtype=float)
    return np.asarray(
        [np.interp(t, t_ref, values[:, k]) for k in range(values.shape[1])],
        dtype=float,
    )


def load_reference(direction: str):
    path = TRAJ_DIR / f"com_shift_{direction}_ref.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing C0 reference: {path}")
    with np.load(path) as archive:
        t_ref = np.asarray(archive["t_ref"], dtype=float).reshape(-1)
        q_ref = np.asarray(archive["q_ref"], dtype=float)
    if q_ref.shape != (len(t_ref), 6):
        raise RuntimeError(f"Unexpected q_ref shape: {q_ref.shape}")
    return path, t_ref, q_ref


def run_replay(direction: str, alpha_pulse_rad: float):
    reference_path, t_ref, q_ref = load_reference(direction)

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)

    joint_ids = [
        require_name(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES
    ]
    qpos_addresses = [int(model.jnt_qposadr[item]) for item in joint_ids]
    actuator_ids = actuator_ids_for_joints(model, joint_ids)

    root_pitch_joint_id = require_name(
        model, mujoco.mjtObj.mjOBJ_JOINT, ROOT_PITCH_JOINT
    )
    root_pitch_qpos_address = int(model.jnt_qposadr[root_pitch_joint_id])
    root_pitch_dof_address = int(model.jnt_dofadr[root_pitch_joint_id])

    mujoco.mj_resetData(model, data)
    for address, value in zip(qpos_addresses, q_ref[0]):
        data.qpos[address] = value
    mujoco.mj_forward(model, data)

    proxy = PositionActuatorProxy(6, dt)
    proxy.reset(q_ref[0])

    rows = []
    duration = float(t_ref[-1])
    steps = int(round(duration / dt)) + 1
    pulse_end = PULSE_TIME_S + PULSE_DURATION_S

    for _ in range(steps):
        t = min(float(data.time), duration)
        q_nominal = interpolate(t, t_ref, q_ref)

        active = PULSE_TIME_S <= t < pulse_end
        delta_q = G * alpha_pulse_rad if active else np.zeros(6, dtype=float)
        q_command = q_nominal + delta_q
        q_proxy = proxy.update(q_command)

        data.ctrl[:] = 0.0
        for actuator_id, value in zip(actuator_ids, q_proxy):
            data.ctrl[actuator_id] = value
        mujoco.mj_step(model, data)

        rows.append({
            "time_s": float(data.time),
            "root_pitch_rad": float(data.qpos[root_pitch_qpos_address]),
            "root_pitch_rate_rad_s": float(data.qvel[root_pitch_dof_address]),
            "pulse_active": int(active),
            "alpha_pulse_rad": float(alpha_pulse_rad if active else 0.0),
            "ncon": int(data.ncon),
            **{
                f"delta_q_{idx}_rad": float(delta_q[idx])
                for idx in range(6)
            },
        })

    output = pd.DataFrame(rows)
    return output, reference_path


def at_time(dataframe: pd.DataFrame, time_s: float, column: str) -> float:
    return float(
        np.interp(
            time_s,
            dataframe["time_s"].to_numpy(dtype=float),
            dataframe[column].to_numpy(dtype=float),
        )
    )


def analyze_direction(direction: str):
    pulse_alpha = math.radians(PULSE_ALPHA_DEG)

    baseline, reference_path = run_replay(direction, 0.0)
    plus, _ = run_replay(direction, +pulse_alpha)
    minus, _ = run_replay(direction, -pulse_alpha)

    times = baseline["time_s"].to_numpy(dtype=float)
    base_pitch = baseline["root_pitch_rad"].to_numpy(dtype=float)
    plus_pitch = plus["root_pitch_rad"].to_numpy(dtype=float)
    minus_pitch = minus["root_pitch_rad"].to_numpy(dtype=float)

    plus_delta = plus_pitch - base_pitch
    minus_delta = minus_pitch - base_pitch
    symmetric_delta = 0.5 * (plus_delta - minus_delta)
    authority = symmetric_delta / pulse_alpha

    start = PULSE_TIME_S
    end = min(float(times[-1]), PULSE_TIME_S + 0.300)
    window = (times >= start) & (times <= end)
    window_authority = authority[window]
    window_times = times[window]

    peak_index = int(np.argmax(np.abs(window_authority)))
    peak_authority = float(window_authority[peak_index])
    peak_time = float(window_times[peak_index])
    authority_sign = int(np.sign(peak_authority))

    probes = {}
    for delay in PROBE_DELAYS_S:
        t_probe = min(float(times[-1]), PULSE_TIME_S + delay)
        probes[f"{int(round(delay * 1000))}ms"] = {
            "time_s": t_probe,
            "signed_pitch_authority_rad_per_rad": at_time(
                pd.DataFrame({"time_s": times, "value": authority}),
                t_probe,
                "value",
            ),
            "plus_pitch_change_deg": math.degrees(at_time(
                pd.DataFrame({"time_s": times, "value": plus_delta}),
                t_probe,
                "value",
            )),
            "minus_pitch_change_deg": math.degrees(at_time(
                pd.DataFrame({"time_s": times, "value": minus_delta}),
                t_probe,
                "value",
            )),
        }

    baseline_pitch_at_pulse = at_time(baseline, PULSE_TIME_S, "root_pitch_rad")
    baseline_rate_at_pulse = at_time(baseline, PULSE_TIME_S, "root_pitch_rate_rad_s")

    # A positive pitch error must use correction alpha with opposite sign to the
    # measured positive-alpha pitch authority.
    recommended_alpha_sign_for_positive_pitch_error = -authority_sign
    recommended_gain_note = (
        "Positive alpha increases measured root pitch; use negative alpha for a positive pitch error."
        if authority_sign > 0 else
        "Positive alpha decreases measured root pitch; use positive alpha for a positive pitch error."
        if authority_sign < 0 else
        "Authority is near zero; do not construct QP-MPC until joint distribution is revised."
    )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    baseline_path = LOG_DIR / f"c1_qp0_baseline_{direction}.csv"
    plus_path = LOG_DIR / f"c1_qp0_plus_pulse_{direction}.csv"
    minus_path = LOG_DIR / f"c1_qp0_minus_pulse_{direction}.csv"
    baseline.to_csv(baseline_path, index=False)
    plus.to_csv(plus_path, index=False)
    minus.to_csv(minus_path, index=False)

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(times, np.rad2deg(plus_delta), label="+alpha pulse minus baseline")
    axis.plot(times, np.rad2deg(minus_delta), label="-alpha pulse minus baseline")
    axis.axvline(PULSE_TIME_S, linestyle="--", label="pulse start")
    axis.axvline(PULSE_TIME_S + PULSE_DURATION_S, linestyle="--", label="pulse end")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Root-pitch deviation from C0 baseline (deg)")
    axis.set_title(f"C1-QP0 pitch authority pulse response — {direction}")
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    pulse_plot = FIG_DIR / f"c1_qp0_{direction}_pulse_response.png"
    figure.savefig(pulse_plot, dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(times, authority, label="symmetric pitch authority")
    axis.axvline(PULSE_TIME_S, linestyle="--", label="pulse start")
    axis.axvline(PULSE_TIME_S + PULSE_DURATION_S, linestyle="--", label="pulse end")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Pitch authority (rad pitch / rad alpha)")
    axis.set_title(f"C1-QP0 identified signed pitch authority — {direction}")
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    authority_plot = FIG_DIR / f"c1_qp0_{direction}_authority.png"
    figure.savefig(authority_plot, dpi=180)
    plt.close(figure)

    return {
        "direction": direction,
        "reference": str(reference_path),
        "baseline_alignment": {
            "baseline_root_pitch_at_pulse_rad": baseline_pitch_at_pulse,
            "baseline_root_pitch_at_pulse_deg": math.degrees(baseline_pitch_at_pulse),
            "baseline_root_pitch_rate_at_pulse_rad_s": baseline_rate_at_pulse,
            "note": (
                "This C0 MuJoCo baseline trajectory, rather than the raw Pinocchio "
                "free-flyer pitch, must be used as the fixed-phase pitch reference "
                "for the MuJoCo QP-MPC validation."
            ),
        },
        "pulse": {
            "start_time_s": PULSE_TIME_S,
            "duration_s": PULSE_DURATION_S,
            "alpha_pulse_deg": PULSE_ALPHA_DEG,
            "joint_distribution_G_rad_per_rad": [float(item) for item in G],
        },
        "identified_authority": {
            "peak_signed_authority_rad_per_rad": peak_authority,
            "peak_response_time_s": peak_time,
            "sign_of_positive_alpha_effect": authority_sign,
            "recommended_alpha_sign_for_positive_pitch_error": recommended_alpha_sign_for_positive_pitch_error,
            "recommendation": recommended_gain_note,
            "probe_values": probes,
        },
        "integrity": {
            "baseline_min_contacts": int(baseline["ncon"].min()),
            "plus_min_contacts": int(plus["ncon"].min()),
            "minus_min_contacts": int(minus["ncon"].min()),
            "finite_authority": bool(np.all(np.isfinite(authority))),
            "nonzero_authority": bool(abs(peak_authority) > 1e-4),
        },
        "outputs": {
            "baseline_csv": str(baseline_path),
            "plus_pulse_csv": str(plus_path),
            "minus_pulse_csv": str(minus_path),
            "pulse_plot": str(pulse_plot),
            "authority_plot": str(authority_plot),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Identify signed pitch authority for C1 QP-MPC."
    )
    parser.add_argument(
        "--direction",
        choices=("forward", "backward", "both"),
        default="both",
    )
    args = parser.parse_args()
    directions = ("forward", "backward") if args.direction == "both" else (args.direction,)

    print("===== C1-QP0: MUJOCO PITCH-AUTHORITY CALIBRATION =====")
    print("No body disturbance is injected in this stage.")
    print(
        f"Pulse: +/-{PULSE_ALPHA_DEG:.2f} deg alpha for "
        f"{PULSE_DURATION_S:.3f}s at t={PULSE_TIME_S:.3f}s"
    )
    print("The C0 MuJoCo baseline becomes the fixed-phase pitch reference.")

    report = {
        "stage": "C1_QP0_MuJoCo_pitch_authority_and_reference_alignment",
        "purpose": (
            "Identify the signed root-pitch response to the FDDP-guided joint "
            "correction distribution and export the C0 MuJoCo nominal baseline."
        ),
        "directions": {},
    }

    for direction in directions:
        result = analyze_direction(direction)
        report["directions"][direction] = result
        authority = result["identified_authority"]
        print()
        print(f"----- {direction.upper()} -----")
        print(
            "Baseline root pitch at pulse time (deg): "
            f"{result['baseline_alignment']['baseline_root_pitch_at_pulse_deg']:+.4f}"
        )
        print(
            "Peak signed authority (rad/rad): "
            f"{authority['peak_signed_authority_rad_per_rad']:+.6f} "
            f"at t={authority['peak_response_time_s']:.3f}s"
        )
        print("Recommendation: " + authority["recommendation"])
        print(
            "Authority valid: "
            f"{result['integrity']['finite_authority'] and result['integrity']['nonzero_authority']}"
        )

    summary_path = LOG_DIR / "c1_qp0_pitch_authority_calibration.yaml"
    summary_path.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    print()
    print("C1-QP0 CALIBRATION COMPLETE")
    print(f"Saved summary:\n  {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nC1-QP0 CALIBRATION FAILED: {error}", file=sys.stderr)
        sys.exit(1)
