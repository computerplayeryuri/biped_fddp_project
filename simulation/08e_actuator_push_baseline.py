"""Stage 5C: actuator-aware C0 baseline under short horizontal pushes.

C0 means: neutral joint reference only. No FDDP, no MPC and no phase
re-synchronisation. Unlike the earlier pitch sweep, the simulated plant DOES
include the ST3215 command-side delay, first-order lag and command-rate limit.

The script applies a short force pulse along the robot forward/backward Y axis
at the electronics lump body, then records recovery metrics.
"""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import yaml

from actuator_reference_filter import ActuatorReferenceFilter
from common import (
    MODEL_PATH,
    PROJECT,
    LOG_DIR,
    ContactDebouncer,
    append_csv,
    contact_sensor_values,
    ensure_output_dirs,
    get_actuated_q,
    hip_height_m,
    is_fallen,
    qpos_address,
    torso_pitch_rad,
)

# Deterministic screening conditions. These are simulation-only stress levels,
# not claims about a human push magnitude on the physical robot.
PUSH_FORCES_N = [0.0, 2.0, -2.0, 4.0, -4.0, 6.0, -6.0]
PUSH_START_S = 1.00
PUSH_DURATION_S = 0.12
TOTAL_TIME_S = 4.00
RECOVERY_BAND_DEG = 1.0
CONTROL_REFERENCE = np.zeros(6, dtype=float)


def label_from_force(force_n: float) -> str:
    if abs(force_n) < 1e-12:
        return "zero0N"
    sign = "plus" if force_n > 0 else "minus"
    return f"{sign}{int(abs(force_n))}N"


def first_settling_time_after_push(times: np.ndarray, pitch_deg: np.ndarray) -> float:
    """First time after the pulse that stays within ±1 deg until trial end."""
    valid = np.abs(pitch_deg) <= RECOVERY_BAND_DEG
    t_after = PUSH_START_S + PUSH_DURATION_S
    for idx in range(len(times)):
        if times[idx] >= t_after and bool(np.all(valid[idx:])):
            return float(times[idx])
    return float("nan")


def main() -> None:
    ensure_output_dirs()

    config_path = PROJECT / "config" / "actuator_model.yaml"
    with config_path.open("r") as file:
        actuator_config = yaml.safe_load(file)

    control_dt = float(actuator_config["control_timestep_s"])

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    push_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "electronics_lump_link"
    )
    if push_body_id < 0:
        raise RuntimeError("electronics_lump_link was not found in the MuJoCo model.")

    root_y_adr = qpos_address(model, "root_forward_y")
    ctrl_min = model.actuator_ctrlrange[:, 0].copy()
    ctrl_max = model.actuator_ctrlrange[:, 1].copy()

    summary: list[dict] = []
    print("===== STAGE 5C: ACTUATOR-AWARE PUSH BASELINE (C0) =====")
    print("Controller: neutral reference through the actuator filter only.")
    print("No FDDP, no MPC, no contact-event phase re-synchronisation.\n")

    for force_n in PUSH_FORCES_N:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)

        filter_ = ActuatorReferenceFilter(
            n_joints=6,
            control_dt_s=control_dt,
            delay_s=float(actuator_config["servo_delay_s"]),
            time_constant_s=float(actuator_config["servo_time_constant_s"]),
            safe_speed_rad_s=float(actuator_config["safe_speed_rad_s"]),
            max_change_per_cycle_rad=float(
                actuator_config["max_reference_change_per_cycle_rad"]
            ),
            initial_reference=CONTROL_REFERENCE,
        )

        debouncer = ContactDebouncer()
        rows: list[dict] = []
        next_control_time = 0.0
        next_record_time = 0.0
        fall_detected = False

        while data.time < TOTAL_TIME_S:
            if data.time + 1e-12 >= next_control_time:
                q_cmd = filter_.update(CONTROL_REFERENCE)
                data.ctrl[:] = np.clip(q_cmd, ctrl_min, ctrl_max)
                next_control_time += control_dt

            # Clear previous external force at every physics step.
            data.xfrc_applied[:] = 0.0
            force_active = PUSH_START_S <= data.time < (PUSH_START_S + PUSH_DURATION_S)
            if force_active:
                # Index 1 corresponds to the global forward/backward Y force.
                data.xfrc_applied[push_body_id, 1] = force_n

            mujoco.mj_step(model, data)

            if data.time + 1e-12 >= next_record_time:
                raw = contact_sensor_values(model, data)
                switches = debouncer.update(raw)
                q = get_actuated_q(model, data)
                pitch_deg = math.degrees(torso_pitch_rad(model, data))
                fallen_now = is_fallen(model, data)

                row = {
                    "time_s": float(data.time),
                    "push_force_N": float(force_n if force_active else 0.0),
                    "trial_force_N": float(force_n),
                    "torso_pitch_deg": float(pitch_deg),
                    "hip_height_m": float(hip_height_m(model, data)),
                    "root_forward_y_m": float(data.qpos[root_y_adr]),
                    "joint_rmse_deg": float(math.degrees(np.sqrt(np.mean(q * q)))),
                    "fall_flag": int(fallen_now),
                    **{f"q_{i}": float(v) for i, v in enumerate(q)},
                    **raw,
                    **switches,
                }
                rows.append(row)
                next_record_time += control_dt

                if fallen_now:
                    fall_detected = True
                    break

        label = label_from_force(force_n)
        log_path = LOG_DIR / f"actuator_push_baseline_{label}.csv"
        append_csv(log_path, rows)

        times = np.asarray([r["time_s"] for r in rows], dtype=float)
        pitch = np.asarray([r["torso_pitch_deg"] for r in rows], dtype=float)
        forward = np.asarray([r["root_forward_y_m"] for r in rows], dtype=float)
        rmse = np.asarray([r["joint_rmse_deg"] for r in rows], dtype=float)
        settling = (
            first_settling_time_after_push(times, pitch)
            if not fall_detected
            else float("nan")
        )

        last = rows[-1]
        result = {
            "trial": label,
            "push_force_N": float(force_n),
            "pulse_start_s": PUSH_START_S,
            "pulse_duration_s": PUSH_DURATION_S,
            "fall": int(fall_detected),
            "recovered_within_1deg": int((not fall_detected) and np.isfinite(settling)),
            "settling_time_s": settling,
            "peak_abs_pitch_deg": float(np.max(np.abs(pitch))),
            "peak_forward_displacement_mm": float(1000.0 * np.max(np.abs(forward))),
            "peak_joint_rmse_deg": float(np.max(rmse)),
            "final_left_full_support": int(last["left_full_support"]),
            "final_right_full_support": int(last["right_full_support"]),
            "log_file": log_path.name,
        }
        summary.append(result)

        settle_text = f"{settling:.3f} s" if np.isfinite(settling) else "not recovered"
        print(
            f"Push {force_n:+4.1f} N | fall={bool(result['fall'])} | "
            f"recovered={bool(result['recovered_within_1deg'])} | "
            f"settling={settle_text} | "
            f"peak pitch={result['peak_abs_pitch_deg']:.2f} deg | "
            f"peak forward shift={result['peak_forward_displacement_mm']:.1f} mm"
        )

    summary_path = LOG_DIR / "actuator_push_baseline_summary.csv"
    append_csv(summary_path, summary)
    print(f"\nSaved summary: {summary_path}")
    print("ACTUATOR-AWARE PUSH BASELINE FINISHED")


if __name__ == "__main__":
    main()
