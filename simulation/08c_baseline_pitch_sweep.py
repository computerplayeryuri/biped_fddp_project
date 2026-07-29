"""Stage 5B: C0 neutral-position baseline under initial pitch offsets.

This script intentionally uses no FDDP, no MPC, no actuator-aware reference
filter, and no contact-event synchronisation. It identifies the disturbance
range at which the neutral position-control baseline stops being reliable.
"""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from common import (
    MODEL_PATH,
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

# This is deliberately a deterministic, one-trial-per-condition screening run.
# Later robustness experiments will add repeat trials and randomised noise.
PITCH_OFFSETS_DEG = [0.0, 5.0, -5.0, 10.0, -10.0, 15.0, -15.0, 20.0, -20.0]
TOTAL_TIME_S = 6.0
CONTROL_DT_S = 0.020
RECOVERY_BAND_DEG = 1.0


def label_from_offset(offset_deg: float) -> str:
    if abs(offset_deg) < 1e-12:
        return "zero0deg"
    sign = "plus" if offset_deg > 0 else "minus"
    return f"{sign}{int(abs(offset_deg))}deg"


def first_settling_time(times: np.ndarray, pitch_deg: np.ndarray) -> float:
    """Return first time that stays inside the recovery band until trial end."""
    within = np.abs(pitch_deg) <= RECOVERY_BAND_DEG
    for idx in range(len(times)):
        if times[idx] >= 0.10 and bool(np.all(within[idx:])):
            return float(times[idx])
    return float("nan")


def run_trial(model: mujoco.MjModel, offset_deg: float) -> dict:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    root_pitch_adr = qpos_address(model, "root_pitch_x")
    data.qpos[root_pitch_adr] = math.radians(offset_deg)
    data.ctrl[:] = 0.0  # C0: neutral position reference only
    mujoco.mj_forward(model, data)

    debouncer = ContactDebouncer()
    rows: list[dict] = []
    next_log_time = 0.0
    fallen = False

    while data.time < TOTAL_TIME_S:
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)

        if data.time + 1e-12 >= next_log_time:
            raw = contact_sensor_values(model, data)
            switches = debouncer.update(raw)
            q = get_actuated_q(model, data)
            pitch_deg = math.degrees(torso_pitch_rad(model, data))

            row = {
                "time_s": float(data.time),
                "initial_pitch_deg": float(offset_deg),
                "torso_pitch_deg": float(pitch_deg),
                "hip_height_m": float(hip_height_m(model, data)),
                "joint_rmse_deg": float(math.degrees(np.sqrt(np.mean(q * q)))),
                "fall_flag": int(is_fallen(model, data)),
                **{f"q_{i}": float(value) for i, value in enumerate(q)},
                **raw,
                **switches,
            }
            rows.append(row)
            next_log_time += CONTROL_DT_S

            if row["fall_flag"]:
                fallen = True
                break

    label = label_from_offset(offset_deg)
    log_path = LOG_DIR / f"baseline_pitch_sweep_{label}.csv"
    append_csv(log_path, rows)

    times = np.asarray([row["time_s"] for row in rows], dtype=float)
    pitches = np.asarray([row["torso_pitch_deg"] for row in rows], dtype=float)
    rmses = np.asarray([row["joint_rmse_deg"] for row in rows], dtype=float)

    settled = first_settling_time(times, pitches) if not fallen else float("nan")
    recovered = bool((not fallen) and np.isfinite(settled))
    last = rows[-1]

    return {
        "trial": label,
        "initial_pitch_deg": float(offset_deg),
        "fall": int(fallen),
        "recovered_within_1deg": int(recovered),
        "settling_time_s": settled,
        "peak_abs_pitch_deg": float(np.max(np.abs(pitches))),
        "final_pitch_deg": float(pitches[-1]),
        "peak_joint_rmse_deg": float(np.max(rmses)),
        "final_left_full_support": int(last["left_full_support"]),
        "final_right_full_support": int(last["right_full_support"]),
        "log_file": log_path.name,
    }


def main() -> None:
    ensure_output_dirs()
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    print("===== STAGE 5B: INITIAL-PITCH SWEEP (C0) =====")
    print("Controller: neutral position reference only; no FDDP, no MPC.")
    print("Goal: identify the recoverable initial-pitch range of the baseline.\n")

    summary: list[dict] = []
    for offset in PITCH_OFFSETS_DEG:
        result = run_trial(model, offset)
        summary.append(result)
        settle_text = (
            f"{result['settling_time_s']:.3f} s"
            if np.isfinite(result["settling_time_s"])
            else "not recovered"
        )
        print(
            f"Initial pitch {offset:+5.1f} deg | "
            f"fall={bool(result['fall'])} | "
            f"recovered={bool(result['recovered_within_1deg'])} | "
            f"settling={settle_text} | "
            f"peak RMSE={result['peak_joint_rmse_deg']:.3f} deg"
        )

    summary_path = LOG_DIR / "baseline_pitch_sweep_summary.csv"
    append_csv(summary_path, summary)
    print(f"\nSaved summary: {summary_path}")
    print("INITIAL-PITCH SWEEP FINISHED")


if __name__ == "__main__":
    main()
