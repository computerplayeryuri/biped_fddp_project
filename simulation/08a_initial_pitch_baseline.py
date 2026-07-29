from __future__ import annotations

"""Stage 5: C0 baseline under an initial sagittal-plane pitch offset.

This is intentionally NOT FDDP and NOT MPC.  It applies the same neutral
joint reference used in the standing test, but begins each trial with a small
floating-base pitch offset.  The resulting response is the baseline that the
later MPC controller must improve.

The script runs three deterministic conditions:
  - 0 deg  : nominal standing reference
  - +5 deg : initial positive base pitch
  - -5 deg : initial negative base pitch

The sign is the MuJoCo +X rotation convention used by the current planar
model.  Treat it as a reproducible simulation convention, not yet as a
human-facing "forward/backward" direction label.
"""

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
    get_actuated_dq,
    get_actuated_q,
    hip_height_m,
    is_fallen,
    qpos_address,
    torso_pitch_rad,
)

# Trial design.  Keep the first disturbance deliberately small and safe.
DURATION_S = 6.0
PITCH_OFFSETS_DEG = [0.0, +5.0, -5.0]
Q_REF = np.zeros(6, dtype=float)

# Geometry values from the current simple MuJoCo model.  They are used only to
# start a tilted robot with the lower foot edge close to the ground rather than
# deeply penetrating it.  Replace with calculated geometry once the final
# CAD-derived model is available.
HIP_TO_FOOT_CENTRE_M = 0.285
FOOT_HALF_LENGTH_M = 0.0725
FOOT_HALF_THICKNESS_M = 0.003


def initialise_pitch_offset(model, data, pitch_deg: float) -> None:
    """Reset to standing and apply a small root pitch about the support area."""
    mujoco.mj_resetDataKeyframe(model, data, 0)

    theta = math.radians(float(pitch_deg))

    root_y_addr = qpos_address(model, "root_forward_y")
    root_z_addr = qpos_address(model, "root_vertical_z")
    root_pitch_addr = qpos_address(model, "root_pitch_x")

    # Shift the root so the foot centre remains near the original horizontal
    # location.  Raise the body slightly so the lower toe/heel edge starts at
    # contact instead of inside the floor.
    data.qpos[root_y_addr] = -HIP_TO_FOOT_CENTRE_M * math.sin(theta)
    data.qpos[root_z_addr] = (
        HIP_TO_FOOT_CENTRE_M * math.cos(theta)
        + FOOT_HALF_LENGTH_M * abs(math.sin(theta))
        + FOOT_HALF_THICKNESS_M * math.cos(theta)
        - (HIP_TO_FOOT_CENTRE_M + FOOT_HALF_THICKNESS_M)
    )
    data.qpos[root_pitch_addr] = theta
    data.qvel[:] = 0.0
    data.ctrl[:] = Q_REF
    mujoco.mj_forward(model, data)


def run_trial(model, actuator_config: dict, pitch_offset_deg: float) -> dict:
    data = mujoco.MjData(model)
    initialise_pitch_offset(model, data, pitch_offset_deg)

    control_dt = float(actuator_config["control_timestep_s"])
    filter_ = ActuatorReferenceFilter(
        n_joints=6,
        control_dt_s=control_dt,
        delay_s=float(actuator_config["servo_delay_s"]),
        time_constant_s=float(actuator_config["servo_time_constant_s"]),
        safe_speed_rad_s=float(actuator_config["safe_speed_rad_s"]),
        max_change_per_cycle_rad=float(
            actuator_config["max_reference_change_per_cycle_rad"]
        ),
    )

    ctrl_min = model.actuator_ctrlrange[:, 0].copy()
    ctrl_max = model.actuator_ctrlrange[:, 1].copy()

    debouncer = ContactDebouncer()
    rows: list[dict] = []
    next_control_time = 0.0
    next_record_time = 0.0
    fall_detected = False

    while data.time < DURATION_S:
        if data.time + 1e-12 >= next_control_time:
            q_cmd = filter_.update(Q_REF)
            q_cmd = np.clip(q_cmd, ctrl_min, ctrl_max)
            data.ctrl[:] = q_cmd
            next_control_time += control_dt

        mujoco.mj_step(model, data)

        if data.time + 1e-12 >= next_record_time:
            q = get_actuated_q(model, data)
            dq = get_actuated_dq(model, data)
            raw_contacts = contact_sensor_values(model, data)
            switches = debouncer.update(raw_contacts)
            pitch_rad = torso_pitch_rad(model, data)
            fallen_now = is_fallen(model, data)

            row = {
                "time_s": float(data.time),
                "initial_pitch_deg": float(pitch_offset_deg),
                "hip_height_m": hip_height_m(model, data),
                "torso_pitch_rad": pitch_rad,
                "torso_pitch_deg": math.degrees(pitch_rad),
                "pitch_error_deg": math.degrees(pitch_rad),
                "fall_flag": int(fallen_now),
                "joint_rmse_to_cmd_deg": float(
                    math.degrees(np.sqrt(np.mean((q - q_cmd) ** 2)))
                ),
            }
            row.update({f"q_cmd_{i}": float(q_cmd[i]) for i in range(6)})
            row.update({f"q_{i}": float(q[i]) for i in range(6)})
            row.update({f"dq_{i}": float(dq[i]) for i in range(6)})
            row.update(raw_contacts)
            row.update(switches)
            rows.append(row)

            next_record_time += control_dt
            if fallen_now:
                fall_detected = True
                break

    sign = "plus" if pitch_offset_deg > 0 else "minus" if pitch_offset_deg < 0 else "zero"
    magnitude = int(abs(pitch_offset_deg))
    csv_path = LOG_DIR / f"baseline_initial_pitch_{sign}{magnitude}deg.csv"
    append_csv(csv_path, rows)

    initial_abs = abs(float(pitch_offset_deg))
    final_abs = abs(rows[-1]["torso_pitch_deg"]) if rows else float("nan")
    max_abs = max(abs(row["torso_pitch_deg"]) for row in rows) if rows else float("nan")
    recovered = bool(
        (not fall_detected)
        and math.isfinite(final_abs)
        and final_abs <= 1.0
    )

    return {
        "initial_pitch_deg": float(pitch_offset_deg),
        "csv_path": csv_path,
        "fall_detected": fall_detected,
        "recovered_to_within_1deg": recovered,
        "final_pitch_deg": rows[-1]["torso_pitch_deg"] if rows else float("nan"),
        "max_abs_pitch_deg": max_abs,
        "final_left_full_support": rows[-1]["left_full_support"] if rows else -1,
        "final_right_full_support": rows[-1]["right_full_support"] if rows else -1,
    }


def main() -> None:
    config_path = PROJECT / "config" / "actuator_model.yaml"
    with config_path.open("r") as file:
        actuator_config = yaml.safe_load(file)

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    ensure_output_dirs()

    print("===== STAGE 5: INITIAL-PITCH BASELINE (C0) =====")
    print("Controller: neutral position reference only; no FDDP, no MPC.")
    print("Purpose: establish the recovery baseline for later controller comparison.\n")

    summary_rows: list[dict] = []
    for pitch_offset_deg in PITCH_OFFSETS_DEG:
        result = run_trial(model, actuator_config, pitch_offset_deg)
        summary_rows.append(result)
        print(
            f"Initial pitch {pitch_offset_deg:+.1f} deg | "
            f"fall={result['fall_detected']} | "
            f"final pitch={result['final_pitch_deg']:+.2f} deg | "
            f"recovered within 1 deg={result['recovered_to_within_1deg']} | "
            f"log={result['csv_path'].name}"
        )

    summary_path = LOG_DIR / "baseline_initial_pitch_summary.csv"
    append_csv(summary_path, summary_rows)
    print(f"\nSaved summary: {summary_path}")
    print("INITIAL-PITCH BASELINE FINISHED")


if __name__ == "__main__":
    main()
