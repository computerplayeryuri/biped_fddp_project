#!/usr/bin/env python3
"""
C1-QP — Fixed-phase reduced-order QP-MPC closed-loop MuJoCo validation.

Architecture
------------
C0:
    offline 6D-contact FDDP -> q_ref, dq_ref, state_ref

C1-QP:
    MuJoCo measured pitch/pitch-rate
    + fixed-time FDDP reference phase
    + reduced-order constrained QP-MPC
    -> alpha_0
    -> delta_q = G * alpha_0
    -> q_cmd = q_ref + delta_q
    -> position-actuator proxy -> MuJoCo

This is intentionally not a second full nonlinear FDDP solve every 10 ms.
The runtime benchmark established that low-iteration FDDP solutions retained
large shooting gaps, while dynamically consistent solves were approximately
100 ms or slower. The QP layer therefore performs the high-frequency online
correction and preserves FDDP as the nonlinear nominal trajectory generator.

Important scope
---------------
- This is fixed-time-phase C1. It does NOT yet use contact-event phase sync.
- The QP prediction model is reduced-order and deliberately documented as a
  local pitch-error model; it is not a replacement for the full FDDP model.
- tau_ref is not sent as a command. Only q_ref + delta_q is used.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import argparse
import math
import sys
import time

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
import yaml


# ------------------------------- Paths ---------------------------------------
ROOT = Path(__file__).resolve().parents[1]
XML_PATH = ROOT / "model" / "biped_planar.xml"
TRAJ_DIR = ROOT / "transfer" / "trajectories"
LOG_DIR = ROOT / "results" / "logs"
FIG_DIR = ROOT / "results" / "figures"

# ------------------------- Existing model naming -----------------------------
JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
]
ROOT_PITCH_JOINT = "root_pitch_x"
TORSO_BODY = "hip_plate_link"

# --------------------- Actuator-aware C0 execution proxy --------------------
COMMAND_DELAY_S = 0.020
ACTUATOR_TIME_CONSTANT_S = 0.030
MAX_COMMAND_SPEED_RAD_S = 8.0

# --------------------------- C1 QP-MPC settings -----------------------------
# The QP controller is updated every 20 ms (50 Hz); MuJoCo retains its 2 ms
# internal step from the XML. This avoids claiming an unrealistic 100-Hz
# nonlinear FDDP solve rate.
QP_UPDATE_S = 0.020
QP_HORIZON = 10                         # 0.20 s reduced-order horizon
QP_GRADIENT_ITERATIONS = 80
QP_MAX_ALPHA_RAD = math.radians(5.0)    # virtual correction amplitude limit
QP_COMMAND_SLEW_RAD = math.radians(1.5) # alpha change / QP update

# Local discrete body-pitch error model:
# x = [pitch_error_rad, pitch_rate_error_rad_s]
# x(k+1) = A x(k) + B alpha(k)
#
# Positive alpha represents a positive pitch-correction allocation. The sign
# of B encodes that positive alpha reduces a positive pitch error.
REDUCED_A_DAMPING = 0.85
REDUCED_B_GAIN = 12.0                   # rad/s^2 per rad virtual correction
QP_Q_PITCH = 180.0
QP_Q_RATE = 8.0
QP_R_ALPHA = 0.35
QP_S_SLEW = 1.2

# FDDP-guided symmetric correction distribution G [rad joint / rad alpha].
# It is derived from the successful C1-A forward local replan:
# +2 deg pitch error led approximately to
#   [+0.54, +0.12, -0.46] deg on each leg.
# Thus alpha is the reduced-order amplitude and G distributes it into
# physically small, symmetric hip/knee/ankle position corrections.
G_CORRECTION = np.array(
    [0.27, 0.062, -0.228, 0.27, 0.062, -0.228],
    dtype=float,
)

# Controlled, reproducible disturbance injected into the simulator.
DISTURBANCE_TIME_S = 0.400
DISTURBANCE_PITCH_DEG = 2.0
DISTURBANCE_PITCH_RATE_RAD_S = 0.25

# Evaluation window and pass gates.
SETTLING_BAND_DEG = 0.50
MIN_IMPROVEMENT_RATIO = 0.10             # C1 final error must be >=10% lower
MAX_C1_CORRECTION_DEG = 2.0              # maximum per-joint QP correction
MIN_TOTAL_CONTACTS = 1
MAX_TORSO_PITCH_DEG = 15.0
MIN_TORSO_HEIGHT_RATIO = 0.65


class PositionActuatorProxy:
    """Delay + first-order lag + joint-position speed limit."""

    def __init__(self, count: int, dt: float):
        self.count = int(count)
        self.dt = float(dt)
        self.delay_steps = int(round(COMMAND_DELAY_S / self.dt))
        self.history = deque(maxlen=self.delay_steps + 1)
        self.output = np.zeros(self.count, dtype=float)

    def reset(self, initial_q: np.ndarray) -> None:
        initial_q = np.asarray(initial_q, dtype=float).reshape(self.count)
        self.history.clear()
        for _ in range(self.delay_steps + 1):
            self.history.append(initial_q.copy())
        self.output = initial_q.copy()

    def update(self, command_q: np.ndarray) -> np.ndarray:
        command_q = np.asarray(command_q, dtype=float).reshape(self.count)
        self.history.append(command_q.copy())
        delayed = self.history[0]

        alpha = min(1.0, self.dt / ACTUATOR_TIME_CONSTANT_S)
        step = alpha * (delayed - self.output)
        step_limit = MAX_COMMAND_SPEED_RAD_S * self.dt
        step = np.clip(step, -step_limit, step_limit)
        self.output = self.output + step
        return self.output.copy()


def name_or_none(model, object_type, object_id: int):
    return mujoco.mj_id2name(model, object_type, object_id) or None


def require_joint(model, name: str) -> int:
    identifier = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if identifier < 0:
        available = [
            name_or_none(model, mujoco.mjtObj.mjOBJ_JOINT, index)
            for index in range(model.njnt)
        ]
        raise RuntimeError(f"Joint '{name}' unavailable. Available: {available}")
    return int(identifier)


def require_body(model, name: str) -> int:
    identifier = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if identifier < 0:
        available = [
            name_or_none(model, mujoco.mjtObj.mjOBJ_BODY, index)
            for index in range(model.nbody)
        ]
        raise RuntimeError(f"Body '{name}' unavailable. Available: {available}")
    return int(identifier)


def map_actuators(model, joint_ids: list[int]) -> list[int]:
    output = []
    for joint_id in joint_ids:
        candidates = [
            actuator_id
            for actuator_id in range(model.nu)
            if int(model.actuator_trnid[actuator_id, 0]) == joint_id
        ]
        if len(candidates) != 1:
            available = [
                name_or_none(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
                for index in range(model.nu)
            ]
            raise RuntimeError(
                f"Expected one actuator mapped to joint id {joint_id}, found {candidates}. "
                f"Available actuators: {available}"
            )
        output.append(candidates[0])
    return output


def mass_weighted_com(model, data) -> np.ndarray:
    masses = np.asarray(model.body_mass, dtype=float)
    xipos = np.asarray(data.xipos, dtype=float).reshape(model.nbody, 3)
    total = float(np.sum(masses))
    if total <= 0:
        raise RuntimeError("Non-positive total body mass.")
    return np.sum(masses[:, None] * xipos, axis=0) / total


def load_c0_reference(direction: str):
    path = TRAJ_DIR / f"com_shift_{direction}_ref.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing C0 FDDP reference: {path}")

    with np.load(path) as archive:
        required = ("t_ref", "q_ref", "dq_ref", "state_ref", "com_y_target")
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise RuntimeError(f"{path} missing {missing}; contains {archive.files}")

        t_ref = np.asarray(archive["t_ref"], dtype=float).reshape(-1)
        q_ref = np.asarray(archive["q_ref"], dtype=float)
        dq_ref = np.asarray(archive["dq_ref"], dtype=float)
        state_ref = np.asarray(archive["state_ref"], dtype=float)
        com_target = np.asarray(archive["com_y_target"], dtype=float).reshape(-1)

    if q_ref.shape != (len(t_ref), 6):
        raise RuntimeError(f"Unexpected q_ref shape {q_ref.shape}.")
    if dq_ref.shape != (len(t_ref), 6):
        raise RuntimeError(f"Unexpected dq_ref shape {dq_ref.shape}.")
    if state_ref.shape != (len(t_ref), 25):
        raise RuntimeError(f"Unexpected state_ref shape {state_ref.shape}.")
    if com_target.shape != t_ref.shape:
        raise RuntimeError("com_y_target shape mismatch.")

    return path, t_ref, q_ref, dq_ref, state_ref, com_target


def interpolate(t: float, t_ref: np.ndarray, values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return np.asarray(np.interp(t, t_ref, values), dtype=float)
    return np.asarray(
        [np.interp(t, t_ref, values[:, index]) for index in range(values.shape[1])],
        dtype=float,
    )


def pinocchio_freeflyer_pitch(state_ref: np.ndarray) -> np.ndarray:
    """Root pitch about +X from state_ref quaternion [x,y,z,w]."""
    x = state_ref[:, 3]
    y = state_ref[:, 4]
    z = state_ref[:, 5]
    w = state_ref[:, 6]
    return np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))


def projected_box_qp(
    H: np.ndarray,
    f: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    step_size: float | None = None,
):
    """
    Solve convex box QP:
        min 0.5 U.T H U + f.T U
        s.t. lower <= U <= upper
    using deterministic projected gradient descent.

    H is positive definite by construction. This avoids additional solver
    dependencies while retaining an explicit convex QP.
    """
    if step_size is None:
        eigen_max = float(np.max(np.linalg.eigvalsh(H)))
        step_size = 1.0 / max(eigen_max, 1e-9)
    solution = np.zeros_like(f)
    for _ in range(QP_GRADIENT_ITERATIONS):
        solution = np.clip(
            solution - step_size * (H @ solution + f),
            lower,
            upper,
        )
    return solution


class FixedPhaseQPMPC:
    """
    Reduced-order constrained QP MPC for root-pitch tracking.

    alpha is a scalar virtual correction that is mapped to six joint-position
    corrections through the FDDP-guided G_CORRECTION distribution.
    """

    def __init__(self, update_dt: float):
        self.dt = float(update_dt)
        self.A = np.array(
            [[1.0, self.dt], [0.0, REDUCED_A_DAMPING]],
            dtype=float,
        )
        self.B = np.array(
            [[0.0], [-REDUCED_B_GAIN * self.dt]],
            dtype=float,
        )
        self.previous_alpha = 0.0
        self.H, self.Sx, self.Su, self.Qbar, self.D = self._build_condensed_qp()
        eigen_max = float(np.max(np.linalg.eigvalsh(self.H)))
        self.qp_step_size = 1.0 / max(eigen_max, 1e-9)

    def _build_condensed_qp(self):
        nx = 2
        nu = 1
        N = QP_HORIZON

        Sx = np.zeros((nx * N, nx), dtype=float)
        Su = np.zeros((nx * N, nu * N), dtype=float)

        A_power = np.eye(nx)
        for row_step in range(N):
            A_power = self.A @ A_power
            Sx[row_step * nx:(row_step + 1) * nx, :] = A_power

            for column_step in range(row_step + 1):
                power = np.linalg.matrix_power(self.A, row_step - column_step)
                Su[
                    row_step * nx:(row_step + 1) * nx,
                    column_step * nu:(column_step + 1) * nu,
                ] = power @ self.B

        Q = np.diag([QP_Q_PITCH, QP_Q_RATE])
        Qbar = np.kron(np.eye(N), Q)
        Rbar = QP_R_ALPHA * np.eye(N)

        # Slew penalty ||D U - d_prev||^2, where D creates U0-prev and
        # Uk-U(k-1) terms.
        D = np.eye(N)
        for index in range(1, N):
            D[index, index - 1] = -1.0

        H = Su.T @ Qbar @ Su + Rbar + QP_S_SLEW * (D.T @ D)
        H = 0.5 * (H + H.T) + 1e-10 * np.eye(N)
        return H, Sx, Su, Qbar, D

    def solve(self, pitch_error: float, pitch_rate_error: float):
        state_error = np.array([pitch_error, pitch_rate_error], dtype=float)
        N = QP_HORIZON

        d_prev = np.zeros(N, dtype=float)
        d_prev[0] = self.previous_alpha

        f = self.Su.T @ self.Qbar @ self.Sx @ state_error - QP_S_SLEW * (self.D.T @ d_prev)

        lower = np.full(N, -QP_MAX_ALPHA_RAD, dtype=float)
        upper = np.full(N, QP_MAX_ALPHA_RAD, dtype=float)

        # Enforce first move rate bound relative to previous alpha. Later
        # smoothness is cost-based because separate pairwise rate constraints
        # would require a general inequality-QP solver.
        lower[0] = max(lower[0], self.previous_alpha - QP_COMMAND_SLEW_RAD)
        upper[0] = min(upper[0], self.previous_alpha + QP_COMMAND_SLEW_RAD)

        start = time.perf_counter_ns()
        sequence = projected_box_qp(self.H, f, lower, upper, step_size=self.qp_step_size)
        solve_time_us = (time.perf_counter_ns() - start) / 1e3
        alpha = float(sequence[0])
        self.previous_alpha = alpha
        return alpha, sequence, float(solve_time_us)


def settling_time(time_s: np.ndarray, error_deg: np.ndarray, start_time: float):
    """First time after the disturbance at which error remains inside band."""
    indices = np.where(time_s >= start_time)[0]
    for index in indices:
        if np.all(np.abs(error_deg[index:]) <= SETTLING_BAND_DEG):
            return float(time_s[index] - start_time)
    return None


def replay_case(direction: str, use_qp_mpc: bool):
    reference_path, t_ref, q_ref, dq_ref, state_ref, com_y_target = load_c0_reference(direction)

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)

    joint_ids = [require_joint(model, name) for name in JOINT_NAMES]
    joint_qpos = [int(model.jnt_qposadr[item]) for item in joint_ids]
    joint_dof = [int(model.jnt_dofadr[item]) for item in joint_ids]
    actuator_ids = map_actuators(model, joint_ids)

    root_pitch_joint_id = require_joint(model, ROOT_PITCH_JOINT)
    root_pitch_qpos = int(model.jnt_qposadr[root_pitch_joint_id])
    root_pitch_dof = int(model.jnt_dofadr[root_pitch_joint_id])
    torso_body_id = require_body(model, TORSO_BODY)

    desired_root_pitch = pinocchio_freeflyer_pitch(state_ref)
    desired_root_pitch_rate = state_ref[:, 13 + 3]

    mujoco.mj_resetData(model, data)
    for address, value in zip(joint_qpos, q_ref[0]):
        data.qpos[address] = value
    for address in joint_dof:
        data.qvel[address] = 0.0
    mujoco.mj_forward(model, data)

    initial_torso_height = float(data.xpos[torso_body_id][2])
    initial_com_y = float(mass_weighted_com(model, data)[1])

    proxy = PositionActuatorProxy(6, dt)
    proxy.reset(q_ref[0])

    qpm = FixedPhaseQPMPC(QP_UPDATE_S)
    update_steps = max(1, int(round(QP_UPDATE_S / dt)))
    current_delta_q = np.zeros(6, dtype=float)

    duration = float(t_ref[-1])
    steps = int(round(duration / dt)) + 1
    disturbed = False
    records = []

    for step in range(steps):
        sim_time = min(float(data.time), duration)

        if not disturbed and sim_time >= DISTURBANCE_TIME_S:
            data.qpos[root_pitch_qpos] += math.radians(DISTURBANCE_PITCH_DEG)
            data.qvel[root_pitch_dof] += DISTURBANCE_PITCH_RATE_RAD_S
            mujoco.mj_forward(model, data)
            disturbed = True

        q_nominal = interpolate(sim_time, t_ref, q_ref)
        theta_ref = float(interpolate(sim_time, t_ref, desired_root_pitch))
        omega_ref = float(interpolate(sim_time, t_ref, desired_root_pitch_rate))

        theta_measured = float(data.qpos[root_pitch_qpos])
        omega_measured = float(data.qvel[root_pitch_dof])
        pitch_error = theta_measured - theta_ref
        rate_error = omega_measured - omega_ref

        qp_solve_us = 0.0
        alpha = 0.0
        if use_qp_mpc and step % update_steps == 0:
            alpha, _sequence, qp_solve_us = qpm.solve(pitch_error, rate_error)
            current_delta_q = G_CORRECTION * alpha
        elif use_qp_mpc:
            alpha = qpm.previous_alpha
        else:
            current_delta_q[:] = 0.0

        q_command = q_nominal + current_delta_q
        q_proxy = proxy.update(q_command)

        data.ctrl[:] = 0.0
        for actuator_id, command in zip(actuator_ids, q_proxy):
            data.ctrl[actuator_id] = command
        mujoco.mj_step(model, data)

        theta_after = float(data.qpos[root_pitch_qpos])
        omega_after = float(data.qvel[root_pitch_dof])
        theta_ref_after = float(interpolate(min(float(data.time), duration), t_ref, desired_root_pitch))
        torso_height = float(data.xpos[torso_body_id][2])
        com_y = float(mass_weighted_com(model, data)[1])

        row = {
            "time_s": float(data.time),
            "reference_pitch_rad": theta_ref_after,
            "measured_pitch_rad": theta_after,
            "pitch_error_rad": theta_after - theta_ref_after,
            "pitch_error_deg": math.degrees(theta_after - theta_ref_after),
            "reference_pitch_rate_rad_s": omega_ref,
            "measured_pitch_rate_rad_s": omega_after,
            "pitch_rate_error_rad_s": omega_after - omega_ref,
            "qp_alpha_rad": alpha if use_qp_mpc else 0.0,
            "qp_solve_time_us": qp_solve_us,
            "torso_height_m": torso_height,
            "com_y_change_m": com_y - initial_com_y,
            "com_y_target_change_m": float(
                interpolate(min(float(data.time), duration), t_ref, com_y_target)
                - com_y_target[0]
            ),
            "ncon": int(data.ncon),
        }
        for index, joint_name in enumerate(JOINT_NAMES):
            row[f"{joint_name}_nominal_rad"] = float(q_nominal[index])
            row[f"{joint_name}_delta_q_rad"] = float(current_delta_q[index])
            row[f"{joint_name}_command_rad"] = float(q_command[index])
            row[f"{joint_name}_proxy_rad"] = float(q_proxy[index])
            row[f"{joint_name}_measured_rad"] = float(data.qpos[joint_qpos[index]])
        records.append(row)

    dataframe = pd.DataFrame(records)
    errors = dataframe["pitch_error_deg"].to_numpy()
    times = dataframe["time_s"].to_numpy()
    after = times >= DISTURBANCE_TIME_S

    initial_error = float(errors[np.argmax(after)])
    peak_error = float(np.max(np.abs(errors[after])))
    final_error = float(errors[-1])
    settling = settling_time(times, errors, DISTURBANCE_TIME_S)
    minimum_height = float(dataframe["torso_height_m"].min())
    max_correction_deg = float(
        np.rad2deg(
            np.abs(
                dataframe[
                    [f"{name}_delta_q_rad" for name in JOINT_NAMES]
                ].to_numpy()
            ).max()
        )
    )
    qp_times = dataframe.loc[dataframe["qp_solve_time_us"] > 0.0, "qp_solve_time_us"]
    qp_median_us = float(np.median(qp_times)) if len(qp_times) else 0.0
    qp_max_us = float(np.max(qp_times)) if len(qp_times) else 0.0

    prefix = "c1_qp_mpc" if use_qp_mpc else "c0_reference_only"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = LOG_DIR / f"{prefix}_{direction}_pitch_disturbance.csv"
    dataframe.to_csv(csv_path, index=False)

    metrics = {
        "case": prefix,
        "direction": direction,
        "reference": str(reference_path),
        "disturbance": {
            "time_s": DISTURBANCE_TIME_S,
            "root_pitch_deg": DISTURBANCE_PITCH_DEG,
            "root_pitch_rate_rad_s": DISTURBANCE_PITCH_RATE_RAD_S,
        },
        "pitch_tracking": {
            "initial_error_deg_after_disturbance": initial_error,
            "peak_abs_error_deg_after_disturbance": peak_error,
            "final_error_deg": final_error,
            "settling_time_to_0p5deg_s": settling,
        },
        "stability": {
            "initial_torso_height_m": initial_torso_height,
            "minimum_torso_height_m": minimum_height,
            "min_global_contact_count": int(dataframe["ncon"].min()),
        },
        "qp_metrics": {
            "update_period_s": QP_UPDATE_S if use_qp_mpc else None,
            "median_solve_time_us": qp_median_us if use_qp_mpc else None,
            "max_solve_time_us": qp_max_us if use_qp_mpc else None,
            "max_abs_joint_correction_deg": max_correction_deg,
        },
        "csv": str(csv_path),
    }
    return dataframe, metrics


def plot_comparison(direction: str, c0: pd.DataFrame, c1: pd.DataFrame):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(c0["time_s"], c0["pitch_error_deg"], label="C0: FDDP reference only")
    axis.plot(c1["time_s"], c1["pitch_error_deg"], label="C1: fixed-phase QP-MPC")
    axis.axvline(DISTURBANCE_TIME_S, linestyle="--", label="injected disturbance")
    axis.axhline(SETTLING_BAND_DEG, linestyle=":")
    axis.axhline(-SETTLING_BAND_DEG, linestyle=":")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Pitch tracking error (deg)")
    axis.set_title(f"C0 vs C1 fixed-phase QP-MPC pitch disturbance — {direction}")
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    pitch_plot = FIG_DIR / f"c1_qp_mpc_{direction}_pitch_comparison.png"
    figure.savefig(pitch_plot, dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 6))
    for name in JOINT_NAMES:
        axis.plot(
            c1["time_s"],
            np.rad2deg(c1[f"{name}_delta_q_rad"]),
            label=f"{name} correction",
        )
    axis.axvline(DISTURBANCE_TIME_S, linestyle="--", label="injected disturbance")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("QP joint correction (deg)")
    axis.set_title(f"C1 QP-MPC joint-position corrections — {direction}")
    axis.grid(True)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    command_plot = FIG_DIR / f"c1_qp_mpc_{direction}_joint_corrections.png"
    figure.savefig(command_plot, dpi=180)
    plt.close(figure)

    return pitch_plot, command_plot


def main():
    parser = argparse.ArgumentParser(
        description="C0 vs C1 fixed-phase QP-MPC closed-loop MuJoCo comparison."
    )
    parser.add_argument(
        "--direction",
        choices=("forward", "backward", "both"),
        default="both",
    )
    args = parser.parse_args()
    directions = ("forward", "backward") if args.direction == "both" else (args.direction,)

    print("===== C1-QP: FIXED-PHASE QP-MPC CLOSED-LOOP MUJOCO VALIDATION =====")
    print("C0 reference + 50 Hz reduced-order QP correction + actuator proxy.")
    print("This is fixed-time phase only; C2 contact-event phase correction is not used.")
    print(f"QP horizon: {QP_HORIZON} x {QP_UPDATE_S:.3f}s = {QP_HORIZON * QP_UPDATE_S:.3f}s")
    print(f"Injected disturbance at {DISTURBANCE_TIME_S:.3f}s: "
          f"{DISTURBANCE_PITCH_DEG:+.1f} deg, {DISTURBANCE_PITCH_RATE_RAD_S:+.2f} rad/s")

    report = {
        "stage": "C1_fixed_phase_reduced_order_QP_MPC_closed_loop_MuJoCo",
        "purpose": (
            "Compare C0 FDDP-reference-only replay against C1 fixed-phase "
            "reduced-order QP-MPC position correction under identical root-pitch disturbance."
        ),
        "scope_note": (
            "QP-MPC is the high-frequency online correction layer. Full nonlinear "
            "FDDP remains the offline nominal trajectory generator. C2 contact-event "
            "phase synchronisation is not active."
        ),
        "architecture": {
            "nominal_reference": "C0 FDDP q_ref/dq_ref/state_ref",
            "online_state": ["root_pitch", "root_pitch_rate"],
            "qp_output": "alpha_0",
            "joint_position_command": "q_cmd = q_ref + G_correction * alpha_0",
            "qp_update_period_s": QP_UPDATE_S,
            "qp_prediction_horizon_s": QP_HORIZON * QP_UPDATE_S,
            "joint_distribution_G_rad_per_rad": [float(v) for v in G_CORRECTION],
        },
        "reduced_pitch_model": {
            "state": "[pitch_error_rad, pitch_rate_error_rad_s]",
            "A": [[1.0, QP_UPDATE_S], [0.0, REDUCED_A_DAMPING]],
            "B": [[0.0], [-REDUCED_B_GAIN * QP_UPDATE_S]],
            "note": (
                "Local reduced-order prediction model for the online QP correction; "
                "full nonlinear contact dynamics are retained in the offline FDDP trajectory."
            ),
        },
        "disturbance": {
            "time_s": DISTURBANCE_TIME_S,
            "root_pitch_deg": DISTURBANCE_PITCH_DEG,
            "root_pitch_rate_rad_s": DISTURBANCE_PITCH_RATE_RAD_S,
        },
        "directions": {},
    }

    overall_pass = True
    for direction in directions:
        c0_df, c0_metrics = replay_case(direction, use_qp_mpc=False)
        c1_df, c1_metrics = replay_case(direction, use_qp_mpc=True)

        c0_final_abs = abs(c0_metrics["pitch_tracking"]["final_error_deg"])
        c1_final_abs = abs(c1_metrics["pitch_tracking"]["final_error_deg"])
        improvement = (
            (c0_final_abs - c1_final_abs) / max(c0_final_abs, 1e-9)
        )
        c1_stable = bool(
            c1_metrics["stability"]["minimum_torso_height_m"]
            >= MIN_TORSO_HEIGHT_RATIO * c1_metrics["stability"]["initial_torso_height_m"]
            and c1_metrics["stability"]["min_global_contact_count"] >= MIN_TOTAL_CONTACTS
            and c1_metrics["pitch_tracking"]["peak_abs_error_deg_after_disturbance"]
            <= MAX_TORSO_PITCH_DEG
            and c1_metrics["qp_metrics"]["max_abs_joint_correction_deg"]
            <= MAX_C1_CORRECTION_DEG
        )
        c1_better = bool(improvement >= MIN_IMPROVEMENT_RATIO)
        passed = bool(c1_stable and c1_better)
        overall_pass = overall_pass and passed

        pitch_plot, command_plot = plot_comparison(direction, c0_df, c1_df)

        report["directions"][direction] = {
            "C0": c0_metrics,
            "C1_QP_MPC": c1_metrics,
            "comparison": {
                "absolute_final_pitch_error_C0_deg": c0_final_abs,
                "absolute_final_pitch_error_C1_deg": c1_final_abs,
                "final_pitch_error_improvement_ratio": improvement,
                "C1_stability_pass": c1_stable,
                "C1_improves_C0_by_at_least_10_percent": c1_better,
                "integrity_passed": passed,
            },
            "plots": {
                "pitch_comparison": str(pitch_plot),
                "joint_corrections": str(command_plot),
            },
        }

        print()
        print(f"----- {direction.upper()} C0 vs C1-QP RESULT -----")
        print(
            "C0 / C1 final |pitch error| (deg): "
            f"{c0_final_abs:.4f} / {c1_final_abs:.4f}"
        )
        print(f"C1 final-error improvement: {100.0 * improvement:.2f}%")
        print(
            "C1 QP solve time median/max (us): "
            f"{c1_metrics['qp_metrics']['median_solve_time_us']:.2f} / "
            f"{c1_metrics['qp_metrics']['max_solve_time_us']:.2f}"
        )
        print(
            "C1 max |joint correction| (deg): "
            f"{c1_metrics['qp_metrics']['max_abs_joint_correction_deg']:.4f}"
        )
        print(
            "C1 min contacts / min torso height (m): "
            f"{c1_metrics['stability']['min_global_contact_count']} / "
            f"{c1_metrics['stability']['minimum_torso_height_m']:.4f}"
        )
        print(f"C1-QP comparison integrity: {'PASSED' if passed else 'FAILED'}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = LOG_DIR / "c1_qp_mpc_closed_loop_summary.yaml"
    summary_path.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")

    print()
    print("C1-QP CLOSED-LOOP VALIDATION " + ("PASSED" if overall_pass else "COMPLETED WITH FAILURES"))
    print(f"Saved summary:\n  {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nC1-QP CLOSED-LOOP VALIDATION FAILED: {error}", file=sys.stderr)
        sys.exit(1)
