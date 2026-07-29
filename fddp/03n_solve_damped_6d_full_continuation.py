#!/usr/bin/env python3
"""
STAGE 7C — Damped 6D-contact FDDP full-amplitude continuation.

Validated formulation carried forward from Stage 7C micro-shift:
  - two 6D flat-foot contacts fixed at the actual x0 world poses;
  - JMinvJt_damping = 1e-8;
  - zero Baumgarte gains;
  - state-tracking reference is a scaled contact-consistent warm-start,
    NOT static x0 at every node;
  - FDDP solves a sequence of small CoM_y amplitudes, each seeded from the
    previous validated FDDP solution.

This script solves both:
  forward:  +0.5 -> +1.0 -> ... -> +6.0 mm
  backward: -0.5 -> -1.0 -> ... -> -6.0 mm

Only a final trajectory that passes all integrity gates is exported as:
  transfer/trajectories/com_shift_forward_ref.npz
  transfer/trajectories/com_shift_forward_ref_summary.yaml
  transfer/trajectories/com_shift_backward_ref.npz
  transfer/trajectories/com_shift_backward_ref_summary.yaml

tau_ref is FDDP analysis output only. It must NOT be sent to ST3215 servos.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import pinocchio as pin
import crocoddyl


# ------------------------------- Configuration -------------------------------
DT = 0.010
HORIZON_NODES = 120

# Start from the already-passed 0.5-mm validation and grow by 0.5 mm.
HOMOTOPY_MAGNITUDES_M = [
    0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030,
    0.0035, 0.0040, 0.0045, 0.0050, 0.0055, 0.0060,
]
MAX_ITER_PER_STAGE = 220
RETRY_MAX_ITER = 420

# Chosen from the finite-difference derivative audit.
JM_INV_JT_DAMPING = 1.0e-8
CONTACT_GAINS_KP_KD = np.array([0.0, 0.0], dtype=float)

LEFT_FOOT_FRAME = "left_foot_contact_fixed"
RIGHT_FOOT_FRAME = "right_foot_contact_fixed"

WARMSTART_FULL_SHIFT_M = 0.006

RUNNING_STATE_TRACKING_WEIGHT = 2.0
RUNNING_CONTROL_WEIGHT = 1.0e-3
TERMINAL_STATE_TRACKING_WEIGHT = 10.0
COM_TRACKING_WEIGHT_Y = 5.0e4
COM_TRACKING_WEIGHT_XZ = 2.0e2

# Final full-trajectory integrity gates.
MAX_SUPPORT_TRANSLATION_ERROR_M = 5.0e-4     # 0.5 mm
MAX_SUPPORT_ROTATION_ERROR_RAD = np.deg2rad(0.5)
MIN_JOINT_LIMIT_MARGIN_RAD = 1.0e-2          # 0.57 deg
MIN_ACHIEVED_SHIFT_M = 0.0050                # 5.0 mm
MAX_COM_Y_TRACKING_ERROR_M = 5.0e-4          # 0.5 mm
MAX_FINAL_COM_Y_RETURN_ERROR_M = 2.0e-4      # 0.2 mm
MAX_FINAL_GAP_NORM = 1.0e-7

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = PROJECT_ROOT / "model" / "biped_v1.urdf"
X0_PATH = PROJECT_ROOT / "transfer" / "initialisation" / "fddp_double_support_x0.npz"
TRAJECTORY_DIR = PROJECT_ROOT / "transfer" / "trajectories"
TRACE_DIR = PROJECT_ROOT / "transfer" / "initialisation"
LOG_DIR = PROJECT_ROOT / "results" / "logs"
FIG_DIR = PROJECT_ROOT / "results" / "figures"


def write_yaml(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")


def load_x0(path: Path, expected_size: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing initial state: {path}")

    with np.load(path) as loaded:
        for key in ("x0", "state", "initial_state"):
            if key in loaded.files:
                x0 = np.asarray(loaded[key], dtype=float).reshape(-1)
                break
        else:
            raise RuntimeError(
                f"No x0-like key in {path}; available keys: {loaded.files}"
            )

    if x0.size != expected_size:
        raise RuntimeError(f"x0 size {x0.size}; expected {expected_size}.")
    return x0


def load_warmstart(
    direction: str,
    expected_x_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    path = TRAJECTORY_DIR / f"com_shift_{direction}_warmstart.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing warm-start: {path}\n"
            "Run: python fddp/03f_build_double_support_com_shift_warmstart.py"
        )

    with np.load(path) as loaded:
        required = ("xs_init", "com_y_target", "t_ref")
        missing = [key for key in required if key not in loaded.files]
        if missing:
            raise RuntimeError(f"{path} missing keys {missing}; has {loaded.files}")

        xs = np.asarray(loaded["xs_init"], dtype=float)
        com_y_target = np.asarray(loaded["com_y_target"], dtype=float).reshape(-1)
        t_ref = np.asarray(loaded["t_ref"], dtype=float).reshape(-1)

    if xs.shape != expected_x_shape:
        raise RuntimeError(f"{path}: xs shape {xs.shape}; expected {expected_x_shape}")
    if com_y_target.shape != (HORIZON_NODES + 1,):
        raise RuntimeError(f"{path}: com_y_target shape is invalid")
    if t_ref.shape != (HORIZON_NODES + 1,):
        raise RuntimeError(f"{path}: t_ref shape is invalid")

    return xs, com_y_target, t_ref, path


def support_references(model, q0, left_frame_id, right_frame_id):
    data = model.createData()
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)
    return data.oMf[left_frame_id].copy(), data.oMf[right_frame_id].copy()


def scale_warmstart(model, x0, xs_full, ratio: float) -> np.ndarray:
    """
    Scale the configuration on Pinocchio's manifold instead of scaling raw
    quaternion coefficients. Scale tangent velocity linearly.
    """
    q0 = x0[:model.nq]
    output = np.zeros_like(xs_full)

    for k, x_full in enumerate(xs_full):
        q_full = x_full[:model.nq]
        v_full = x_full[model.nq:]

        tangent_difference = np.asarray(
            pin.difference(model, q0, q_full),
            dtype=float,
        ).reshape(-1)

        output[k, :model.nq] = np.asarray(
            pin.integrate(model, q0, ratio * tangent_difference),
            dtype=float,
        ).reshape(-1)
        output[k, model.nq:] = ratio * v_full

    output[0] = x0.copy()
    return output


def compute_com(model, xs: np.ndarray) -> np.ndarray:
    data = model.createData()
    values = np.zeros((xs.shape[0], 3), dtype=float)
    for k, x in enumerate(xs):
        values[k] = np.asarray(
            pin.centerOfMass(model, data, x[:model.nq]),
            dtype=float,
        ).reshape(3)
    return values


def build_action(
    state,
    actuation,
    x_reference,
    com_target,
    left_frame_id,
    right_frame_id,
    left_reference,
    right_reference,
    state_weight,
    control_weight,
    dt,
):
    nu = actuation.nu
    contacts = crocoddyl.ContactModelMultiple(state, nu)

    contacts.addContact(
        LEFT_FOOT_FRAME,
        crocoddyl.ContactModel6D(
            state,
            left_frame_id,
            left_reference,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            nu,
            CONTACT_GAINS_KP_KD,
        ),
    )
    contacts.addContact(
        RIGHT_FOOT_FRAME,
        crocoddyl.ContactModel6D(
            state,
            right_frame_id,
            right_reference,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            nu,
            CONTACT_GAINS_KP_KD,
        ),
    )

    costs = crocoddyl.CostModelSum(state, nu)

    state_residual = crocoddyl.ResidualModelState(state, x_reference, nu)
    costs.addCost(
        "warmstart_state_tracking",
        crocoddyl.CostModelResidual(state, state_residual),
        state_weight,
    )

    control_residual = crocoddyl.ResidualModelControl(state, np.zeros(nu))
    costs.addCost(
        "control_regularisation",
        crocoddyl.CostModelResidual(state, control_residual),
        control_weight,
    )

    com_residual = crocoddyl.ResidualModelCoMPosition(state, com_target, nu)
    com_activation = crocoddyl.ActivationModelWeightedQuad(
        np.array(
            [
                COM_TRACKING_WEIGHT_XZ,
                COM_TRACKING_WEIGHT_Y,
                COM_TRACKING_WEIGHT_XZ,
            ],
            dtype=float,
        )
    )
    costs.addCost(
        "com_tracking",
        crocoddyl.CostModelResidual(state, com_activation, com_residual),
        1.0,
    )

    differential = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state,
        actuation,
        contacts,
        costs,
        JM_INV_JT_DAMPING,
        False,
    )
    return crocoddyl.IntegratedActionModelEuler(differential, dt)


def build_problem(
    model,
    x0,
    xs_reference,
    com_y_target,
    left_frame_id,
    right_frame_id,
    left_reference,
    right_reference,
):
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    initial_com = compute_com(model, x0[None, :])[0]

    running_models = []
    for k in range(HORIZON_NODES):
        target = initial_com.copy()
        target[1] = com_y_target[k]
        running_models.append(
            build_action(
                state,
                actuation,
                xs_reference[k],
                target,
                left_frame_id,
                right_frame_id,
                left_reference,
                right_reference,
                RUNNING_STATE_TRACKING_WEIGHT,
                RUNNING_CONTROL_WEIGHT,
                DT,
            )
        )

    terminal_target = initial_com.copy()
    terminal_target[1] = com_y_target[-1]
    terminal_model = build_action(
        state,
        actuation,
        xs_reference[-1],
        terminal_target,
        left_frame_id,
        right_frame_id,
        left_reference,
        right_reference,
        TERMINAL_STATE_TRACKING_WEIGHT,
        0.0,
        0.0,
    )

    return crocoddyl.ShootingProblem(x0, running_models, terminal_model), state, actuation.nu


def total_cost(problem, xs, us) -> float:
    problem.calc(xs, us)
    return float(
        sum(float(data.cost) for data in problem.runningDatas)
        + float(problem.terminalData.cost)
    )


def gap_summary(problem, state, xs, us) -> dict:
    problem.calc(xs, us)
    gaps = []
    for k, data in enumerate(problem.runningDatas):
        gap = np.asarray(
            state.diff(xs[k + 1], data.xnext),
            dtype=float,
        ).reshape(-1)
        gaps.append(float(np.linalg.norm(gap)))

    gaps = np.asarray(gaps, dtype=float)
    return {
        "max_gap_norm": float(np.max(gaps)),
        "rms_gap_norm": float(np.sqrt(np.mean(gaps**2))),
        "first_gap_norm": float(gaps[0]),
        "last_gap_norm": float(gaps[-1]),
    }


def build_iteration_row(
    *,
    direction: str,
    stage_index: int,
    attempt: str,
    iteration: int,
    total_cost: float,
    max_gap_norm: float,
    rms_gap_norm: float,
    stop_criterion: float,
    regularization: float,
    step_length_alpha: float,
    iteration_wall_time_ms: float,
) -> dict:
    """Normalize one FDDP iteration record for CSV and YAML export."""
    return {
        "direction": str(direction),
        "stage_index": int(stage_index),
        "attempt": str(attempt),
        "iteration": int(iteration),
        "total_cost": float(total_cost),
        "max_gap_norm": float(max_gap_norm),
        "rms_gap_norm": float(rms_gap_norm),
        "stop_criterion": float(stop_criterion),
        "regularization": float(regularization),
        "step_length_alpha": float(step_length_alpha),
        "iteration_wall_time_ms": float(iteration_wall_time_ms),
    }


class SolverIterationRecorder(crocoddyl.CallbackAbstract):
    """Capture FDDP iteration diagnostics without changing solver behavior."""

    def __init__(self, problem, state, metadata: dict):
        crocoddyl.CallbackAbstract.__init__(self)
        self.problem = problem
        self.state = state
        self.metadata = dict(metadata)
        self.records: list[dict] = []
        self._last_tick = time.perf_counter()

    def __call__(self, solver):
        now = time.perf_counter()
        elapsed_ms = (now - self._last_tick) * 1000.0
        self._last_tick = now
        gaps = gap_summary(self.problem, self.state, solver.xs, solver.us)
        self.records.append(
            build_iteration_row(
                **self.metadata,
                iteration=int(solver.iter),
                total_cost=float(solver.cost),
                max_gap_norm=gaps["max_gap_norm"],
                rms_gap_norm=gaps["rms_gap_norm"],
                stop_criterion=float(solver.stop),
                regularization=float(solver.preg),
                step_length_alpha=float(solver.stepLength),
                iteration_wall_time_ms=elapsed_ms,
            )
        )


def write_convergence_artifacts(records: list[dict]) -> dict:
    """Write reproducible E1 logs and figures from callback records."""
    if not records:
        raise RuntimeError("No FDDP iteration records were captured.")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(records)
    csv_path = LOG_DIR / "e1_fddp_convergence_iterations.csv"
    dataframe.to_csv(csv_path, index=False)

    cost_gap_path = FIG_DIR / "e1_fddp_cost_and_gap.png"
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for (direction, attempt), group in dataframe.groupby(["direction", "attempt"], sort=False):
        label = f"{direction} {attempt}"
        axes[0].plot(group.index, group["total_cost"], marker="o", markersize=2, label=label)
        axes[1].plot(group.index, group["max_gap_norm"], marker="o", markersize=2, label=label)
    axes[0].set_ylabel("Total cost")
    axes[1].set_ylabel("Max shooting gap norm")
    axes[1].set_xlabel("Logged solver iteration")
    if np.all(dataframe["total_cost"].to_numpy(dtype=float) > 0.0):
        axes[0].set_yscale("log")
    if np.all(dataframe["max_gap_norm"].to_numpy(dtype=float) > 0.0):
        axes[1].set_yscale("log")
    for axis in axes:
        axis.grid(True)
        axis.legend()
    figure.tight_layout()
    figure.savefig(cost_gap_path, dpi=180)
    plt.close(figure)

    time_path = FIG_DIR / "e1_fddp_iteration_time.png"
    figure, axis = plt.subplots(figsize=(11, 4.5))
    for (direction, attempt), group in dataframe.groupby(["direction", "attempt"], sort=False):
        axis.plot(group.index, group["iteration_wall_time_ms"], marker="o", markersize=2, label=f"{direction} {attempt}")
    axis.set_xlabel("Logged solver iteration")
    axis.set_ylabel("Iteration wall time (ms)")
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(time_path, dpi=180)
    plt.close(figure)

    summary_path = LOG_DIR / "e1_fddp_convergence_summary.yaml"
    summary = {
        "stage": "E1_fddp_convergence_and_feasibility",
        "record_count": int(len(dataframe)),
        "directions": sorted(str(value) for value in dataframe["direction"].unique()),
        "outputs": {
            "iteration_csv": str(csv_path),
            "cost_and_gap_figure": str(cost_gap_path),
            "iteration_time_figure": str(time_path),
        },
    }
    write_yaml(summary_path, summary)
    summary["outputs"]["summary_yaml"] = str(summary_path)
    return summary


def support_error_summary(
    model,
    xs,
    left_frame_id,
    right_frame_id,
    left_reference,
    right_reference,
):
    data = model.createData()
    left_t_max = right_t_max = 0.0
    left_r_max = right_r_max = 0.0

    for x in xs:
        pin.forwardKinematics(model, data, x[:model.nq])
        pin.updateFramePlacements(model, data)

        left_t_max = max(
            left_t_max,
            float(
                np.linalg.norm(
                    data.oMf[left_frame_id].translation - left_reference.translation
                )
            ),
        )
        right_t_max = max(
            right_t_max,
            float(
                np.linalg.norm(
                    data.oMf[right_frame_id].translation - right_reference.translation
                )
            ),
        )
        left_r_max = max(
            left_r_max,
            float(
                np.linalg.norm(
                    pin.log3(
                        left_reference.rotation.T @ data.oMf[left_frame_id].rotation
                    )
                )
            ),
        )
        right_r_max = max(
            right_r_max,
            float(
                np.linalg.norm(
                    pin.log3(
                        right_reference.rotation.T @ data.oMf[right_frame_id].rotation
                    )
                )
            ),
        )

    return {
        "max_left_translation_error_m": float(left_t_max),
        "max_right_translation_error_m": float(right_t_max),
        "max_left_rotation_error_rad": float(left_r_max),
        "max_right_rotation_error_rad": float(right_r_max),
    }


def joint_limit_summary(model, xs):
    joints = xs[:, 7:model.nq]
    lower = np.asarray(model.lowerPositionLimit[7:model.nq], dtype=float)
    upper = np.asarray(model.upperPositionLimit[7:model.nq], dtype=float)

    lower_margin = joints - lower[None, :]
    upper_margin = upper[None, :] - joints
    min_lower = float(np.min(lower_margin))
    min_upper = float(np.min(upper_margin))
    minimum = float(min(min_lower, min_upper))

    return {
        "minimum_lower_margin_rad": min_lower,
        "minimum_upper_margin_rad": min_upper,
        "minimum_margin_rad": minimum,
        "within_position_limits": bool(
            np.all(lower_margin >= -1e-10) and np.all(upper_margin >= -1e-10)
        ),
        "has_safety_margin": bool(minimum >= MIN_JOINT_LIMIT_MARGIN_RAD),
    }


def solve_stage(problem, state, xs_seed, us_seed, stage_metadata: dict):
    """
    Try the normal iteration budget first. If it does not converge, retry from
    the resulting finite trajectory with a larger budget once.
    """
    attempts = []
    records = []
    current_xs = [x.copy() for x in xs_seed]
    current_us = [u.copy() for u in us_seed]

    for label, iterations in (
        ("primary", MAX_ITER_PER_STAGE),
        ("retry", RETRY_MAX_ITER),
    ):
        initial_cost = total_cost(problem, current_xs, current_us)
        solver = crocoddyl.SolverFDDP(problem)
        solver.th_stop = 1e-7
        recorder = SolverIterationRecorder(
            problem,
            state,
            {**stage_metadata, "attempt": label},
        )
        solver.setCallbacks([recorder])
        returned_success = bool(
            solver.solve(current_xs, current_us, iterations, False)
        )
        records.extend(recorder.records)
        stop_passed = bool(float(solver.stop) <= float(solver.th_stop))
        accepted = bool(returned_success or stop_passed)

        xs = [np.asarray(x, dtype=float).copy() for x in solver.xs]
        us = [np.asarray(u, dtype=float).copy() for u in solver.us]
        final_cost = total_cost(problem, xs, us)

        attempt = {
            "attempt": label,
            "max_iterations": int(iterations),
            "return_success": bool(returned_success),
            "stop_criterion": float(solver.stop),
            "threshold": float(solver.th_stop),
            "accepted_convergence": bool(accepted),
            "initial_cost": float(initial_cost),
            "final_cost": float(final_cost),
            "iteration_record_count": int(len(recorder.records)),
        }
        attempts.append(attempt)

        finite = (
            np.isfinite(final_cost)
            and all(np.all(np.isfinite(x)) for x in xs)
            and all(np.all(np.isfinite(u)) for u in us)
        )

        if accepted and finite:
            return xs, us, attempt, attempts, records

        current_xs = xs
        current_us = us

    return current_xs, current_us, attempts[-1], attempts, records


def solve_direction(
    direction: str,
    sign: float,
    model,
    x0,
    left_frame_id,
    right_frame_id,
    left_reference,
    right_reference,
):
    state_dim = model.nq + model.nv
    xs_full, target_full, t_ref, warmstart_path = load_warmstart(
        direction,
        (HORIZON_NODES + 1, state_dim),
    )

    trace_path = TRACE_DIR / f"damped_6d_{direction}_continuation_trace.yaml"
    initial_y = float(target_full[0])

    trace = {
        "stage": "7C_damped_6d_contact_full_continuation",
        "direction": direction,
        "purpose": (
            "Record every continuation stage before exporting a full-scale "
            "FDDP CoM-shift reference."
        ),
        "contact_dynamics": {
            "type": "two 6D flat-foot contacts",
            "JMinvJt_damping": float(JM_INV_JT_DAMPING),
            "baumgarte_gains_Kp_Kd": [float(v) for v in CONTACT_GAINS_KP_KD],
        },
        "warmstart_source": str(warmstart_path),
        "magnitudes_m": [float(v) for v in HOMOTOPY_MAGNITUDES_M],
        "stages": [],
        "status": "running",
    }
    write_yaml(trace_path, trace)

    previous_xs = None
    previous_us = None
    final_xs = None
    final_us = None
    final_target = None
    final_reference = None
    iteration_records = []

    for stage_index, magnitude in enumerate(HOMOTOPY_MAGNITUDES_M, start=1):
        signed_magnitude = sign * magnitude
        ratio = magnitude / WARMSTART_FULL_SHIFT_M

        xs_reference = scale_warmstart(model, x0, xs_full, ratio)
        target = initial_y + ratio * (target_full - initial_y)

        problem, state, nu = build_problem(
            model,
            x0,
            xs_reference,
            target,
            left_frame_id,
            right_frame_id,
            left_reference,
            right_reference,
        )

        # The first stage uses the validated scaled warm-start. Later stages
        # use the preceding FDDP solution, which is closer dynamically.
        if previous_xs is None:
            xs_seed = [x.copy() for x in xs_reference]
            us_seed = [np.zeros(nu, dtype=float) for _ in range(HORIZON_NODES)]
            seed_type = "scaled_kinematic_warmstart"
        else:
            xs_seed = [x.copy() for x in previous_xs]
            us_seed = [u.copy() for u in previous_us]
            seed_type = "previous_fddp_solution"

        initial_gaps = gap_summary(problem, state, xs_seed, us_seed)
        xs, us, final_attempt, attempts, stage_records = solve_stage(
            problem,
            state,
            xs_seed,
            us_seed,
            {
                "direction": direction,
                "stage_index": int(stage_index),
            },
        )
        iteration_records.extend(stage_records)

        xs_array = np.asarray(xs, dtype=float)
        us_array = np.asarray(us, dtype=float)
        final_gaps = gap_summary(problem, state, xs, us)
        stage_cost = total_cost(problem, xs, us)

        stage_record = {
            "stage_index": int(stage_index),
            "requested_peak_shift_y_m": float(signed_magnitude),
            "warmstart_scale_ratio": float(ratio),
            "seed_type": seed_type,
            "initial_shooting_gaps": initial_gaps,
            "attempts": attempts,
            "accepted_convergence": bool(final_attempt["accepted_convergence"]),
            "final_cost": float(stage_cost),
            "final_shooting_gaps": final_gaps,
        }
        trace["stages"].append(stage_record)
        write_yaml(trace_path, trace)

        print(
            f"  {direction:8s} stage {stage_index:02d}/{len(HOMOTOPY_MAGNITUDES_M)} | "
            f"target {signed_magnitude * 1000.0:+.2f} mm | "
            f"seed={seed_type} | accepted={final_attempt['accepted_convergence']} | "
            f"stop={final_attempt['stop_criterion']:.3e} | "
            f"gap={final_gaps['max_gap_norm']:.3e} | "
            f"cost={stage_cost:.3e}"
        )

        valid_stage = (
            final_attempt["accepted_convergence"]
            and np.isfinite(stage_cost)
            and final_gaps["max_gap_norm"] <= MAX_FINAL_GAP_NORM
            and np.all(np.isfinite(xs_array))
            and np.all(np.isfinite(us_array))
        )
        if not valid_stage:
            trace["status"] = "failed"
            trace["failure_stage_index"] = int(stage_index)
            trace["failure_requested_peak_shift_y_m"] = float(signed_magnitude)
            write_yaml(trace_path, trace)
            raise RuntimeError(
                f"{direction}: continuation rejected at "
                f"{signed_magnitude * 1000.0:+.2f} mm. "
                f"Inspect {trace_path}."
            )

        previous_xs = xs
        previous_us = us
        final_xs = xs_array
        final_us = us_array
        final_target = target
        final_reference = xs_reference

    # Evaluate final scale.
    com = compute_com(model, final_xs)
    initial_com_y = float(com[0, 1])
    if sign > 0:
        achieved_shift = float(np.max(com[:, 1]) - initial_com_y)
    else:
        achieved_shift = float(np.min(com[:, 1]) - initial_com_y)

    tracking_error = float(np.max(np.abs(com[:, 1] - final_target)))
    return_error = float(abs(com[-1, 1] - initial_com_y))
    supports = support_error_summary(
        model,
        final_xs,
        left_frame_id,
        right_frame_id,
        left_reference,
        right_reference,
    )
    limits = joint_limit_summary(model, final_xs)

    final_problem, final_state, _ = build_problem(
        model,
        x0,
        final_reference,
        final_target,
        left_frame_id,
        right_frame_id,
        left_reference,
        right_reference,
    )
    final_gaps = gap_summary(
        final_problem,
        final_state,
        [x.copy() for x in final_xs],
        [u.copy() for u in final_us],
    )

    print()
    print(f"----- {direction.upper()} FINAL DAMPED 6D FDDP RESULT -----")
    print(
        "Requested / achieved CoM_y shift: "
        f"{sign * HOMOTOPY_MAGNITUDES_M[-1] * 1000.0:+.3f} / "
        f"{achieved_shift * 1000.0:+.3f} mm"
    )
    print(f"Peak |CoM_y tracking error|: {tracking_error * 1000.0:.5f} mm")
    print(f"Final CoM_y return error: {return_error * 1000.0:.5f} mm")
    print(
        "Final shooting gaps max/RMS: "
        f"{final_gaps['max_gap_norm']:.3e} / {final_gaps['rms_gap_norm']:.3e}"
    )
    print(
        "Max L/R foot translation error: "
        f"{supports['max_left_translation_error_m'] * 1000.0:.5f} / "
        f"{supports['max_right_translation_error_m'] * 1000.0:.5f} mm"
    )
    print(
        "Max L/R foot rotation error: "
        f"{np.rad2deg(supports['max_left_rotation_error_rad']):.5f} / "
        f"{np.rad2deg(supports['max_right_rotation_error_rad']):.5f} deg"
    )
    print(
        "Minimum lower/upper joint-limit margin: "
        f"{limits['minimum_lower_margin_rad']:.6f} / "
        f"{limits['minimum_upper_margin_rad']:.6f} rad"
    )

    failures = []
    if final_gaps["max_gap_norm"] > MAX_FINAL_GAP_NORM:
        failures.append("final shooting gap exceeds threshold")
    if abs(achieved_shift) < MIN_ACHIEVED_SHIFT_M:
        failures.append("achieved CoM shift is below 5 mm")
    if tracking_error > MAX_COM_Y_TRACKING_ERROR_M:
        failures.append("CoM tracking error exceeds threshold")
    if return_error > MAX_FINAL_COM_Y_RETURN_ERROR_M:
        failures.append("final CoM return error exceeds threshold")
    if supports["max_left_translation_error_m"] > MAX_SUPPORT_TRANSLATION_ERROR_M:
        failures.append("left foot translation error exceeds threshold")
    if supports["max_right_translation_error_m"] > MAX_SUPPORT_TRANSLATION_ERROR_M:
        failures.append("right foot translation error exceeds threshold")
    if supports["max_left_rotation_error_rad"] > MAX_SUPPORT_ROTATION_ERROR_RAD:
        failures.append("left foot rotation error exceeds threshold")
    if supports["max_right_rotation_error_rad"] > MAX_SUPPORT_ROTATION_ERROR_RAD:
        failures.append("right foot rotation error exceeds threshold")
    if not limits["within_position_limits"]:
        failures.append("joint position limit violated")
    elif not limits["has_safety_margin"]:
        failures.append("joint limit safety margin too small")

    if failures:
        trace["status"] = "failed_final_integrity"
        trace["final_failures"] = failures
        write_yaml(trace_path, trace)
        print("Full continuation integrity: FAILED")
        for item in failures:
            print(f"  - {item}")
        raise RuntimeError(
            f"{direction}: full-scale trajectory rejected; inspect {trace_path}."
        )

    # Export only after all final physics/kinematics gates pass.
    q_ref = final_xs[:, 7:model.nq]
    dq_ref = final_xs[:, model.nq + 6:]
    phase = np.full(HORIZON_NODES + 1, "double_support", dtype="U16")

    if q_ref.shape != (HORIZON_NODES + 1, 6):
        raise RuntimeError(f"Unexpected q_ref shape: {q_ref.shape}")
    if dq_ref.shape != (HORIZON_NODES + 1, 6):
        raise RuntimeError(f"Unexpected dq_ref shape: {dq_ref.shape}")

    trajectory_path = TRAJECTORY_DIR / f"com_shift_{direction}_ref.npz"
    summary_path = TRAJECTORY_DIR / f"com_shift_{direction}_ref_summary.yaml"

    np.savez(
        trajectory_path,
        t_ref=t_ref,
        q_ref=q_ref,
        dq_ref=dq_ref,
        tau_ref=final_us,
        state_ref=final_xs,
        com_ref_world=com,
        com_y_target=final_target,
        phase=phase,
        direction=np.asarray(direction),
        contact_mode=np.asarray("double_support_6d_flat_foot"),
        left_contact_active=np.ones(HORIZON_NODES + 1, dtype=np.uint8),
        right_contact_active=np.ones(HORIZON_NODES + 1, dtype=np.uint8),
        jminvjt_damping=np.asarray(JM_INV_JT_DAMPING),
        warmstart_source=np.asarray(str(warmstart_path)),
    )

    summary = {
        "stage": "7C_damped_6d_contact_fddp_full_continuation",
        "task": f"double_support_{direction}_com_shift_and_return",
        "purpose": (
            "Generate a dynamically feasible full ±6 mm nominal CoM-shift "
            "reference using FDDP continuation from a validated micro-shift "
            "formulation."
        ),
        "coordinate_convention": {"forward_axis": "+Y", "pitch_axis": "+X"},
        "model": {
            "urdf": str(URDF_PATH),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "ndx": 24,
            "nu": 6,
        },
        "contact_dynamics": {
            "type": "two 6D flat-foot contacts",
            "reference_mode": "exact world foot poses at x0",
            "JMinvJt_damping": float(JM_INV_JT_DAMPING),
            "baumgarte_gains_Kp_Kd": [float(v) for v in CONTACT_GAINS_KP_KD],
            **supports,
        },
        "continuation": {
            "magnitudes_m": [float(v) for v in HOMOTOPY_MAGNITUDES_M],
            "max_iterations_per_stage": int(MAX_ITER_PER_STAGE),
            "retry_max_iterations": int(RETRY_MAX_ITER),
            "trace_file": str(trace_path),
            "stage_count": int(len(trace["stages"])),
        },
        "warmstart": {
            "source": str(warmstart_path),
            "reference_type": "scaled contact-consistent kinematic warm-start",
        },
        "com_motion": {
            "requested_peak_shift_y_m": float(sign * HOMOTOPY_MAGNITUDES_M[-1]),
            "achieved_peak_shift_y_m": float(achieved_shift),
            "peak_abs_y_tracking_error_m": float(tracking_error),
            "final_return_error_y_m": float(return_error),
        },
        "final_shooting_gaps": final_gaps,
        "joint_limits": limits,
        "outputs": {
            "trajectory_npz": str(trajectory_path),
            "q_ref_shape": [int(v) for v in q_ref.shape],
            "dq_ref_shape": [int(v) for v in dq_ref.shape],
            "tau_ref_shape": [int(v) for v in final_us.shape],
            "state_ref_shape": [int(v) for v in final_xs.shape],
        },
    }
    write_yaml(summary_path, summary)

    trace["status"] = "passed"
    trace["final_trajectory"] = str(trajectory_path)
    trace["final_summary"] = str(summary_path)
    trace["final_com_motion"] = summary["com_motion"]
    trace["final_shooting_gaps"] = final_gaps
    write_yaml(trace_path, trace)

    print("Full continuation integrity: PASSED")
    print(f"Saved:\n  {trajectory_path}\n  {summary_path}\n  {trace_path}")
    return iteration_records


def main():
    print("===== STAGE 7C: DAMPED 6D-CONTACT FDDP FULL CONTINUATION =====")
    print("Validated model: 6D flat-foot contacts + derivative-consistent damping.")
    print("Task: double-support CoM shift and return; this is NOT walking.")
    print(f"URDF: {URDF_PATH}")
    print(
        f"JMinvJt damping: {JM_INV_JT_DAMPING:.0e}; "
        f"contact gains: {CONTACT_GAINS_KP_KD.tolist()}"
    )
    print(
        "Continuation targets (mm): "
        + ", ".join(f"{m * 1000.0:.1f}" for m in HOMOTOPY_MAGNITUDES_M)
    )

    if not URDF_PATH.exists():
        raise FileNotFoundError(f"URDF not found: {URDF_PATH}")

    model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
    if model.nq != 13 or model.nv != 12:
        raise RuntimeError(
            f"Unexpected model dimensions nq={model.nq}, nv={model.nv}; "
            "expected nq=13, nv=12."
        )

    x0 = load_x0(X0_PATH, model.nq + model.nv)
    left_frame_id = model.getFrameId(LEFT_FOOT_FRAME)
    right_frame_id = model.getFrameId(RIGHT_FOOT_FRAME)
    if left_frame_id >= model.nframes or right_frame_id >= model.nframes:
        raise RuntimeError(
            f"Could not resolve support frames: {LEFT_FOOT_FRAME}, {RIGHT_FOOT_FRAME}"
        )

    left_reference, right_reference = support_references(
        model, x0[:model.nq], left_frame_id, right_frame_id
    )

    forward_records = solve_direction(
        "forward",
        +1.0,
        model,
        x0,
        left_frame_id,
        right_frame_id,
        left_reference,
        right_reference,
    )
    backward_records = solve_direction(
        "backward",
        -1.0,
        model,
        x0,
        left_frame_id,
        right_frame_id,
        left_reference,
        right_reference,
    )

    e1_summary = write_convergence_artifacts(forward_records + backward_records)
    print(f"E1 iteration records: {e1_summary['record_count']}")
    print(f"E1 summary: {e1_summary['outputs']['summary_yaml']}")
    print()
    print("STAGE 7C DAMPED 6D FULL CONTINUATION PASSED")
    print("Next stage: replay the exported q_ref/dq_ref in MuJoCo with actuator dynamics.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nDAMPED 6D FULL CONTINUATION FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
