#!/usr/bin/env python3
"""FDDP single-step contact schedule and swing-foot reference helpers.

This module starts the first FDDP-generated gait increment: one right-foot
swing while the left foot remains in stance.  The solver is added only after
the schedule and reference contract is tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import crocoddyl
import sys
import pinocchio as pin


DT = 0.010
HORIZON_NODES = 120
STEP_FORWARD_Y_M = 0.010
SWING_CLEARANCE_Z_M = 0.020


@dataclass(frozen=True)
class PhaseSpec:
    """Fixed contact state for one contiguous interval of running nodes."""

    name: str
    node_count: int
    right_contact_active: bool
    left_contact_active: bool


def build_phase_schedule() -> list[PhaseSpec]:
    """Return the validated DS -> left-stance SS -> DS phase schedule."""
    return [
        PhaseSpec("double_support", 20, True, True),
        PhaseSpec("left_stance_right_swing", 70, False, True),
        PhaseSpec("double_support_landing", 30, True, True),
    ]


def validate_configuration(
    *,
    dt_s: float = DT,
    step_forward_m: float = STEP_FORWARD_Y_M,
    clearance_m: float = SWING_CLEARANCE_Z_M,
) -> None:
    """Reject invalid deterministic timing and swing-foot parameters."""
    if float(dt_s) <= 0.0:
        raise ValueError("DT must be positive.")
    if int(sum(phase.node_count for phase in build_phase_schedule())) != HORIZON_NODES:
        raise ValueError("Phase node counts must equal HORIZON_NODES.")
    if float(step_forward_m) <= 0.0:
        raise ValueError("STEP_FORWARD_Y_M must be positive.")
    if float(clearance_m) < 0.010:
        raise ValueError("SWING_CLEARANCE_Z_M must be at least 0.010 m.")


def quintic_progress(s: float) -> float:
    """Return C2-continuous scalar progress, clipped to the unit interval."""
    value = float(np.clip(s, 0.0, 1.0))
    return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)


def swing_reference_poses(right_start: pin.SE3) -> list[pin.SE3]:
    """Build one world-frame right-foot swing reference for all trajectory nodes."""
    validate_configuration()
    swing_start_node = 20
    swing_node_count = 70
    poses: list[pin.SE3] = []

    for node_index in range(HORIZON_NODES + 1):
        progress = quintic_progress((node_index - swing_start_node) / swing_node_count)
        translation = right_start.translation.copy()
        translation[1] += STEP_FORWARD_Y_M * progress
        translation[2] += SWING_CLEARANCE_Z_M * 4.0 * progress * (1.0 - progress)
        poses.append(pin.SE3(right_start.rotation.copy(), translation))

    return poses


# FDDP single-step problem configuration.  This solver is intentionally
# independent from Stage 7C and its continuation/warm-start trajectory.
COM_FORWARD_Y_M = 0.004
JM_INV_JT_DAMPING = 1.0e-8
CONTACT_GAINS_KP_KD = np.array([0.0, 0.0], dtype=float)
RUNNING_STATE_WEIGHT = 0.1
TERMINAL_STATE_WEIGHT = 1.0
RUNNING_CONTROL_WEIGHT = 1.0e-3
COM_TRACKING_WEIGHT_Y = 5.0e4
COM_TRACKING_WEIGHT_XZ = 2.0e2
SWING_TRACKING_WEIGHT = 1.0e8
SUPPORT_TRACKING_WEIGHT = 1.0e8
JOINT_SAFETY_WEIGHT = 1.0e6

LEFT_FOOT_FRAME = "left_foot_contact_fixed"
RIGHT_FOOT_FRAME = "right_foot_contact_fixed"
LEFT_CONTACT_NAME = "left_foot_6d"
RIGHT_CONTACT_NAME = "right_foot_6d"
LEFT_SUPPORT_COST_NAME = "left_support_se3_tracking"
RIGHT_SUPPORT_COST_NAME = "right_support_se3_tracking"
SWING_COST_NAME = "right_swing_se3_tracking"
JOINT_SAFETY_COST_NAME = "joint_safety_barrier"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = PROJECT_ROOT / "model" / "biped_v1.urdf"
X0_PATH = (
    PROJECT_ROOT / "transfer" / "initialisation" / "fddp_double_support_x0.npz"
)


def load_x0(path: Path, expected_size: int) -> np.ndarray:
    """Load the measured static double-support seed without any gait warm start."""
    if not path.exists():
        raise FileNotFoundError(f"Missing initial state: {path}")
    with np.load(path) as loaded:
        for key in ("x0", "state", "initial_state"):
            if key in loaded.files:
                x0 = np.asarray(loaded[key], dtype=float).reshape(-1)
                break
        else:
            raise RuntimeError(f"{path} has no x0-like key; keys: {loaded.files}")
    if x0.size != expected_size:
        raise RuntimeError(f"x0 size {x0.size}; expected {expected_size}.")
    return x0


def _frame_references(
    model: pin.Model,
    q0: np.ndarray,
    right_frame_id: int,
    left_frame_id: int,
) -> tuple[pin.SE3, pin.SE3]:
    data = model.createData()
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)
    return data.oMf[right_frame_id].copy(), data.oMf[left_frame_id].copy()


def _com_targets(model: pin.Model, x0: np.ndarray) -> np.ndarray:
    data = model.createData()
    initial_com = np.asarray(
        pin.centerOfMass(model, data, x0[: model.nq]), dtype=float
    ).reshape(3)
    targets = np.zeros((HORIZON_NODES + 1, 3), dtype=float)
    for node_index in range(HORIZON_NODES + 1):
        target = initial_com.copy()
        target[1] += COM_FORWARD_Y_M * quintic_progress(
            node_index / HORIZON_NODES
        )
        targets[node_index] = target
    return targets


def _joint_safety_activation(
    model: pin.Model, state, x0: np.ndarray
):
    """Keep every actuated joint at least the required margin inside its limits."""
    lower = np.full(state.ndx, -np.inf, dtype=float)
    upper = np.full(state.ndx, np.inf, dtype=float)
    joint_tangent = slice(6, 6 + (model.nv - 6))
    q_joint = x0[7:model.nq]
    lower[joint_tangent] = (
        model.lowerPositionLimit[7:model.nq]
        + JOINT_BARRIER_MARGIN_RAD
        - q_joint
    )
    upper[joint_tangent] = (
        model.upperPositionLimit[7:model.nq]
        - JOINT_BARRIER_MARGIN_RAD
        - q_joint
    )
    if np.any(lower[joint_tangent] >= upper[joint_tangent]):
        raise RuntimeError("Joint safety margin leaves no feasible configuration.")
    return crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(lower, upper)
    )


def _build_action_model(
    *,
    state,
    actuation,
    phase: PhaseSpec,
    x_reference: np.ndarray,
    com_target: np.ndarray,
    right_frame_id: int,
    left_frame_id: int,
    right_reference: pin.SE3,
    left_reference: pin.SE3,
    joint_safety_activation,
    dt_s: float,
):
    """Create one contact-dynamics node matching the declared phase."""
    nu = actuation.nu
    contacts = crocoddyl.ContactModelMultiple(state, nu)
    if phase.right_contact_active:
        contacts.addContact(
            RIGHT_CONTACT_NAME,
            crocoddyl.ContactModel6D(
                state,
                right_frame_id,
                right_reference,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                nu,
                CONTACT_GAINS_KP_KD,
            ),
        )
    if phase.left_contact_active:
        contacts.addContact(
            LEFT_CONTACT_NAME,
            crocoddyl.ContactModel6D(
                state,
                left_frame_id,
                left_reference,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                nu,
                CONTACT_GAINS_KP_KD,
            ),
        )

    costs = crocoddyl.CostModelSum(state, nu)
    state_residual = crocoddyl.ResidualModelState(state, x_reference, nu)
    costs.addCost(
        "static_x0_state_regularisation",
        crocoddyl.CostModelResidual(state, state_residual),
        TERMINAL_STATE_WEIGHT if dt_s == 0.0 else RUNNING_STATE_WEIGHT,
    )

    joint_safety_residual = crocoddyl.ResidualModelState(
        state, x_reference, nu
    )
    costs.addCost(
        JOINT_SAFETY_COST_NAME,
        crocoddyl.CostModelResidual(
            state, joint_safety_activation, joint_safety_residual
        ),
        JOINT_SAFETY_WEIGHT,
    )

    com_residual = crocoddyl.ResidualModelCoMPosition(state, com_target, nu)
    com_activation = crocoddyl.ActivationModelWeightedQuad(
        np.array(
            [COM_TRACKING_WEIGHT_XZ, COM_TRACKING_WEIGHT_Y, COM_TRACKING_WEIGHT_XZ],
            dtype=float,
        )
    )
    costs.addCost(
        "com_forward_tracking",
        crocoddyl.CostModelResidual(state, com_activation, com_residual),
        1.0,
    )

    if phase.right_contact_active:
        right_support_residual = crocoddyl.ResidualModelFramePlacement(
            state, right_frame_id, right_reference, nu
        )
        costs.addCost(
            RIGHT_SUPPORT_COST_NAME,
            crocoddyl.CostModelResidual(state, right_support_residual),
            SUPPORT_TRACKING_WEIGHT,
        )
    if phase.left_contact_active:
        left_support_residual = crocoddyl.ResidualModelFramePlacement(
            state, left_frame_id, left_reference, nu
        )
        costs.addCost(
            LEFT_SUPPORT_COST_NAME,
            crocoddyl.CostModelResidual(state, left_support_residual),
            SUPPORT_TRACKING_WEIGHT,
        )

    if not phase.right_contact_active:
        swing_residual = crocoddyl.ResidualModelFramePlacement(
            state, right_frame_id, right_reference, nu
        )
        costs.addCost(
            SWING_COST_NAME,
            crocoddyl.CostModelResidual(state, swing_residual),
            SWING_TRACKING_WEIGHT,
        )

    if dt_s > 0.0:
        control_residual = crocoddyl.ResidualModelControl(state, np.zeros(nu))
        costs.addCost(
            "control_regularisation",
            crocoddyl.CostModelResidual(state, control_residual),
            RUNNING_CONTROL_WEIGHT,
        )

    differential = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contacts, costs, JM_INV_JT_DAMPING, False
    )
    return crocoddyl.IntegratedActionModelEuler(differential, dt_s)


def build_single_step_problem(model: pin.Model, x0: np.ndarray):
    """Build DS -> left-stance SS -> landing DS FDDP without a gait warm start."""
    validate_configuration()
    expected_state_size = model.nq + model.nv
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    if x0.size != expected_state_size:
        raise ValueError(f"x0 size {x0.size}; expected {expected_state_size}.")

    right_frame_id = model.getFrameId(RIGHT_FOOT_FRAME)
    left_frame_id = model.getFrameId(LEFT_FOOT_FRAME)
    if right_frame_id >= model.nframes or left_frame_id >= model.nframes:
        raise RuntimeError("Could not resolve both flat-foot contact frames.")

    right_start, left_reference = _frame_references(
        model, x0[: model.nq], right_frame_id, left_frame_id
    )
    swing_references = swing_reference_poses(right_start)
    com_targets = _com_targets(model, x0)
    phases = build_phase_schedule()
    phase_by_node = [
        phase for phase in phases for _ in range(phase.node_count)
    ]
    if len(phase_by_node) != HORIZON_NODES:
        raise RuntimeError("Expanded phase schedule has an invalid node count.")

    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    joint_safety_activation = _joint_safety_activation(model, state, x0)
    running_models = [
        _build_action_model(
            state=state,
            actuation=actuation,
            phase=phase,
            x_reference=x0,
            com_target=com_targets[node_index],
            right_frame_id=right_frame_id,
            left_frame_id=left_frame_id,
            right_reference=swing_references[node_index],
            left_reference=left_reference,
            joint_safety_activation=joint_safety_activation,
            dt_s=DT,
        )
        for node_index, phase in enumerate(phase_by_node)
    ]
    terminal_model = _build_action_model(
        state=state,
        actuation=actuation,
        phase=phases[-1],
        x_reference=x0,
        com_target=com_targets[-1],
        right_frame_id=right_frame_id,
        left_frame_id=left_frame_id,
        right_reference=swing_references[-1],
        left_reference=left_reference,
        joint_safety_activation=joint_safety_activation,
        dt_s=0.0,
    )
    metadata = {
        "right_frame_id": right_frame_id,
        "left_frame_id": left_frame_id,
        "right_start": right_start,
        "left_reference": left_reference,
        "swing_references": swing_references,
        "com_targets": com_targets,
        "phase_by_node": phase_by_node,
    }
    return crocoddyl.ShootingProblem(x0, running_models, terminal_model), state, metadata


MAX_FINAL_SHOOTING_GAP_NORM = 1.0e-7
MAX_ACTIVE_SUPPORT_TRANSLATION_ERROR_M = 5.0e-4
MAX_ACTIVE_SUPPORT_ROTATION_ERROR_RAD = np.deg2rad(0.5)
MAX_LANDING_TRANSLATION_ERROR_M = 1.0e-3
MAX_LANDING_ROTATION_ERROR_RAD = np.deg2rad(1.0)
MIN_SWING_CLEARANCE_M = 0.010
MIN_JOINT_LIMIT_MARGIN_RAD = 1.0e-2
JOINT_BARRIER_MARGIN_RAD = 1.2e-2
MIN_FORWARD_COM_DISPLACEMENT_M = 0.002


def evaluate_acceptance(diagnostics: dict) -> dict:
    """Apply every approved gate before any nominal trajectory export."""
    required = (
        "max_shooting_gap_norm",
        "max_active_support_translation_error_m",
        "max_active_support_rotation_error_rad",
        "landing_translation_error_m",
        "landing_rotation_error_rad",
        "min_swing_clearance_m",
        "min_joint_limit_margin_rad",
        "achieved_forward_com_displacement_m",
    )
    missing = [name for name in required if name not in diagnostics]
    if missing:
        raise ValueError(f"Missing single-step diagnostics: {missing}")
    values = {name: float(diagnostics[name]) for name in required}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("Single-step diagnostics must all be finite.")

    failures = []
    if values["max_shooting_gap_norm"] > MAX_FINAL_SHOOTING_GAP_NORM:
        failures.append("final shooting gap exceeds threshold")
    if (
        values["max_active_support_translation_error_m"]
        > MAX_ACTIVE_SUPPORT_TRANSLATION_ERROR_M
    ):
        failures.append("active-support translation error exceeds threshold")
    if (
        values["max_active_support_rotation_error_rad"]
        > MAX_ACTIVE_SUPPORT_ROTATION_ERROR_RAD
    ):
        failures.append("active-support rotation error exceeds threshold")
    if values["landing_translation_error_m"] > MAX_LANDING_TRANSLATION_ERROR_M:
        failures.append("landing position error exceeds threshold")
    if values["landing_rotation_error_rad"] > MAX_LANDING_ROTATION_ERROR_RAD:
        failures.append("landing rotation error exceeds threshold")
    if values["min_swing_clearance_m"] < MIN_SWING_CLEARANCE_M:
        failures.append("swing clearance below threshold")
    if values["min_joint_limit_margin_rad"] < MIN_JOINT_LIMIT_MARGIN_RAD:
        failures.append("joint safety margin below threshold")
    if (
        values["achieved_forward_com_displacement_m"]
        < MIN_FORWARD_COM_DISPLACEMENT_M
    ):
        failures.append("forward CoM displacement below threshold")

    return {
        "passed": not failures,
        "failures": failures,
        "diagnostics": values,
        "thresholds": {
            "max_shooting_gap_norm": MAX_FINAL_SHOOTING_GAP_NORM,
            "max_active_support_translation_error_m": (
                MAX_ACTIVE_SUPPORT_TRANSLATION_ERROR_M
            ),
            "max_active_support_rotation_error_rad": (
                MAX_ACTIVE_SUPPORT_ROTATION_ERROR_RAD
            ),
            "max_landing_translation_error_m": MAX_LANDING_TRANSLATION_ERROR_M,
            "max_landing_rotation_error_rad": MAX_LANDING_ROTATION_ERROR_RAD,
            "min_swing_clearance_m": MIN_SWING_CLEARANCE_M,
            "min_joint_limit_margin_rad": MIN_JOINT_LIMIT_MARGIN_RAD,
            "min_forward_com_displacement_m": MIN_FORWARD_COM_DISPLACEMENT_M,
        },
    }


def build_static_seed(
    x0: np.ndarray, *, control_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Repeat the static initial state; never import a scripted-gait warm start."""
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    if x0.size == 0:
        raise ValueError("x0 must not be empty.")
    if int(control_dim) <= 0:
        raise ValueError("control_dim must be positive.")
    xs_seed = np.repeat(x0[None, :], HORIZON_NODES + 1, axis=0)
    us_seed = np.zeros((HORIZON_NODES, int(control_dim)), dtype=float)
    return xs_seed, us_seed


# Runtime and artifact locations.  A failed solve writes diagnostics only.
MAX_SOLVER_ITER = 500
TRAJECTORY_DIR = PROJECT_ROOT / "transfer" / "trajectories"
LOG_DIR = PROJECT_ROOT / "results" / "logs"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
TRAJECTORY_PATH = TRAJECTORY_DIR / "fddp_single_step_right_swing_ref.npz"
TRAJECTORY_SUMMARY_PATH = (
    TRAJECTORY_DIR / "fddp_single_step_right_swing_ref_summary.yaml"
)
DIAGNOSTIC_YAML_PATH = (
    LOG_DIR / "fddp_single_step_right_swing_diagnostics.yaml"
)
DIAGNOSTIC_CSV_PATH = LOG_DIR / "fddp_single_step_right_swing_diagnostics.csv"
FIGURE_PATH = FIG_DIR / "fddp_single_step_right_swing_com_clearance.png"


def yaml_safe_value(value):
    """Convert NumPy and path values recursively for safe YAML serialization."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [yaml_safe_value(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): yaml_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [yaml_safe_value(item) for item in value]
    return value


def _write_yaml(path: Path, value: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(yaml_safe_value(value), sort_keys=False), encoding="utf-8")


def _trajectory_com(model: pin.Model, xs: np.ndarray) -> np.ndarray:
    data = model.createData()
    com = np.zeros((xs.shape[0], 3), dtype=float)
    for node_index, x in enumerate(xs):
        com[node_index] = np.asarray(
            pin.centerOfMass(model, data, x[: model.nq]), dtype=float
        ).reshape(3)
    return com


def _max_active_support_errors(
    model: pin.Model, xs: np.ndarray, metadata: dict
) -> tuple[float, float, float, float]:
    data = model.createData()
    max_translation = 0.0
    max_rotation = 0.0
    landing_translation = 0.0
    landing_rotation = 0.0

    for node_index, x in enumerate(xs):
        pin.forwardKinematics(model, data, x[: model.nq])
        pin.updateFramePlacements(model, data)
        phase = metadata["phase_by_node"][
            min(node_index, HORIZON_NODES - 1)
        ]
        active = []
        if phase.right_contact_active:
            active.append(
                (
                    metadata["right_frame_id"],
                    metadata["swing_references"][node_index],
                )
            )
        if phase.left_contact_active:
            active.append(
                (metadata["left_frame_id"], metadata["left_reference"])
            )
        for frame_id, reference in active:
            actual = data.oMf[frame_id]
            max_translation = max(
                max_translation,
                float(np.linalg.norm(actual.translation - reference.translation)),
            )
            max_rotation = max(
                max_rotation,
                float(
                    np.linalg.norm(
                        pin.log3(reference.rotation.T @ actual.rotation)
                    )
                ),
            )

        if node_index == HORIZON_NODES:
            actual = data.oMf[metadata["right_frame_id"]]
            reference = metadata["swing_references"][-1]
            landing_translation = float(
                np.linalg.norm(actual.translation - reference.translation)
            )
            landing_rotation = float(
                np.linalg.norm(
                    pin.log3(reference.rotation.T @ actual.rotation)
                )
            )

    return (
        max_translation,
        max_rotation,
        landing_translation,
        landing_rotation,
    )


def compute_single_step_diagnostics(
    *,
    model: pin.Model,
    problem,
    state,
    xs: np.ndarray,
    us: np.ndarray,
    metadata: dict,
    solver,
) -> tuple[dict, np.ndarray]:
    """Measure every acceptance quantity from the solved dynamics trajectory."""
    xs_list = [np.asarray(x, dtype=float).copy() for x in xs]
    us_list = [np.asarray(u, dtype=float).copy() for u in us]
    problem.calc(xs_list, us_list)
    gaps = [
        float(
            np.linalg.norm(state.diff(xs_list[k + 1], data.xnext))
        )
        for k, data in enumerate(problem.runningDatas)
    ]
    com = _trajectory_com(model, xs)
    (
        max_support_translation,
        max_support_rotation,
        landing_translation,
        landing_rotation,
    ) = _max_active_support_errors(model, xs, metadata)

    data = model.createData()
    swing_clearance = 0.0
    for node_index in range(20, 90):
        pin.forwardKinematics(model, data, xs[node_index, : model.nq])
        pin.updateFramePlacements(model, data)
        swing_clearance = max(
            swing_clearance,
            float(
                data.oMf[metadata["right_frame_id"]].translation[2]
                - metadata["right_start"].translation[2]
            ),
        )

    joints = xs[:, 7 : model.nq]
    lower = model.lowerPositionLimit[7 : model.nq]
    upper = model.upperPositionLimit[7 : model.nq]
    minimum_margin = float(
        min(np.min(joints - lower), np.min(upper - joints))
    )
    diagnostics = {
        "max_shooting_gap_norm": float(max(gaps)),
        "rms_shooting_gap_norm": float(np.sqrt(np.mean(np.square(gaps)))),
        "max_active_support_translation_error_m": max_support_translation,
        "max_active_support_rotation_error_rad": max_support_rotation,
        "landing_translation_error_m": landing_translation,
        "landing_rotation_error_rad": landing_rotation,
        "min_swing_clearance_m": swing_clearance,
        "min_joint_limit_margin_rad": minimum_margin,
        "achieved_forward_com_displacement_m": float(
            np.max(com[:, 1]) - com[0, 1]
        ),
        "solver_return_success": bool(solver.iter < MAX_SOLVER_ITER),
        "solver_iterations": int(solver.iter),
        "solver_stop_criterion": float(solver.stop),
        "solver_cost": float(solver.cost),
    }
    return diagnostics, com


def _diagnostic_summary(
    *, diagnostics: dict, acceptance: dict, exported: bool
) -> dict:
    return {
        "stage": "fddp_single_step_right_swing",
        "task": "one right-foot swing with left-foot single support",
        "coordinate_convention": {
            "forward_axis": "+Y",
            "vertical_axis": "+Z",
            "pitch_axis": "+X",
        },
        "claim_boundary": (
            "FDDP nominal planning evidence only; tau_ref is analysis output "
            "and is not a hardware torque command."
        ),
        "exported_nominal_trajectory": bool(exported),
        "diagnostics": diagnostics,
        "acceptance": acceptance,
    }


def _write_passed_artifacts(
    *, model: pin.Model, xs: np.ndarray, us: np.ndarray, metadata: dict,
    com: np.ndarray, diagnostics: dict, acceptance: dict
) -> None:
    """Export only after every acceptance gate has passed."""
    import csv
    import matplotlib.pyplot as plt

    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    q_ref = xs[:, 7 : model.nq]
    dq_ref = xs[:, model.nq + 6 :]
    if q_ref.shape != (HORIZON_NODES + 1, 6):
        raise RuntimeError(f"Unexpected q_ref shape: {q_ref.shape}")
    if dq_ref.shape != (HORIZON_NODES + 1, 6):
        raise RuntimeError(f"Unexpected dq_ref shape: {dq_ref.shape}")

    phases = np.asarray(
        [phase.name for phase in metadata["phase_by_node"]]
        + [metadata["phase_by_node"][-1].name],
        dtype="U32",
    )
    right_active = np.asarray(
        [phase.right_contact_active for phase in metadata["phase_by_node"]]
        + [True],
        dtype=np.uint8,
    )
    left_active = np.asarray(
        [phase.left_contact_active for phase in metadata["phase_by_node"]]
        + [True],
        dtype=np.uint8,
    )
    t_ref = np.arange(HORIZON_NODES + 1, dtype=float) * DT
    np.savez(
        TRAJECTORY_PATH,
        t_ref=t_ref,
        q_ref=q_ref,
        dq_ref=dq_ref,
        tau_ref=us,
        state_ref=xs,
        com_ref_world=com,
        phase=phases,
        right_contact_active=right_active,
        left_contact_active=left_active,
        contact_mode=np.asarray("DS_left_stance_SS_DS_landing"),
        seed_type=np.asarray("static_x0_zero_control"),
        tau_ref_hardware_usage=np.asarray("analysis_only"),
    )

    summary = _diagnostic_summary(
        diagnostics=diagnostics, acceptance=acceptance, exported=True
    )
    summary["outputs"] = {
        "trajectory_npz": str(TRAJECTORY_PATH),
        "diagnostic_yaml": str(DIAGNOSTIC_YAML_PATH),
        "diagnostic_csv": str(DIAGNOSTIC_CSV_PATH),
        "figure": str(FIGURE_PATH),
    }
    _write_yaml(TRAJECTORY_SUMMARY_PATH, summary)
    _write_yaml(DIAGNOSTIC_YAML_PATH, summary)

    with DIAGNOSTIC_CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=sorted(diagnostics.keys())
        )
        writer.writeheader()
        writer.writerow(diagnostics)

    right_clearance = []
    data = model.createData()
    for x in xs:
        pin.forwardKinematics(model, data, x[: model.nq])
        pin.updateFramePlacements(model, data)
        right_clearance.append(
            data.oMf[metadata["right_frame_id"]].translation[2]
            - metadata["right_start"].translation[2]
        )
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t_ref, com[:, 1] - com[0, 1], label="CoM forward (+Y)")
    axes[0].axhline(COM_FORWARD_Y_M, color="k", linestyle="--", label="target")
    axes[0].set_ylabel("displacement (m)")
    axes[0].grid(True)
    axes[0].legend()
    axes[1].plot(t_ref, right_clearance, label="right foot clearance")
    axes[1].axhline(MIN_SWING_CLEARANCE_M, color="k", linestyle="--", label="minimum")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("clearance (m)")
    axes[1].grid(True)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def run_single_step() -> dict:
    """Solve once from static x0 and enforce the acceptance gate."""
    if not URDF_PATH.exists():
        raise FileNotFoundError(f"URDF not found: {URDF_PATH}")
    model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
    if model.nq != 13 or model.nv != 12:
        raise RuntimeError(
            f"Unexpected model dimensions nq={model.nq}, nv={model.nv}."
        )
    x0 = load_x0(X0_PATH, model.nq + model.nv)
    problem, state, metadata = build_single_step_problem(model, x0)
    xs_seed, us_seed = build_static_seed(x0, control_dim=6)
    solver = crocoddyl.SolverFDDP(problem)
    solver.th_stop = 1.0e-7
    returned_success = bool(
        solver.solve(
            [x.copy() for x in xs_seed],
            [u.copy() for u in us_seed],
            MAX_SOLVER_ITER,
            False,
        )
    )
    xs = np.asarray(solver.xs, dtype=float)
    us = np.asarray(solver.us, dtype=float)
    diagnostics, com = compute_single_step_diagnostics(
        model=model,
        problem=problem,
        state=state,
        xs=xs,
        us=us,
        metadata=metadata,
        solver=solver,
    )
    diagnostics["solver_return_success"] = returned_success
    acceptance = evaluate_acceptance(diagnostics)
    exported = bool(acceptance["passed"])
    if exported:
        _write_passed_artifacts(
            model=model,
            xs=xs,
            us=us,
            metadata=metadata,
            com=com,
            diagnostics=diagnostics,
            acceptance=acceptance,
        )
    else:
        _write_yaml(
            DIAGNOSTIC_YAML_PATH,
            _diagnostic_summary(
                diagnostics=diagnostics,
                acceptance=acceptance,
                exported=False,
            ),
        )
    return {
        "diagnostics": diagnostics,
        "acceptance": acceptance,
        "exported": exported,
    }


def main() -> int:
    result = run_single_step()
    print("FDDP single-step right swing acceptance:")
    for name, value in result["diagnostics"].items():
        print(f"  {name}: {value}")
    if not result["acceptance"]["passed"]:
        print("Nominal export withheld:")
        for failure in result["acceptance"]["failures"]:
            print(f"  - {failure}")
        return 1
    print(f"Saved nominal trajectory: {TRAJECTORY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
