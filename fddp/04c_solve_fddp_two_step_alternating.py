#!/usr/bin/env python3
"""Fixed-contact helpers for a minimal two-step alternating FDDP experiment.

The experiment is DS -> left swing -> DS -> right swing -> DS. It is a
simulation feasibility step, not continuous gait or a hardware controller.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import crocoddyl
import numpy as np
import pinocchio as pin

DT = 0.010
HORIZON_NODES = 200
STEP_FORWARD_Y_M = 0.010
SWING_CLEARANCE_Z_M = 0.020
LEFT_SWING_START_NODE = 20
LEFT_SWING_NODE_COUNT = 70
RIGHT_SWING_START_NODE = 110
RIGHT_SWING_NODE_COUNT = 70


@dataclass(frozen=True)
class PhaseSpec:
    """One fixed contact interval; fields are ordered right then left."""

    name: str
    node_count: int
    right_contact_active: bool
    left_contact_active: bool


def build_phase_schedule() -> list[PhaseSpec]:
    """Return the validated DS-left-SS-DS-right-SS-DS schedule."""
    return [
        PhaseSpec("double_support_start", 20, True, True),
        PhaseSpec("right_stance_left_swing", 70, True, False),
        PhaseSpec("double_support_mid", 20, True, True),
        PhaseSpec("left_stance_right_swing", 70, False, True),
        PhaseSpec("double_support_landing", 20, True, True),
    ]


def validate_configuration(
    *,
    dt_s: float = DT,
    step_forward_m: float = STEP_FORWARD_Y_M,
    clearance_m: float = SWING_CLEARANCE_Z_M,
) -> None:
    """Reject timing and swing values outside the safety contract."""
    if float(dt_s) <= 0.0:
        raise ValueError("DT must be positive.")
    if sum(phase.node_count for phase in build_phase_schedule()) != HORIZON_NODES:
        raise ValueError("Phase node counts must equal HORIZON_NODES.")
    if float(step_forward_m) <= 0.0:
        raise ValueError("STEP_FORWARD_Y_M must be positive.")
    if float(clearance_m) < 0.010:
        raise ValueError("SWING_CLEARANCE_Z_M must be at least 0.010 m.")


def quintic_progress(s: float) -> float:
    """Return clipped C2-continuous progression in [0, 1]."""
    value = float(np.clip(s, 0.0, 1.0))
    return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)


SWING_TIME_WARP_EXPONENT = 0.75


def swing_progress(s: float) -> float:
    """C2-continuous symmetric progress with faster takeoff and landing."""
    value = float(np.clip(s, 0.0, 1.0))
    if value in (0.0, 1.0):
        return value
    forward = value**SWING_TIME_WARP_EXPONENT
    backward = (1.0 - value) ** SWING_TIME_WARP_EXPONENT
    return quintic_progress(forward / (forward + backward))


def _single_swing_reference(
    start: pin.SE3, swing_start_node: int, swing_node_count: int
) -> list[pin.SE3]:
    poses: list[pin.SE3] = []
    for node_index in range(HORIZON_NODES + 1):
        progress = swing_progress(
            (node_index - swing_start_node) / swing_node_count
        )
        translation = start.translation.copy()
        translation[1] += STEP_FORWARD_Y_M * progress
        translation[2] += (
            SWING_CLEARANCE_Z_M * 4.0 * progress * (1.0 - progress)
        )
        poses.append(pin.SE3(start.rotation.copy(), translation))
    return poses


def swing_reference_poses(
    left_start: pin.SE3, right_start: pin.SE3
) -> tuple[list[pin.SE3], list[pin.SE3]]:
    """Return world-frame left-then-right swing references for all nodes."""
    validate_configuration()
    return (
        _single_swing_reference(
            left_start, LEFT_SWING_START_NODE, LEFT_SWING_NODE_COUNT
        ),
        _single_swing_reference(
            right_start, RIGHT_SWING_START_NODE, RIGHT_SWING_NODE_COUNT
        ),
    )


COM_FORWARD_Y_M = 0.008
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
JOINT_BARRIER_MARGIN_RAD = 1.2e-2

LEFT_FOOT_FRAME = "left_foot_contact_fixed"
RIGHT_FOOT_FRAME = "right_foot_contact_fixed"
LEFT_CONTACT_NAME = "left_foot_6d"
RIGHT_CONTACT_NAME = "right_foot_6d"
LEFT_SUPPORT_COST_NAME = "left_support_se3_tracking"
RIGHT_SUPPORT_COST_NAME = "right_support_se3_tracking"
LEFT_SWING_COST_NAME = "left_swing_se3_tracking"
RIGHT_SWING_COST_NAME = "right_swing_se3_tracking"
JOINT_SAFETY_COST_NAME = "joint_safety_barrier"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = PROJECT_ROOT / "model" / "biped_v1.urdf"
X0_PATH = PROJECT_ROOT / "transfer" / "initialisation" / "fddp_double_support_x0.npz"


def load_x0(path: Path, expected_size: int) -> np.ndarray:
    """Load the static double-support seed; never a scripted gait warm start."""
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
    model: pin.Model, q0: np.ndarray, left_frame_id: int, right_frame_id: int
) -> tuple[pin.SE3, pin.SE3]:
    data = model.createData()
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)
    return data.oMf[left_frame_id].copy(), data.oMf[right_frame_id].copy()


def _com_targets(model: pin.Model, x0: np.ndarray) -> np.ndarray:
    data = model.createData()
    initial_com = np.asarray(
        pin.centerOfMass(model, data, x0[:model.nq]), dtype=float
    ).reshape(3)
    targets = np.zeros((HORIZON_NODES + 1, 3), dtype=float)
    for node_index in range(HORIZON_NODES + 1):
        target = initial_com.copy()
        target[1] += COM_FORWARD_Y_M * quintic_progress(
            node_index / HORIZON_NODES
        )
        targets[node_index] = target
    return targets


def _joint_safety_activation(model: pin.Model, state, x0: np.ndarray):
    lower = np.full(state.ndx, -np.inf, dtype=float)
    upper = np.full(state.ndx, np.inf, dtype=float)
    tangent = slice(6, 6 + (model.nv - 6))
    q_joint = x0[7:model.nq]
    lower[tangent] = (
        model.lowerPositionLimit[7:model.nq] + JOINT_BARRIER_MARGIN_RAD - q_joint
    )
    upper[tangent] = (
        model.upperPositionLimit[7:model.nq] - JOINT_BARRIER_MARGIN_RAD - q_joint
    )
    if np.any(lower[tangent] >= upper[tangent]):
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
    left_frame_id: int,
    right_frame_id: int,
    left_reference: pin.SE3,
    right_reference: pin.SE3,
    joint_safety_activation,
    dt_s: float,
    enable_force_jacobians: bool = False,
):
    """Create one contact-dynamics model with support and swing costs."""
    nu = actuation.nu
    contacts = crocoddyl.ContactModelMultiple(state, nu)
    if phase.left_contact_active:
        contacts.addContact(
            LEFT_CONTACT_NAME,
            crocoddyl.ContactModel6D(
                state, left_frame_id, left_reference,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED, nu, CONTACT_GAINS_KP_KD,
            ),
        )
    if phase.right_contact_active:
        contacts.addContact(
            RIGHT_CONTACT_NAME,
            crocoddyl.ContactModel6D(
                state, right_frame_id, right_reference,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED, nu, CONTACT_GAINS_KP_KD,
            ),
        )
    costs = crocoddyl.CostModelSum(state, nu)
    state_residual = crocoddyl.ResidualModelState(state, x_reference, nu)
    costs.addCost(
        "static_x0_state_regularisation",
        crocoddyl.CostModelResidual(state, state_residual),
        TERMINAL_STATE_WEIGHT if dt_s == 0.0 else RUNNING_STATE_WEIGHT,
    )
    safety_residual = crocoddyl.ResidualModelState(state, x_reference, nu)
    costs.addCost(
        JOINT_SAFETY_COST_NAME,
        crocoddyl.CostModelResidual(state, joint_safety_activation, safety_residual),
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
    if phase.left_contact_active:
        residual = crocoddyl.ResidualModelFramePlacement(
            state, left_frame_id, left_reference, nu
        )
        costs.addCost(
            LEFT_SUPPORT_COST_NAME,
            crocoddyl.CostModelResidual(state, residual),
            SUPPORT_TRACKING_WEIGHT,
        )
    else:
        residual = crocoddyl.ResidualModelFramePlacement(
            state, left_frame_id, left_reference, nu
        )
        costs.addCost(
            LEFT_SWING_COST_NAME,
            crocoddyl.CostModelResidual(state, residual),
            SWING_TRACKING_WEIGHT,
        )
    if phase.right_contact_active:
        residual = crocoddyl.ResidualModelFramePlacement(
            state, right_frame_id, right_reference, nu
        )
        costs.addCost(
            RIGHT_SUPPORT_COST_NAME,
            crocoddyl.CostModelResidual(state, residual),
            SUPPORT_TRACKING_WEIGHT,
        )
    else:
        residual = crocoddyl.ResidualModelFramePlacement(
            state, right_frame_id, right_reference, nu
        )
        costs.addCost(
            RIGHT_SWING_COST_NAME,
            crocoddyl.CostModelResidual(state, residual),
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
        state,
        actuation,
        contacts,
        costs,
        JM_INV_JT_DAMPING,
        bool(enable_force_jacobians),
    )
    return crocoddyl.IntegratedActionModelEuler(differential, dt_s)


def build_two_step_problem(model: pin.Model, x0: np.ndarray):
    """Build fixed DS-left-SS-DS-right-SS-DS FDDP from static x0."""
    validate_configuration()
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    if x0.size != model.nq + model.nv:
        raise ValueError(f"x0 size {x0.size}; expected {model.nq + model.nv}.")
    left_frame_id = model.getFrameId(LEFT_FOOT_FRAME)
    right_frame_id = model.getFrameId(RIGHT_FOOT_FRAME)
    if left_frame_id >= model.nframes or right_frame_id >= model.nframes:
        raise RuntimeError("Could not resolve both flat-foot contact frames.")
    left_start, right_start = _frame_references(
        model, x0[:model.nq], left_frame_id, right_frame_id
    )
    left_references, right_references = swing_reference_poses(
        left_start, right_start
    )
    com_targets = _com_targets(model, x0)
    phases = build_phase_schedule()
    phase_by_node = [phase for phase in phases for _ in range(phase.node_count)]
    if len(phase_by_node) != HORIZON_NODES:
        raise RuntimeError("Expanded phase schedule has an invalid node count.")
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    safety_activation = _joint_safety_activation(model, state, x0)
    running_models = [
        _build_action_model(
            state=state, actuation=actuation, phase=phase,
            x_reference=x0, com_target=com_targets[node_index],
            left_frame_id=left_frame_id, right_frame_id=right_frame_id,
            left_reference=left_references[node_index],
            right_reference=right_references[node_index],
            joint_safety_activation=safety_activation, dt_s=DT,
        )
        for node_index, phase in enumerate(phase_by_node)
    ]
    terminal_model = _build_action_model(
        state=state, actuation=actuation, phase=phases[-1],
        x_reference=x0, com_target=com_targets[-1],
        left_frame_id=left_frame_id, right_frame_id=right_frame_id,
        left_reference=left_references[-1], right_reference=right_references[-1],
        joint_safety_activation=safety_activation, dt_s=0.0,
    )
    metadata = {
        "left_frame_id": left_frame_id,
        "right_frame_id": right_frame_id,
        "left_start": left_start,
        "right_start": right_start,
        "left_references": left_references,
        "right_references": right_references,
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
MIN_FORWARD_COM_DISPLACEMENT_M = 0.002


def evaluate_acceptance(diagnostics: dict) -> dict:
    """Return the all-gates decision for a two-swing nominal export."""
    failures: list[str] = []
    if diagnostics["max_shooting_gap_norm"] > MAX_FINAL_SHOOTING_GAP_NORM:
        failures.append("shooting gap exceeds threshold")
    if diagnostics["max_active_support_translation_error_m"] > MAX_ACTIVE_SUPPORT_TRANSLATION_ERROR_M:
        failures.append("active support translation error exceeds threshold")
    if diagnostics["max_active_support_rotation_error_rad"] > MAX_ACTIVE_SUPPORT_ROTATION_ERROR_RAD:
        failures.append("active support rotation error exceeds threshold")
    for landing in ("first", "second"):
        if diagnostics[f"{landing}_landing_translation_error_m"] > MAX_LANDING_TRANSLATION_ERROR_M:
            failures.append(f"{landing} landing translation error exceeds threshold")
        if diagnostics[f"{landing}_landing_rotation_error_rad"] > MAX_LANDING_ROTATION_ERROR_RAD:
            failures.append(f"{landing} landing rotation error exceeds threshold")
    if diagnostics["min_left_swing_clearance_m"] < MIN_SWING_CLEARANCE_M:
        failures.append("left swing clearance below threshold")
    if diagnostics["min_right_swing_clearance_m"] < MIN_SWING_CLEARANCE_M:
        failures.append("right swing clearance below threshold")
    if diagnostics["min_joint_limit_margin_rad"] < MIN_JOINT_LIMIT_MARGIN_RAD:
        failures.append("joint limit margin below threshold")
    if diagnostics["achieved_forward_com_displacement_m"] < MIN_FORWARD_COM_DISPLACEMENT_M:
        failures.append("forward CoM displacement below threshold")
    return {
        "passed": not failures,
        "failures": failures,
        "diagnostics": dict(diagnostics),
    }


def build_static_seed(
    x0: np.ndarray, control_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return static-state, zero-control initialisation without gait warm start."""
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    return (
        np.repeat(x0[None, :], HORIZON_NODES + 1, axis=0),
        np.zeros((HORIZON_NODES, control_dim), dtype=float),
    )


MAX_SOLVER_ITER = 500
TRAJECTORY_DIR = PROJECT_ROOT / "transfer" / "trajectories"
LOG_DIR = PROJECT_ROOT / "results" / "logs"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
TRAJECTORY_PATH = TRAJECTORY_DIR / "fddp_two_step_alternating_ref.npz"
TRAJECTORY_SUMMARY_PATH = TRAJECTORY_DIR / "fddp_two_step_alternating_ref_summary.yaml"
DIAGNOSTIC_YAML_PATH = LOG_DIR / "fddp_two_step_alternating_diagnostics.yaml"
DIAGNOSTIC_CSV_PATH = LOG_DIR / "fddp_two_step_alternating_diagnostics.csv"
FIGURE_PATH = FIG_DIR / "fddp_two_step_alternating_com_clearance.png"


def _trajectory_com(model: pin.Model, xs: np.ndarray) -> np.ndarray:
    data = model.createData()
    com = np.zeros((xs.shape[0], 3), dtype=float)
    for index, x in enumerate(xs):
        com[index] = np.asarray(
            pin.centerOfMass(model, data, x[:model.nq]), dtype=float
        ).reshape(3)
    return com


def _placement_error(actual: pin.SE3, reference: pin.SE3) -> tuple[float, float]:
    return (
        float(np.linalg.norm(actual.translation - reference.translation)),
        float(np.linalg.norm(pin.log3(reference.rotation.T @ actual.rotation))),
    )


def _support_and_landing_errors(
    model: pin.Model, xs: np.ndarray, metadata: dict
) -> tuple[float, float, float, float, float, float]:
    data = model.createData()
    max_translation = 0.0
    max_rotation = 0.0
    first_landing = (0.0, 0.0)
    second_landing = (0.0, 0.0)
    for node_index, x in enumerate(xs):
        pin.forwardKinematics(model, data, x[:model.nq])
        pin.updateFramePlacements(model, data)
        phase = metadata["phase_by_node"][min(node_index, HORIZON_NODES - 1)]
        if phase.left_contact_active:
            translation, rotation = _placement_error(
                data.oMf[metadata["left_frame_id"]],
                metadata["left_references"][node_index],
            )
            max_translation = max(max_translation, translation)
            max_rotation = max(max_rotation, rotation)
        if phase.right_contact_active:
            translation, rotation = _placement_error(
                data.oMf[metadata["right_frame_id"]],
                metadata["right_references"][node_index],
            )
            max_translation = max(max_translation, translation)
            max_rotation = max(max_rotation, rotation)
        if node_index == 90:
            first_landing = _placement_error(
                data.oMf[metadata["left_frame_id"]],
                metadata["left_references"][node_index],
            )
        if node_index == HORIZON_NODES:
            second_landing = _placement_error(
                data.oMf[metadata["right_frame_id"]],
                metadata["right_references"][node_index],
            )
    return (
        max_translation, max_rotation,
        first_landing[0], first_landing[1],
        second_landing[0], second_landing[1],
    )


def _swing_clearances(
    model: pin.Model, xs: np.ndarray, metadata: dict
) -> tuple[float, float]:
    data = model.createData()
    left_clearance = 0.0
    right_clearance = 0.0
    for node_index, x in enumerate(xs):
        pin.forwardKinematics(model, data, x[:model.nq])
        pin.updateFramePlacements(model, data)
        if LEFT_SWING_START_NODE <= node_index <= LEFT_SWING_START_NODE + LEFT_SWING_NODE_COUNT:
            left_clearance = max(
                left_clearance,
                float(
                    data.oMf[metadata["left_frame_id"]].translation[2]
                    - metadata["left_start"].translation[2]
                ),
            )
        if RIGHT_SWING_START_NODE <= node_index <= RIGHT_SWING_START_NODE + RIGHT_SWING_NODE_COUNT:
            right_clearance = max(
                right_clearance,
                float(
                    data.oMf[metadata["right_frame_id"]].translation[2]
                    - metadata["right_start"].translation[2]
                ),
            )
    return left_clearance, right_clearance


def compute_two_step_diagnostics(
    *, model: pin.Model, problem, state, xs: np.ndarray, us: np.ndarray,
    metadata: dict, solver
) -> tuple[dict, np.ndarray]:
    """Measure every two-step gate from the solved dynamics trajectory."""
    xs_list = [np.asarray(x, dtype=float).copy() for x in xs]
    us_list = [np.asarray(u, dtype=float).copy() for u in us]
    problem.calc(xs_list, us_list)
    gaps = [
        float(np.linalg.norm(state.diff(xs_list[k + 1], data.xnext)))
        for k, data in enumerate(problem.runningDatas)
    ]
    com = _trajectory_com(model, xs)
    (
        support_translation, support_rotation,
        first_translation, first_rotation,
        second_translation, second_rotation,
    ) = _support_and_landing_errors(model, xs, metadata)
    left_clearance, right_clearance = _swing_clearances(model, xs, metadata)
    joints = xs[:, 7:model.nq]
    lower = model.lowerPositionLimit[7:model.nq]
    upper = model.upperPositionLimit[7:model.nq]
    diagnostics = {
        "max_shooting_gap_norm": float(max(gaps)),
        "rms_shooting_gap_norm": float(np.sqrt(np.mean(np.square(gaps)))),
        "max_active_support_translation_error_m": support_translation,
        "max_active_support_rotation_error_rad": support_rotation,
        "first_landing_translation_error_m": first_translation,
        "first_landing_rotation_error_rad": first_rotation,
        "second_landing_translation_error_m": second_translation,
        "second_landing_rotation_error_rad": second_rotation,
        "min_left_swing_clearance_m": left_clearance,
        "min_right_swing_clearance_m": right_clearance,
        "min_joint_limit_margin_rad": float(
            min(np.min(joints - lower), np.min(upper - joints))
        ),
        "achieved_forward_com_displacement_m": float(
            np.max(com[:, 1]) - com[0, 1]
        ),
        "solver_return_success": bool(solver.iter < MAX_SOLVER_ITER),
        "solver_iterations": int(solver.iter),
        "solver_stop_criterion": float(solver.stop),
        "solver_cost": float(solver.cost),
    }
    return diagnostics, com


def _write_yaml(path: Path, value: dict) -> None:
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _summary(diagnostics: dict, acceptance: dict, exported: bool) -> dict:
    return {
        "stage": "fddp_two_step_alternating",
        "task": "fixed DS-left-SS-DS-right-SS-DS two-step feasibility experiment",
        "coordinate_convention": {
            "forward_axis": "+Y", "vertical_axis": "+Z", "pitch_axis": "+X"
        },
        "claim_boundary": (
            "Simulation FDDP planning evidence only; tau_ref is analysis output "
            "and is not a hardware torque command or a hardware walking claim."
        ),
        "exported_nominal_trajectory": bool(exported),
        "diagnostics": diagnostics,
        "acceptance": acceptance,
    }


def _write_passed_artifacts(
    *, model: pin.Model, xs: np.ndarray, us: np.ndarray, metadata: dict,
    com: np.ndarray, diagnostics: dict, acceptance: dict
) -> None:
    import csv
    import matplotlib.pyplot as plt
    for directory in (TRAJECTORY_DIR, LOG_DIR, FIG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    q_ref = xs[:, 7:model.nq]
    dq_ref = xs[:, model.nq + 6:]
    if q_ref.shape != (HORIZON_NODES + 1, 6):
        raise RuntimeError(f"Unexpected q_ref shape: {q_ref.shape}")
    if dq_ref.shape != (HORIZON_NODES + 1, 6):
        raise RuntimeError(f"Unexpected dq_ref shape: {dq_ref.shape}")
    phases = np.asarray(
        [phase.name for phase in metadata["phase_by_node"]]
        + [metadata["phase_by_node"][-1].name], dtype="U32"
    )
    np.savez(
        TRAJECTORY_PATH,
        t_ref=np.arange(HORIZON_NODES + 1, dtype=float) * DT,
        q_ref=q_ref, dq_ref=dq_ref, tau_ref=us, state_ref=xs,
        com_ref_world=com, phase=phases,
        left_contact_active=np.asarray(
            [phase.left_contact_active for phase in metadata["phase_by_node"]]+[True],
            dtype=np.uint8,
        ),
        right_contact_active=np.asarray(
            [phase.right_contact_active for phase in metadata["phase_by_node"]]+[True],
            dtype=np.uint8,
        ),
        contact_mode=np.asarray("DS_right_stance_left_swing_DS_left_stance_right_swing_DS"),
        seed_type=np.asarray("static_x0_zero_control"),
        tau_ref_hardware_usage=np.asarray("analysis_only"),
    )
    summary = _summary(diagnostics, acceptance, True)
    summary["outputs"] = {
        "trajectory_npz": str(TRAJECTORY_PATH),
        "diagnostic_yaml": str(DIAGNOSTIC_YAML_PATH),
        "diagnostic_csv": str(DIAGNOSTIC_CSV_PATH),
        "figure": str(FIGURE_PATH),
    }
    _write_yaml(TRAJECTORY_SUMMARY_PATH, summary)
    _write_yaml(DIAGNOSTIC_YAML_PATH, summary)
    with DIAGNOSTIC_CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(diagnostics))
        writer.writeheader()
        writer.writerow(diagnostics)
    left_clearance = []
    right_clearance = []
    data = model.createData()
    for x in xs:
        pin.forwardKinematics(model, data, x[:model.nq])
        pin.updateFramePlacements(model, data)
        left_clearance.append(
            data.oMf[metadata["left_frame_id"]].translation[2]
            - metadata["left_start"].translation[2]
        )
        right_clearance.append(
            data.oMf[metadata["right_frame_id"]].translation[2]
            - metadata["right_start"].translation[2]
        )
    t_ref = np.arange(HORIZON_NODES + 1, dtype=float) * DT
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t_ref, com[:, 1] - com[0, 1], label="CoM forward (+Y)")
    axes[0].axhline(COM_FORWARD_Y_M, color="k", linestyle="--", label="target")
    axes[0].grid(True); axes[0].legend()
    axes[1].plot(t_ref, left_clearance, label="left clearance")
    axes[1].plot(t_ref, right_clearance, label="right clearance")
    axes[1].axhline(MIN_SWING_CLEARANCE_M, color="k", linestyle="--", label="minimum")
    axes[1].set_xlabel("time (s)"); axes[1].grid(True); axes[1].legend()
    figure.tight_layout(); figure.savefig(FIGURE_PATH, dpi=180); plt.close(figure)


def run_two_step() -> dict:
    """Solve from static x0 and export only if every acceptance gate passes."""
    model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
    x0 = load_x0(X0_PATH, model.nq + model.nv)
    problem, state, metadata = build_two_step_problem(model, x0)
    xs_seed, us_seed = build_static_seed(x0, control_dim=6)
    solver = crocoddyl.SolverFDDP(problem)
    solver.th_stop = 1.0e-7
    returned_success = bool(solver.solve(
        [x.copy() for x in xs_seed], [u.copy() for u in us_seed],
        MAX_SOLVER_ITER, False
    ))
    xs = np.asarray(solver.xs, dtype=float)
    us = np.asarray(solver.us, dtype=float)
    diagnostics, com = compute_two_step_diagnostics(
        model=model, problem=problem, state=state, xs=xs, us=us,
        metadata=metadata, solver=solver
    )
    diagnostics["solver_return_success"] = returned_success
    acceptance = evaluate_acceptance(diagnostics)
    if acceptance["passed"]:
        _write_passed_artifacts(
            model=model, xs=xs, us=us, metadata=metadata, com=com,
            diagnostics=diagnostics, acceptance=acceptance
        )
    else:
        _write_yaml(DIAGNOSTIC_YAML_PATH, _summary(diagnostics, acceptance, False))
    return {"diagnostics": diagnostics, "acceptance": acceptance,
            "exported": bool(acceptance["passed"])}


def main() -> int:
    result = run_two_step()
    print("FDDP two-step alternating acceptance:")
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
