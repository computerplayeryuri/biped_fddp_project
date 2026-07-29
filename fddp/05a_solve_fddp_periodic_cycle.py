"""Periodic two-step FDDP cycle with boundary-continuity gating.

The first TDD slice in this module is intentionally pure: it defines the
translated state reference, boundary metrics, acceptance contract, and failed
artifact behaviour. Solver construction is added behind the same gates.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import crocoddyl
import numpy as np
import pinocchio as pin
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "logs"
DEFAULT_TRAJECTORY_DIR = PROJECT_ROOT / "transfer" / "trajectories"
URDF_PATH = PROJECT_ROOT / "model" / "biped_v1.urdf"
X0_PATH = PROJECT_ROOT / "transfer" / "initialisation" / "fddp_double_support_x0.npz"
TWO_STEP_MODULE_PATH = PROJECT_ROOT / "fddp" / "04c_solve_fddp_two_step_alternating.py"
TWO_STEP_TRAJECTORY_PATH = (
    PROJECT_ROOT / "transfer" / "trajectories" / "fddp_two_step_alternating_ref.npz"
)
TWO_STEP_SUMMARY_PATH = (
    PROJECT_ROOT
    / "transfer"
    / "trajectories"
    / "fddp_two_step_alternating_ref_summary.yaml"
)
CONTACT_CONFIG_PATH = PROJECT_ROOT / "config" / "fddp_multistep_experiment.yaml"
CONTACT_FEASIBILITY_MODULE_PATH = PROJECT_ROOT / "fddp" / "contact_feasibility.py"

LEFT_CONTACT_NAME = "left_foot_6d"
RIGHT_CONTACT_NAME = "right_foot_6d"
LEFT_SOLE_FRAME = "left_sole_contact"
RIGHT_SOLE_FRAME = "right_sole_contact"
SOLE_OFFSET_FROM_ANKLE_LOCAL_M = np.array([0.0, 0.0, -0.043], dtype=float)
FOOT_HALF_LENGTH_M = 0.0725
FRICTION_COEFFICIENT = 0.80
MAXIMUM_FRICTION_UTILISATION = 0.90
MINIMUM_COP_MARGIN_M = 0.002
OPTIMISATION_COP_MARGIN_M = 0.0021
EFFECTIVE_PLANAR_MU = FRICTION_COEFFICIENT * MAXIMUM_FRICTION_UTILISATION
PLANAR_SUPPORT_CONE_WEIGHT = 1.0e3
PLANAR_SUPPORT_CONE_COST_NAME = "planar_support_resultant_cone"
PERIODIC_BOUNDARY_CONTROL_COST_NAME = "periodic_boundary_control"

PLANNED_CYCLE_FORWARD_Y_M = 0.010
HORIZON_NODES = 200
DT = 0.010

CYCLE_GATE_LIMITS = {
    "joint_position_difference_rad": ("max", 1.0e-3),
    "joint_velocity_difference_rad_s": ("max", 1.0e-2),
    "base_height_difference_m": ("max", 1.0e-3),
    "base_pitch_difference_rad": ("max", 1.0e-3),
    "com_velocity_difference_m_s": ("max", 1.0e-3),
    "relative_foot_translation_error_m": ("max", 1.0e-3),
    "relative_foot_rotation_error_rad": ("max", np.deg2rad(0.2)),
    "control_discontinuity": ("max", 0.10),
    "solver_stop": ("max", 1.0e-7),
    "shooting_gap_norm": ("max", 1.0e-7),
    "active_support_translation_error_m": ("max", 5.0e-4),
    "active_support_rotation_error_rad": ("max", np.deg2rad(0.5)),
    "landing_translation_error_m": ("max", 1.0e-3),
    "landing_rotation_error_rad": ("max", np.deg2rad(1.0)),
    "swing_clearance_m": ("min", 1.0e-2),
    "joint_limit_margin_rad": ("min", 1.0e-2),
    "contact_support_node_count": ("min", float(HORIZON_NODES)),
    "contact_violation_count": ("max", 0.0),
    "minimum_contact_normal_force_n": ("min", -1.0e-5),
    "maximum_friction_utilisation": ("max", 0.90),
    "minimum_cop_margin_m": ("min", 0.002),
    "maximum_tangential_speed_m_s": ("max", 0.001),
}


@dataclass(frozen=True)
class HomotopyStage:
    name: str
    terminal_state_weight: float
    boundary_control_weight: float


class PlanarSupportConeResidualData(crocoddyl.ResidualDataAbstract):
    def __init__(self, model, shared):
        crocoddyl.ResidualDataAbstract.__init__(self, model, shared)


class ResidualModelPlanarSupportCone(crocoddyl.ResidualModelAbstract):
    """Non-negative sagittal resultant cone for the active sole contacts."""

    def __init__(self, state, active_contacts, *, nu: int):
        if not active_contacts:
            raise ValueError("planar support cone requires an active contact")
        self.active_contact_names = tuple(name for name, _ in active_contacts)
        self.support_positions_world_m = {
            name: np.asarray(position, dtype=float).reshape(3).copy()
            for name, position in active_contacts
        }
        self.support_min_y_m = min(
            position[1] - FOOT_HALF_LENGTH_M
            for position in self.support_positions_world_m.values()
        )
        self.support_max_y_m = max(
            position[1] + FOOT_HALF_LENGTH_M
            for position in self.support_positions_world_m.values()
        )
        self.inner_support_min_y_m = (
            self.support_min_y_m + OPTIMISATION_COP_MARGIN_M
        )
        self.inner_support_max_y_m = (
            self.support_max_y_m - OPTIMISATION_COP_MARGIN_M
        )
        crocoddyl.ResidualModelAbstract.__init__(
            self, state, 5, nu, True, True, True
        )

    def createData(self, shared):
        return PlanarSupportConeResidualData(self, shared)

    def _contact_data(self, data):
        contacts = data.shared.contacts.contacts.todict()
        missing = [name for name in self.active_contact_names if name not in contacts]
        if missing:
            raise RuntimeError(f"planar support residual missing contacts: {missing}")
        return contacts

    def _cone_matrix(self, name: str) -> np.ndarray:
        y, z = self.support_positions_world_m[name][1:]
        return np.array(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, EFFECTIVE_PLANAR_MU, 0.0, 0.0, 0.0],
                [0.0, 1.0, EFFECTIVE_PLANAR_MU, 0.0, 0.0, 0.0],
                [0.0, -z, y - self.inner_support_min_y_m, 1.0, 0.0, 0.0],
                [0.0, z, self.inner_support_max_y_m - y, -1.0, 0.0, 0.0],
            ],
            dtype=float,
        )

    def calc(self, data, x, u):
        contacts = self._contact_data(data)
        data.r[:] = 0.0
        for name in self.active_contact_names:
            data.r[:] += self._cone_matrix(name) @ np.asarray(
                contacts[name].f.vector, dtype=float
            )

    def calcDiff(self, data, x, u):
        contacts = self._contact_data(data)
        data.Rx[:, :] = 0.0
        data.Ru[:, :] = 0.0
        for name in self.active_contact_names:
            matrix = self._cone_matrix(name)
            data.Rx[:, :] += matrix @ np.asarray(contacts[name].df_dx, dtype=float)
            data.Ru[:, :] += matrix @ np.asarray(contacts[name].df_du, dtype=float)


def build_periodic_homotopy_stages() -> tuple[HomotopyStage, ...]:
    return (
        HomotopyStage("terminal_state_1", 1.0, 0.0),
        HomotopyStage("terminal_state_10", 10.0, 0.0),
        HomotopyStage("terminal_state_100", 100.0, 0.0),
        HomotopyStage("terminal_state_1000", 1000.0, 0.0),
        HomotopyStage("boundary_control_1", 1000.0, 1.0),
        HomotopyStage("boundary_control_10", 1000.0, 10.0),
        HomotopyStage("boundary_control_30", 1000.0, 30.0),
    )


def _quintic_progress(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def build_periodic_state_references(
    x0: np.ndarray,
    *,
    nq: int,
    horizon_nodes: int = HORIZON_NODES,
    planned_forward_y: float = PLANNED_CYCLE_FORWARD_Y_M,
) -> np.ndarray:
    """Translate the static full state while preserving its periodic posture."""

    x0 = np.asarray(x0, dtype=float).reshape(-1)
    if nq < 7 or x0.size <= nq:
        raise ValueError("x0 must contain a free-flyer configuration and velocity")
    if horizon_nodes <= 0:
        raise ValueError("horizon_nodes must be positive")
    if planned_forward_y <= 0.0 or not np.isfinite(planned_forward_y):
        raise ValueError("planned_forward_y must be finite and positive")
    if not np.all(np.isfinite(x0)):
        raise ValueError("x0 must be finite")
    if np.max(np.abs(x0[nq:])) > 1.0e-12:
        raise ValueError("periodic cycle must start from a static x0")

    references = np.repeat(x0[None, :], horizon_nodes + 1, axis=0)
    for node in range(horizon_nodes + 1):
        progress = _quintic_progress(node / horizon_nodes)
        references[node, 1] += progress * planned_forward_y
    references[:, nq:] = 0.0
    return references


def _load_two_step_module():
    module_name = "_fddp_periodic_two_step_base"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, TWO_STEP_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load two-step FDDP module: {TWO_STEP_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_contact_feasibility_module():
    module_name = "_fddp_periodic_contact_feasibility"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name, CONTACT_FEASIBILITY_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"cannot load contact feasibility module: {CONTACT_FEASIBILITY_MODULE_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_x0(path: Path, expected_size: int) -> np.ndarray:
    return _load_two_step_module().load_x0(Path(path), expected_size)


def ensure_sole_contact_frames(model: pin.Model) -> tuple[int, int]:
    """Add idempotent operational frames at the URDF/MJCF sole plane."""

    base = _load_two_step_module()
    frame_ids = []
    for source_name, sole_name in (
        (base.LEFT_FOOT_FRAME, LEFT_SOLE_FRAME),
        (base.RIGHT_FOOT_FRAME, RIGHT_SOLE_FRAME),
    ):
        existing = model.getFrameId(sole_name)
        if existing < model.nframes:
            frame = model.frames[existing]
            if not np.allclose(
                frame.placement.translation,
                SOLE_OFFSET_FROM_ANKLE_LOCAL_M,
                atol=1.0e-12,
            ):
                raise RuntimeError(f"{sole_name} exists at an incompatible placement")
            frame_ids.append(int(existing))
            continue
        source_id = model.getFrameId(source_name)
        if source_id >= model.nframes:
            raise RuntimeError(f"could not resolve source contact frame {source_name}")
        source = model.frames[source_id]
        sole = pin.Frame(
            sole_name,
            source.parentJoint,
            source_id,
            pin.SE3(np.eye(3), SOLE_OFFSET_FROM_ANKLE_LOCAL_M.copy()),
            pin.FrameType.OP_FRAME,
        )
        frame_ids.append(int(model.addFrame(sole, False)))
    return tuple(frame_ids)


def _add_contact_feasibility_cost(
    *, action_model, state, phase, left_reference: pin.SE3, right_reference: pin.SE3
) -> None:
    active_contacts = []
    if phase.left_contact_active:
        active_contacts.append((LEFT_CONTACT_NAME, left_reference.translation))
    if phase.right_contact_active:
        active_contacts.append((RIGHT_CONTACT_NAME, right_reference.translation))
    residual = ResidualModelPlanarSupportCone(
        state, active_contacts, nu=action_model.differential.actuation.nu
    )
    activation = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(
            np.zeros(5, dtype=float), np.full(5, np.inf, dtype=float)
        )
    )
    action_model.differential.costs.addCost(
        PLANAR_SUPPORT_CONE_COST_NAME,
        crocoddyl.CostModelResidual(state, activation, residual),
        PLANAR_SUPPORT_CONE_WEIGHT,
    )


def build_periodic_cycle_problem(
    model: pin.Model,
    x0: np.ndarray,
    config=None,
    *,
    terminal_state_weight: float = 1000.0,
    boundary_control_weight: float = 30.0,
):
    """Build one translated-periodic DS-left-SS-DS-right-SS-DS problem."""

    del config
    terminal_state_weight = float(terminal_state_weight)
    boundary_control_weight = float(boundary_control_weight)
    if not np.isfinite(terminal_state_weight) or terminal_state_weight <= 0.0:
        raise ValueError("terminal_state_weight must be finite and positive")
    if not np.isfinite(boundary_control_weight) or boundary_control_weight < 0.0:
        raise ValueError("boundary_control_weight must be finite and nonnegative")

    base = _load_two_step_module()
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    if x0.size != model.nq + model.nv:
        raise ValueError(f"x0 size {x0.size}; expected {model.nq + model.nv}")

    left_frame_id, right_frame_id = ensure_sole_contact_frames(model)
    left_start, right_start = base._frame_references(
        model, x0[: model.nq], left_frame_id, right_frame_id
    )
    left_references, right_references = base.swing_reference_poses(
        left_start, right_start
    )
    state_references = build_periodic_state_references(x0, nq=model.nq)

    data = model.createData()
    initial_com = np.asarray(
        pin.centerOfMass(model, data, x0[: model.nq]), dtype=float
    ).reshape(3)
    com_targets = np.repeat(initial_com[None, :], HORIZON_NODES + 1, axis=0)
    for node in range(HORIZON_NODES + 1):
        com_targets[node, 1] += (
            PLANNED_CYCLE_FORWARD_Y_M * _quintic_progress(node / HORIZON_NODES)
        )

    phases = base.build_phase_schedule()
    phase_by_node = [phase for phase in phases for _ in range(phase.node_count)]
    if len(phase_by_node) != HORIZON_NODES:
        raise RuntimeError("periodic phase expansion does not contain 200 nodes")

    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    safety_activation = base._joint_safety_activation(model, state, x0)
    running_models = []
    for node, phase in enumerate(phase_by_node):
        action_model = base._build_action_model(
            state=state,
            actuation=actuation,
            phase=phase,
            x_reference=state_references[node],
            com_target=com_targets[node],
            left_frame_id=left_frame_id,
            right_frame_id=right_frame_id,
            left_reference=left_references[node],
            right_reference=right_references[node],
            joint_safety_activation=safety_activation,
            dt_s=DT,
            enable_force_jacobians=True,
        )
        _add_contact_feasibility_cost(
            action_model=action_model,
            state=state,
            phase=phase,
            left_reference=left_references[node],
            right_reference=right_references[node],
        )
        if boundary_control_weight > 0.0 and node in (0, HORIZON_NODES - 1):
            control_residual = crocoddyl.ResidualModelControl(
                state, np.zeros(actuation.nu)
            )
            action_model.differential.costs.addCost(
                PERIODIC_BOUNDARY_CONTROL_COST_NAME,
                crocoddyl.CostModelResidual(state, control_residual),
                boundary_control_weight,
            )
        running_models.append(action_model)

    previous_terminal_weight = base.TERMINAL_STATE_WEIGHT
    try:
        base.TERMINAL_STATE_WEIGHT = terminal_state_weight
        terminal_model = base._build_action_model(
            state=state,
            actuation=actuation,
            phase=phases[-1],
            x_reference=state_references[-1],
            com_target=com_targets[-1],
            left_frame_id=left_frame_id,
            right_frame_id=right_frame_id,
            left_reference=left_references[-1],
            right_reference=right_references[-1],
            joint_safety_activation=safety_activation,
            dt_s=0.0,
        )
    finally:
        base.TERMINAL_STATE_WEIGHT = previous_terminal_weight

    metadata = {
        "seed_provenance": "accepted_fddp_two_step",
        "left_frame_id": left_frame_id,
        "right_frame_id": right_frame_id,
        "left_start": left_start,
        "right_start": right_start,
        "left_references": left_references,
        "right_references": right_references,
        "state_references": state_references,
        "com_targets": com_targets,
        "phase_by_node": phase_by_node,
        "terminal_state_weight": terminal_state_weight,
        "boundary_control_weight": boundary_control_weight,
        "boundary_control_reference": "quasi_static_zero",
        "contact_feasibility_mode": "planar_sagittal_resultant",
        "out_of_plane_policy": "diagnostic_only",
        "planar_support_cone_weight": PLANAR_SUPPORT_CONE_WEIGHT,
        "effective_planar_mu": EFFECTIVE_PLANAR_MU,
        "foot_half_length_m": FOOT_HALF_LENGTH_M,
        "minimum_cop_margin_m": MINIMUM_COP_MARGIN_M,
        "optimisation_cop_margin_m": OPTIMISATION_COP_MARGIN_M,
    }
    problem = crocoddyl.ShootingProblem(x0, running_models, terminal_model)
    return problem, state, metadata


def build_periodic_seed(
    model: pin.Model,
    x0: np.ndarray,
    trajectory_path: Path = TWO_STEP_TRAJECTORY_PATH,
    summary_path: Path = TWO_STEP_SUMMARY_PATH,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load the accepted dynamics-feasible FDDP two-step seed unchanged."""

    trajectory_path = Path(trajectory_path)
    summary_path = Path(summary_path)
    summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
    if not summary.get("exported_nominal_trajectory"):
        raise ValueError("two-step source was not exported as an accepted nominal")
    if not summary.get("acceptance", {}).get("passed"):
        raise ValueError("two-step source did not pass its acceptance gates")

    with np.load(trajectory_path, allow_pickle=False) as loaded:
        required = {"state_ref", "tau_ref", "tau_ref_hardware_usage"}
        missing = sorted(required - set(loaded.files))
        if missing:
            raise ValueError(f"two-step seed is missing arrays: {missing}")
        if str(loaded["tau_ref_hardware_usage"].item()) != "analysis_only":
            raise ValueError("two-step tau_ref boundary is not analysis_only")
        source_xs = np.asarray(loaded["state_ref"], dtype=float)
        source_us = np.asarray(loaded["tau_ref"], dtype=float)

    expected_x_shape = (HORIZON_NODES + 1, model.nq + model.nv)
    expected_u_shape = (HORIZON_NODES, model.nv - 6)
    if source_xs.shape != expected_x_shape or source_us.shape != expected_u_shape:
        raise ValueError(
            f"two-step seed shapes {source_xs.shape}/{source_us.shape}; "
            f"expected {expected_x_shape}/{expected_u_shape}"
        )
    if not np.allclose(source_xs[0], np.asarray(x0, dtype=float), atol=1.0e-12):
        raise ValueError("accepted two-step seed does not start from declared static x0")
    if not np.all(np.isfinite(source_xs)) or not np.all(np.isfinite(source_us)):
        raise ValueError("accepted two-step seed contains non-finite values")

    provenance = {
        "source": "accepted_fddp_two_step",
        "seed_transform": "none_dynamics_feasible",
        "trajectory": str(trajectory_path),
        "summary": str(summary_path),
        "trajectory_sha256": hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
        "tau_ref_hardware_usage": "analysis_only",
    }
    return source_xs.copy(), source_us.copy(), provenance


def _rotation_from_freeflyer(q: np.ndarray) -> np.ndarray:
    quaternion = pin.Quaternion(q[6], q[3], q[4], q[5])
    if abs(quaternion.norm() - 1.0) > 1.0e-6:
        raise ValueError("free-flyer quaternion must be normalised")
    return quaternion.toRotationMatrix()


def _translated_residual(
    terminal: np.ndarray,
    start: np.ndarray,
    planned_forward_y: float,
) -> np.ndarray:
    expected = np.array([0.0, planned_forward_y, 0.0], dtype=float)
    return np.asarray(terminal, dtype=float) - np.asarray(start, dtype=float) - expected


def compute_cycle_boundary_metrics(
    *,
    x_start: np.ndarray,
    x_terminal: np.ndarray,
    u_start: np.ndarray,
    u_terminal: np.ndarray,
    left_start: pin.SE3,
    left_terminal: pin.SE3,
    right_start: pin.SE3,
    right_terminal: pin.SE3,
    com_start: np.ndarray,
    com_terminal: np.ndarray,
    com_velocity_start: np.ndarray,
    com_velocity_terminal: np.ndarray,
    planned_forward_y: float,
    nq: int,
    start_contact_mode: str,
    terminal_contact_mode: str,
) -> dict:
    """Measure cycle-end continuity after removing planned `+Y` translation."""

    x_start = np.asarray(x_start, dtype=float).reshape(-1)
    x_terminal = np.asarray(x_terminal, dtype=float).reshape(-1)
    if x_start.shape != x_terminal.shape or x_start.size <= nq + 6:
        raise ValueError("cycle states must have equal valid multibody dimensions")
    if not np.all(np.isfinite(np.concatenate((x_start, x_terminal)))):
        raise ValueError("cycle states must be finite")

    start_rotation = _rotation_from_freeflyer(x_start[:nq])
    terminal_rotation = _rotation_from_freeflyer(x_terminal[:nq])
    base_rotation_error = pin.log3(start_rotation.T @ terminal_rotation)
    base_translation_residual = _translated_residual(
        x_terminal[:3], x_start[:3], planned_forward_y
    )
    com_translation_residual = _translated_residual(
        com_terminal, com_start, planned_forward_y
    )

    foot_translation_errors = []
    foot_rotation_errors = []
    for start_pose, terminal_pose in (
        (left_start, left_terminal),
        (right_start, right_terminal),
    ):
        foot_translation_errors.append(
            np.linalg.norm(
                _translated_residual(
                    terminal_pose.translation,
                    start_pose.translation,
                    planned_forward_y,
                )
            )
        )
        foot_rotation_errors.append(
            np.linalg.norm(pin.log3(start_pose.rotation.T @ terminal_pose.rotation))
        )

    joint_position_delta = x_terminal[7:nq] - x_start[7:nq]
    joint_velocity_delta = x_terminal[nq + 6 :] - x_start[nq + 6 :]
    u_delta = np.asarray(u_terminal, dtype=float) - np.asarray(u_start, dtype=float)
    if not np.all(np.isfinite(u_delta)):
        raise ValueError("cycle controls must be finite")

    return {
        "floating_base_translation_residual_xyz_m": base_translation_residual.tolist(),
        "floating_base_rotation_residual_xyz_rad": base_rotation_error.tolist(),
        "base_translation_residual_m": float(np.linalg.norm(base_translation_residual)),
        "base_rotation_residual_rad": float(np.linalg.norm(base_rotation_error)),
        "joint_position_difference_rad": float(
            np.max(np.abs(joint_position_delta), initial=0.0)
        ),
        "joint_velocity_difference_rad_s": float(
            np.max(np.abs(joint_velocity_delta), initial=0.0)
        ),
        "base_height_difference_m": float(abs(x_terminal[2] - x_start[2])),
        "base_pitch_difference_rad": float(abs(base_rotation_error[0])),
        "com_translation_residual_m": float(np.linalg.norm(com_translation_residual)),
        "com_velocity_difference_m_s": float(
            np.linalg.norm(
                np.asarray(com_velocity_terminal, dtype=float)
                - np.asarray(com_velocity_start, dtype=float)
            )
        ),
        "relative_foot_translation_error_m": float(max(foot_translation_errors)),
        "relative_foot_rotation_error_rad": float(max(foot_rotation_errors)),
        "control_discontinuity": float(np.max(np.abs(u_delta), initial=0.0)),
        "contact_mode_compatible": bool(
            start_contact_mode == "DS" and terminal_contact_mode == "DS"
        ),
        "start_contact_mode": str(start_contact_mode),
        "terminal_contact_mode": str(terminal_contact_mode),
        "planned_forward_y_m": float(planned_forward_y),
    }


def evaluate_cycle_acceptance(metrics: Mapping[str, object]) -> dict:
    missing = sorted(set(CYCLE_GATE_LIMITS) - set(metrics))
    if missing:
        raise KeyError(f"missing cycle gate metrics: {missing}")
    checks: dict[str, bool] = {}
    for name, (direction, threshold) in CYCLE_GATE_LIMITS.items():
        value = float(metrics[name])
        if not np.isfinite(value):
            raise ValueError(f"cycle metric {name!r} must be finite")
        checks[name] = value <= threshold if direction == "max" else value >= threshold
    checks["contact_mode_compatible"] = metrics.get("contact_mode_compatible") is True
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "diagnostics": dict(metrics),
    }


def _phase_mode(phase) -> str:
    if phase.left_contact_active and phase.right_contact_active:
        return "DS"
    if phase.right_contact_active:
        return "RIGHT_SS"
    if phase.left_contact_active:
        return "LEFT_SS"
    return "FLIGHT"


def _com_velocity(model: pin.Model, x: np.ndarray) -> np.ndarray:
    data = model.createData()
    pin.centerOfMass(model, data, x[: model.nq], x[model.nq :])
    return np.asarray(data.vcom[0], dtype=float).reshape(3).copy()


def _frame_pose(model: pin.Model, x: np.ndarray, frame_id: int) -> pin.SE3:
    data = model.createData()
    pin.forwardKinematics(model, data, x[: model.nq])
    pin.updateFramePlacements(model, data)
    return data.oMf[frame_id].copy()


def compute_periodic_contact_feasibility(
    *, problem, model: pin.Model, metadata: Mapping[str, object], xs, us
) -> dict:
    """Evaluate the same node-level planar contact gates used by 10-step promotion."""

    feasibility = _load_contact_feasibility_module()
    config = yaml.safe_load(CONTACT_CONFIG_PATH.read_text(encoding="utf-8"))
    contact_config = dict(config["contact"])
    xs = np.asarray(xs, dtype=float)
    us = np.asarray(us, dtype=float)
    problem.calc([x.copy() for x in xs], [u.copy() for u in us])
    pin_data = model.createData()
    contact_to_side = {LEFT_CONTACT_NAME: "left", RIGHT_CONTACT_NAME: "right"}
    rows = []
    for node, action_data in enumerate(problem.runningDatas):
        x = xs[node]
        q = x[: model.nq]
        v = x[model.nq :]
        pin.forwardKinematics(model, pin_data, q, v)
        pin.updateFramePlacements(model, pin_data)
        contacts = action_data.differential.multibody.contacts.contacts.todict()
        phase = metadata["phase_by_node"][node]
        expected = {
            side
            for side, active in (
                ("left", phase.left_contact_active),
                ("right", phase.right_contact_active),
            )
            if active
        }
        actual = {contact_to_side[name] for name in contacts}
        if actual != expected:
            raise RuntimeError(
                f"node {node} contact set {actual} disagrees with schedule {expected}"
            )
        for name, contact_data in contacts.items():
            side = contact_to_side[name]
            frame_id = int(contact_data.frame)
            rotation = np.asarray(pin_data.oMf[frame_id].rotation, dtype=float)
            contact_offset = np.asarray(
                model.frames[frame_id].placement.translation, dtype=float
            )
            contact_to_sole = feasibility.contact_frame_to_sole_offset(
                sole_offset_from_ankle_local=SOLE_OFFSET_FROM_ANKLE_LOCAL_M,
                contact_frame_offset_from_ankle_local=contact_offset,
            )
            rows.append(
                {
                    "node": node,
                    "time_s": node * DT,
                    "phase": phase.name,
                    "cycle_id": 0,
                    "side": side,
                    "sole_position_world_m": np.asarray(
                        pin_data.oMf[frame_id].translation, dtype=float
                    ).copy(),
                    "wrench": feasibility.contact_wrench_at_sole(
                        np.asarray(contact_data.f.vector, dtype=float),
                        frame_rotation_world=rotation,
                        sole_offset_local=contact_to_sole,
                    ),
                    "tangential_speed_m_s": feasibility.sole_tangential_speed(
                        np.asarray(contact_data.v.vector, dtype=float),
                        frame_rotation_world=rotation,
                        sole_offset_local=contact_to_sole,
                    ),
                }
            )
    return feasibility.evaluate_planar_support_feasibility(
        rows,
        contact_config,
        foot_half_length_m=FOOT_HALF_LENGTH_M,
    )


def compute_periodic_diagnostics(
    *,
    model: pin.Model,
    problem,
    state,
    xs: np.ndarray,
    us: np.ndarray,
    metadata: Mapping[str, object],
    solver,
) -> tuple[dict, np.ndarray]:
    """Combine dynamics/pose feasibility with translated cycle continuity."""

    base = _load_two_step_module()
    xs = np.asarray(xs, dtype=float)
    us = np.asarray(us, dtype=float)
    base_diagnostics, com = base.compute_two_step_diagnostics(
        model=model,
        problem=problem,
        state=state,
        xs=xs,
        us=us,
        metadata=dict(metadata),
        solver=solver,
    )
    first_phase = metadata["phase_by_node"][0]
    terminal_phase = metadata["phase_by_node"][-1]
    boundary = compute_cycle_boundary_metrics(
        x_start=xs[0],
        x_terminal=xs[-1],
        u_start=us[0],
        u_terminal=us[-1],
        left_start=_frame_pose(model, xs[0], int(metadata["left_frame_id"])),
        left_terminal=_frame_pose(
            model, xs[-1], int(metadata["left_frame_id"])
        ),
        right_start=_frame_pose(model, xs[0], int(metadata["right_frame_id"])),
        right_terminal=_frame_pose(
            model, xs[-1], int(metadata["right_frame_id"])
        ),
        com_start=com[0],
        com_terminal=com[-1],
        com_velocity_start=_com_velocity(model, xs[0]),
        com_velocity_terminal=_com_velocity(model, xs[-1]),
        planned_forward_y=PLANNED_CYCLE_FORWARD_Y_M,
        nq=model.nq,
        start_contact_mode=_phase_mode(first_phase),
        terminal_contact_mode=_phase_mode(terminal_phase),
    )
    contact = compute_periodic_contact_feasibility(
        problem=problem,
        model=model,
        metadata=metadata,
        xs=xs,
        us=us,
    )
    diagnostics = {
        **boundary,
        "solver_stop": float(solver.stop),
        "solver_iterations": int(solver.iter),
        "solver_cost": float(solver.cost),
        "shooting_gap_norm": float(base_diagnostics["max_shooting_gap_norm"]),
        "active_support_translation_error_m": float(
            base_diagnostics["max_active_support_translation_error_m"]
        ),
        "active_support_rotation_error_rad": float(
            base_diagnostics["max_active_support_rotation_error_rad"]
        ),
        "landing_translation_error_m": float(
            max(
                base_diagnostics["first_landing_translation_error_m"],
                base_diagnostics["second_landing_translation_error_m"],
            )
        ),
        "landing_rotation_error_rad": float(
            max(
                base_diagnostics["first_landing_rotation_error_rad"],
                base_diagnostics["second_landing_rotation_error_rad"],
            )
        ),
        "swing_clearance_m": float(
            min(
                base_diagnostics["min_left_swing_clearance_m"],
                base_diagnostics["min_right_swing_clearance_m"],
            )
        ),
        "joint_limit_margin_rad": float(
            base_diagnostics["min_joint_limit_margin_rad"]
        ),
        "achieved_forward_com_displacement_m": float(
            base_diagnostics["achieved_forward_com_displacement_m"]
        ),
        "contact_support_node_count": int(contact["support_node_count"]),
        "contact_violation_count": int(contact["violation_count"]),
        "minimum_contact_normal_force_n": float(
            contact["minimum_normal_force_n"]
        ),
        "maximum_friction_utilisation": float(
            contact["maximum_friction_utilisation"]
        ),
        "minimum_cop_margin_m": float(contact["minimum_cop_margin_m"]),
        "maximum_tangential_speed_m_s": float(
            contact["maximum_tangential_speed_m_s"]
        ),
        "contact_feasibility_accepted": bool(contact["accepted"]),
        "contact_feasibility": contact,
    }
    return diagnostics, com


def _yaml_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_cycle_artifacts(
    result: Mapping[str, object],
    *,
    accepted: bool,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    trajectory_dir: Path | None = None,
) -> dict[str, Path]:
    """Always preserve diagnostics; create NPZ only for an accepted result."""

    acceptance = result.get("acceptance")
    if not isinstance(acceptance, Mapping):
        raise ValueError("result is missing acceptance diagnostics")
    if bool(acceptance.get("passed")) != bool(accepted):
        raise ValueError("accepted flag contradicts the acceptance decision")

    output_dir = Path(output_dir)
    trajectory_dir = Path(trajectory_dir or output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "diagnostic_yaml": output_dir / "fddp_periodic_cycle_diagnostics.yaml",
        "iteration_csv": output_dir / "fddp_periodic_cycle_iterations.csv",
        "trajectory_npz": trajectory_dir / "fddp_periodic_cycle_ref.npz",
    }

    summary = {
        "schema_version": "fddp-periodic-cycle-v1",
        "stage": "fddp_periodic_cycle",
        "exported_nominal_trajectory": bool(accepted),
        "claim_boundary": "simulation FDDP planning evidence only",
        "tau_ref_hardware_usage": "analysis_only",
        "diagnostics": result.get("diagnostics", {}),
        "acceptance": acceptance,
    }
    paths["diagnostic_yaml"].write_text(
        yaml.safe_dump(_yaml_safe(summary), sort_keys=False), encoding="utf-8"
    )

    iteration_history = list(result.get("iteration_history", []))
    fieldnames = sorted(
        {str(key) for row in iteration_history for key in row}
        or {"iteration", "cost", "stop"}
    )
    with paths["iteration_csv"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in iteration_history:
            writer.writerow({key: _yaml_safe(row.get(key)) for key in fieldnames})

    if accepted:
        trajectory = result.get("trajectory")
        if not isinstance(trajectory, Mapping):
            raise ValueError("accepted result is missing trajectory arrays")
        np.savez(paths["trajectory_npz"], **trajectory)
    return paths


def _logger_history(
    logger,
    *,
    stage_index: int,
    stage_name: str,
) -> list[dict]:
    history = []
    count = len(logger.costs)
    for index in range(count):
        history.append(
            {
                "stage_index": int(stage_index),
                "stage_name": str(stage_name),
                "iteration": int(logger.iters[index]),
                "cost": float(logger.costs[index]),
                "stop": float(logger.stops[index]),
                "step_length": float(logger.steps[index]),
                "primal_regularization": float(logger.pregs[index]),
                "dual_regularization": float(logger.dregs[index]),
            }
        )
    return history


def _trajectory_arrays(
    model: pin.Model,
    xs: np.ndarray,
    us: np.ndarray,
    com: np.ndarray,
    metadata: Mapping[str, object],
    provenance: Mapping[str, object],
) -> dict[str, np.ndarray]:
    phases = list(metadata["phase_by_node"])
    phase_names = np.asarray(
        [phase.name for phase in phases] + [phases[-1].name], dtype="U40"
    )
    left_contact = np.asarray(
        [phase.left_contact_active for phase in phases] + [True], dtype=np.uint8
    )
    right_contact = np.asarray(
        [phase.right_contact_active for phase in phases] + [True], dtype=np.uint8
    )
    left_poses = np.asarray(
        [pose.homogeneous for pose in metadata["left_references"]], dtype=float
    )
    right_poses = np.asarray(
        [pose.homogeneous for pose in metadata["right_references"]], dtype=float
    )
    return {
        "schema_version": np.asarray("fddp-periodic-cycle-v1"),
        "t_ref": np.arange(HORIZON_NODES + 1, dtype=float) * DT,
        "q_ref": xs[:, 7 : model.nq],
        "dq_ref": xs[:, model.nq + 6 :],
        "state_ref": xs,
        "tau_ref": us,
        "com_ref_world": com,
        "left_foot_pose_ref": left_poses,
        "right_foot_pose_ref": right_poses,
        "phase": phase_names,
        "left_contact_active": left_contact,
        "right_contact_active": right_contact,
        "cycle_id": np.zeros(HORIZON_NODES + 1, dtype=np.int64),
        "seed_provenance": np.asarray(str(provenance["source"])),
        "tau_ref_hardware_usage": np.asarray("analysis_only"),
        "claim_boundary": np.asarray("simulation_only"),
    }


def solve_periodic_cycle(
    max_iterations: int = 500,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    trajectory_dir: Path = DEFAULT_TRAJECTORY_DIR,
) -> dict:
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
    x0 = load_x0(X0_PATH, model.nq + model.nv)
    xs, us, provenance = build_periodic_seed(model, x0)

    all_history: list[dict] = []
    stage_summaries: list[dict] = []
    diagnostics = None
    com = None
    metadata = None
    completed_stages = 0
    stages = build_periodic_homotopy_stages()

    for stage_index, stage in enumerate(stages):
        problem, state, metadata = build_periodic_cycle_problem(
            model,
            x0,
            terminal_state_weight=stage.terminal_state_weight,
            boundary_control_weight=stage.boundary_control_weight,
        )
        solver = crocoddyl.SolverFDDP(problem)
        solver.th_stop = 1.0e-7
        logger = crocoddyl.CallbackLogger()
        solver.setCallbacks([logger])
        returned_success = bool(
            solver.solve(
                [x.copy() for x in xs],
                [u.copy() for u in us],
                int(max_iterations),
                False,
            )
        )
        xs = np.asarray(solver.xs, dtype=float)
        us = np.asarray(solver.us, dtype=float)
        diagnostics, com = compute_periodic_diagnostics(
            model=model,
            problem=problem,
            state=state,
            xs=xs,
            us=us,
            metadata=metadata,
            solver=solver,
        )
        stage_converged = bool(
            returned_success
            and diagnostics["solver_stop"] <= 1.0e-7
            and diagnostics["shooting_gap_norm"] <= 1.0e-7
        )
        stage_summary = {
            "stage_index": stage_index,
            "stage_name": stage.name,
            "terminal_state_weight": stage.terminal_state_weight,
            "boundary_control_weight": stage.boundary_control_weight,
            "solver_return_success": returned_success,
            "solver_iterations": diagnostics["solver_iterations"],
            "solver_stop": diagnostics["solver_stop"],
            "shooting_gap_norm": diagnostics["shooting_gap_norm"],
            "cost": diagnostics["solver_cost"],
            "stage_converged": stage_converged,
        }
        stage_summaries.append(stage_summary)
        all_history.extend(
            _logger_history(
                logger,
                stage_index=stage_index,
                stage_name=stage.name,
            )
        )
        if not stage_converged:
            break
        completed_stages += 1

    if diagnostics is None or com is None or metadata is None:
        raise RuntimeError("periodic homotopy did not execute any stage")
    diagnostics["solver_return_success"] = bool(
        stage_summaries[-1]["solver_return_success"]
    )
    diagnostics["seed_provenance"] = provenance
    diagnostics["homotopy_stages"] = stage_summaries
    diagnostics["completed_homotopy_stages"] = completed_stages
    diagnostics["required_homotopy_stages"] = len(stages)

    acceptance = evaluate_cycle_acceptance(diagnostics)
    if completed_stages != len(stages):
        acceptance["passed"] = False
        acceptance["checks"]["homotopy_complete"] = False
        acceptance["failures"].append("homotopy_incomplete")
    else:
        acceptance["checks"]["homotopy_complete"] = True

    result = {
        "diagnostics": diagnostics,
        "acceptance": acceptance,
        "iteration_history": all_history,
    }
    if acceptance["passed"]:
        result["trajectory"] = _trajectory_arrays(
            model, xs, us, com, metadata, provenance
        )
    outputs = write_cycle_artifacts(
        result,
        accepted=acceptance["passed"],
        output_dir=output_dir,
        trajectory_dir=trajectory_dir,
    )
    return {
        **result,
        "exported": bool(acceptance["passed"]),
        "outputs": outputs,
    }


def main() -> int:
    result = solve_periodic_cycle()
    print("Periodic FDDP cycle acceptance:")
    for name, passed in result["acceptance"]["checks"].items():
        print(f"  {name}: {passed} ({result['diagnostics'].get(name)})")
    if not result["acceptance"]["passed"]:
        print("Nominal export withheld:")
        for failure in result["acceptance"]["failures"]:
            print(f"  - {failure}")
        return 1
    print(f"Saved periodic trajectory: {result['outputs']['trajectory_npz']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

