"""Gated 4/6/10-step continuation from the accepted periodic FDDP cycle."""

from __future__ import annotations

import csv
import importlib.util
import os
import sys
from pathlib import Path
from typing import Mapping

import crocoddyl
import numpy as np
import pinocchio as pin
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERIODIC_MODULE_PATH = PROJECT_ROOT / "fddp" / "05a_solve_fddp_periodic_cycle.py"
PERIODIC_TRAJECTORY_PATH = (
    PROJECT_ROOT / "transfer" / "trajectories" / "fddp_periodic_cycle_ref.npz"
)
PERIODIC_SUMMARY_PATH = (
    PROJECT_ROOT / "results" / "logs" / "fddp_periodic_cycle_diagnostics.yaml"
)
URDF_PATH = PROJECT_ROOT / "model" / "biped_v1.urdf"
X0_PATH = PROJECT_ROOT / "transfer" / "initialisation" / "fddp_double_support_x0.npz"

RIGHT_CONTACT_NAME = "right_foot_6d"
LEFT_CONTACT_NAME = "left_foot_6d"
PERIODIC_BOUNDARY_CONTROL_COST_NAME = "periodic_boundary_control"
DT = 0.010
CYCLE_NODES = 200
STEPS_PER_CYCLE = 2
STEP_FORWARD_Y_M = 0.010
TERMINAL_STATE_WEIGHT = 1000.0
BOUNDARY_CONTROL_WEIGHT = 30.0

LEFT_SOLE_FRAME = "left_sole_contact"
RIGHT_SOLE_FRAME = "right_sole_contact"
SOLE_OFFSET_FROM_ANKLE_LOCAL_M = np.array([0.0, 0.0, -0.043], dtype=float)
FOOT_HALF_LENGTH_M = 0.0725
FRICTION_COEFFICIENT = 0.80
MAXIMUM_FRICTION_UTILISATION = 0.90
MINIMUM_COP_MARGIN_M = 0.002
OPTIMISATION_COP_MARGIN_M = 0.0027
EFFECTIVE_PLANAR_MU = FRICTION_COEFFICIENT * MAXIMUM_FRICTION_UTILISATION
PLANAR_SUPPORT_CONE_WEIGHTS = {4: 1.0e3, 6: 1.0e4, 10: 1.0e5}
PLANAR_SUPPORT_CONE_COST_NAME = "planar_support_resultant_cone"


class SeedIntegrityError(ValueError):
    pass


class PlanarSupportConeResidualData(crocoddyl.ResidualDataAbstract):
    """Shared-data handle for the combined planar support residual."""

    def __init__(self, model, shared):
        crocoddyl.ResidualDataAbstract.__init__(self, model, shared)


class ResidualModelPlanarSupportCone(crocoddyl.ResidualModelAbstract):
    """Non-negative sagittal resultant cone for the currently active soles."""

    def __init__(self, state, active_contacts, *, nu: int):
        if not active_contacts:
            raise ValueError("planar support cone requires an active contact")
        self.active_contact_names = tuple(name for name, _ in active_contacts)
        self.support_positions_world_m = {
            name: np.asarray(position, dtype=float).reshape(3).copy()
            for name, position in active_contacts
        }
        if len(set(self.active_contact_names)) != len(self.active_contact_names):
            raise ValueError("planar support cone contact names must be unique")
        if not all(
            np.all(np.isfinite(position))
            for position in self.support_positions_world_m.values()
        ):
            raise ValueError("planar support positions must be finite")
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
        missing = [
            name for name in self.active_contact_names if name not in contacts
        ]
        if missing:
            raise RuntimeError(
                f"planar support residual missing active contacts: {missing}"
            )
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
            data.Rx[:, :] += matrix @ np.asarray(
                contacts[name].df_dx, dtype=float
            )
            data.Ru[:, :] += matrix @ np.asarray(
                contacts[name].df_du, dtype=float
            )

def _load_periodic_module():
    name = "_fddp_multistep_periodic_base"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PERIODIC_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load periodic module: {PERIODIC_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_x0(path: Path, expected_size: int) -> np.ndarray:
    return _load_periodic_module().load_x0(Path(path), expected_size)


def continuation_ladder(final_step_count: int = 10) -> tuple[int, ...]:
    if final_step_count != 10:
        raise ValueError("the approved continuation ladder terminates at ten steps")
    return (2, 4, 6, 10)


def required_seed_for(target_steps: int) -> str:
    required = {
        4: "accepted_fddp_periodic_2step",
        6: "accepted_fddp_4step",
        10: "accepted_fddp_6step",
    }
    try:
        return required[int(target_steps)]
    except (KeyError, ValueError) as error:
        raise ValueError(f"unsupported target step count: {target_steps}") from error


def validate_seed_metadata(metadata: Mapping[str, object], target_steps: int) -> None:
    expected_source = required_seed_for(target_steps)
    expected_steps = {4: 2, 6: 4, 10: 6}[int(target_steps)]
    if metadata.get("source") != expected_source:
        raise SeedIntegrityError(
            f"target {target_steps} requires {expected_source}, got {metadata.get('source')}"
        )
    if int(metadata.get("source_step_count", -1)) != expected_steps:
        raise SeedIntegrityError("seed source step count does not match the ladder")
    if metadata.get("accepted") is not True:
        raise SeedIntegrityError("seed source was not accepted")
    if metadata.get("tau_ref_hardware_usage") != "analysis_only":
        raise SeedIntegrityError("seed tau_ref boundary is not analysis_only")
    if "scripted" in str(metadata).lower():
        raise SeedIntegrityError("scripted gait data is forbidden as an FDDP seed")


def extend_control_seed(
    *,
    previous_us: np.ndarray,
    previous_steps: int,
    target_steps: int,
    periodic_us: np.ndarray,
) -> np.ndarray:
    previous_us = np.asarray(previous_us, dtype=float)
    periodic_us = np.asarray(periodic_us, dtype=float)
    if periodic_us.ndim != 2 or periodic_us.shape[0] != CYCLE_NODES:
        raise ValueError("periodic controls must contain one 200-node cycle")
    if previous_us.ndim != 2 or previous_us.shape[1] != periodic_us.shape[1]:
        raise ValueError("previous and periodic controls have incompatible shapes")
    if previous_steps <= 0 or previous_steps % 2:
        raise ValueError("previous_steps must be a positive even count")
    if target_steps <= previous_steps or target_steps % 2:
        raise ValueError("target_steps must be a larger even count")
    expected_previous_nodes = previous_steps // STEPS_PER_CYCLE * CYCLE_NODES
    if previous_us.shape[0] != expected_previous_nodes:
        raise ValueError("previous control horizon disagrees with previous_steps")
    extra_cycles = (target_steps - previous_steps) // STEPS_PER_CYCLE
    return np.concatenate(
        [previous_us] + [periodic_us.copy() for _ in range(extra_cycles)], axis=0
    )


def evaluate_segment(metrics: Mapping[str, float], *, step_count: int) -> dict:
    limits = {
        "solver_stop": ("max", 1.0e-7),
        "shooting_gap_norm": ("max", 1.0e-7),
        "active_support_translation_error_m": ("max", 5.0e-4),
        "active_support_rotation_error_rad": ("max", np.deg2rad(0.5)),
        "landing_translation_error_m": ("max", 1.0e-3),
        "landing_rotation_error_rad": ("max", np.deg2rad(1.0)),
        "swing_clearance_m": ("min", 1.0e-2),
        "joint_limit_margin_rad": ("min", 1.0e-2),
        "forward_displacement_m": ("min", 4.0e-3 * step_count),
        "duration_s": ("min", float(step_count)),
        "contact_support_node_count": (
            "min",
            float(step_count // STEPS_PER_CYCLE * CYCLE_NODES),
        ),
        "contact_violation_count": ("max", 0.0),
        "minimum_contact_normal_force_n": ("min", -1.0e-5),
        "maximum_friction_utilisation": (
            "max",
            MAXIMUM_FRICTION_UTILISATION,
        ),
        "minimum_cop_margin_m": ("min", MINIMUM_COP_MARGIN_M),
        "maximum_tangential_speed_m_s": ("max", 1.0e-3),
    }
    missing = sorted(set(limits) - set(metrics))
    if missing:
        raise KeyError(f"missing segment metrics: {missing}")
    checks = {}
    for name, (direction, threshold) in limits.items():
        value = float(metrics[name])
        if not np.isfinite(value):
            raise ValueError(f"segment metric {name!r} must be finite")
        checks[name] = value <= threshold if direction == "max" else value >= threshold
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "diagnostics": dict(metrics),
    }


def configure_solver(solver):
    solver.th_stop = 1.0e-7
    solver.reg_max = 1.0e14
    return solver


def ensure_sole_contact_frames(model: pin.Model) -> tuple[int, int]:
    """Add idempotent operational frames at the URDF/MJCF sole plane."""

    periodic = _load_periodic_module()
    base = periodic._load_two_step_module()
    output = []
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
            output.append(int(existing))
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
        output.append(int(model.addFrame(sole, False)))
    return tuple(output)


def _add_contact_feasibility_costs(
    *,
    action_model,
    state,
    phase,
    left_reference: pin.SE3,
    right_reference: pin.SE3,
    weight: float,
) -> None:
    """Add one identifiable combined sagittal support-cone cost per node."""

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
        float(weight),
    )

def _translated_pose(pose: pin.SE3, forward_y: float) -> pin.SE3:
    translation = pose.translation.copy()
    translation[1] += forward_y
    return pin.SE3(pose.rotation.copy(), translation)


def _build_references(model: pin.Model, x0: np.ndarray, step_count: int) -> dict:
    periodic = _load_periodic_module()
    base = periodic._load_two_step_module()
    cycle_count = step_count // STEPS_PER_CYCLE
    left_frame_id, right_frame_id = ensure_sole_contact_frames(model)
    left_start, right_start = base._frame_references(
        model, x0[: model.nq], left_frame_id, right_frame_id
    )
    data = model.createData()
    initial_com = np.asarray(
        pin.centerOfMass(model, data, x0[: model.nq]), dtype=float
    ).reshape(3)

    state_blocks = []
    com_blocks = []
    left_blocks = []
    right_blocks = []
    phase_by_node = []
    cycle_id_by_node = []
    phases = base.build_phase_schedule()
    local_phases = [phase for phase in phases for _ in range(phase.node_count)]
    for cycle_id in range(cycle_count):
        shift = cycle_id * STEP_FORWARD_Y_M
        states = periodic.build_periodic_state_references(x0, nq=model.nq)
        states[:, 1] += shift
        com = np.repeat(initial_com[None, :], CYCLE_NODES + 1, axis=0)
        for local_node in range(CYCLE_NODES + 1):
            com[local_node, 1] += shift + STEP_FORWARD_Y_M * periodic._quintic_progress(
                local_node / CYCLE_NODES
            )
        left_cycle_start = _translated_pose(left_start, shift)
        right_cycle_start = _translated_pose(right_start, shift)
        left_poses, right_poses = base.swing_reference_poses(
            left_cycle_start, right_cycle_start
        )
        if cycle_id == 0:
            state_blocks.append(states)
            com_blocks.append(com)
            left_blocks.extend(left_poses)
            right_blocks.extend(right_poses)
        else:
            state_blocks.append(states[1:])
            com_blocks.append(com[1:])
            left_blocks.extend(left_poses[1:])
            right_blocks.extend(right_poses[1:])
        phase_by_node.extend(local_phases)
        cycle_id_by_node.extend([cycle_id] * CYCLE_NODES)
    return {
        "left_frame_id": left_frame_id,
        "right_frame_id": right_frame_id,
        "left_start": left_start,
        "right_start": right_start,
        "state_references": np.concatenate(state_blocks, axis=0),
        "com_targets": np.concatenate(com_blocks, axis=0),
        "left_references": left_blocks,
        "right_references": right_blocks,
        "phase_by_node": phase_by_node,
        "cycle_id_by_node": np.asarray(cycle_id_by_node, dtype=int),
        "phases": phases,
    }


def build_multistep_problem(model: pin.Model, x0: np.ndarray, *, step_count: int):
    if step_count < 4 or step_count % 2:
        raise ValueError("multi-step problem requires an even step count >= 4")
    periodic = _load_periodic_module()
    base = periodic._load_two_step_module()
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    if x0.size != model.nq + model.nv:
        raise ValueError("x0 dimension does not match the model")
    metadata = _build_references(model, x0, step_count)
    contact_weight = PLANAR_SUPPORT_CONE_WEIGHTS[step_count]
    horizon_nodes = step_count // STEPS_PER_CYCLE * CYCLE_NODES

    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    safety_activation = base._joint_safety_activation(model, state, x0)
    running_models = []
    for node in range(horizon_nodes):
        action_model = base._build_action_model(
            state=state,
            actuation=actuation,
            phase=metadata["phase_by_node"][node],
            x_reference=metadata["state_references"][node],
            com_target=metadata["com_targets"][node],
            left_frame_id=metadata["left_frame_id"],
            right_frame_id=metadata["right_frame_id"],
            left_reference=metadata["left_references"][node],
            right_reference=metadata["right_references"][node],
            joint_safety_activation=safety_activation,
            dt_s=DT,
            enable_force_jacobians=True,
        )
        _add_contact_feasibility_costs(
            action_model=action_model,
            state=state,
            phase=metadata["phase_by_node"][node],
            left_reference=metadata["left_references"][node],
            right_reference=metadata["right_references"][node],
            weight=contact_weight,
        )
        local_node = node % CYCLE_NODES
        if local_node in (0, CYCLE_NODES - 1):
            residual = crocoddyl.ResidualModelControl(
                state, np.zeros(actuation.nu)
            )
            action_model.differential.costs.addCost(
                PERIODIC_BOUNDARY_CONTROL_COST_NAME,
                crocoddyl.CostModelResidual(state, residual),
                BOUNDARY_CONTROL_WEIGHT,
            )
        running_models.append(action_model)

    previous_terminal_weight = base.TERMINAL_STATE_WEIGHT
    try:
        base.TERMINAL_STATE_WEIGHT = TERMINAL_STATE_WEIGHT
        terminal_model = base._build_action_model(
            state=state,
            actuation=actuation,
            phase=metadata["phases"][-1],
            x_reference=metadata["state_references"][-1],
            com_target=metadata["com_targets"][-1],
            left_frame_id=metadata["left_frame_id"],
            right_frame_id=metadata["right_frame_id"],
            left_reference=metadata["left_references"][-1],
            right_reference=metadata["right_references"][-1],
            joint_safety_activation=safety_activation,
            dt_s=0.0,
        )
    finally:
        base.TERMINAL_STATE_WEIGHT = previous_terminal_weight
    metadata["step_count"] = step_count
    metadata["horizon_nodes"] = horizon_nodes
    metadata["terminal_state_weight"] = TERMINAL_STATE_WEIGHT
    metadata["boundary_control_weight"] = BOUNDARY_CONTROL_WEIGHT
    metadata["contact_feasibility_mode"] = "planar_sagittal_resultant"
    metadata["out_of_plane_policy"] = "diagnostic_only"
    metadata["planar_support_cone_weight"] = contact_weight
    metadata["effective_planar_mu"] = EFFECTIVE_PLANAR_MU
    metadata["foot_half_length_m"] = FOOT_HALF_LENGTH_M
    metadata["minimum_cop_margin_m"] = MINIMUM_COP_MARGIN_M
    metadata["optimisation_cop_margin_m"] = OPTIMISATION_COP_MARGIN_M
    problem = crocoddyl.ShootingProblem(x0, running_models, terminal_model)
    return problem, state, metadata


def _load_source_trajectory(
    source_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    source_path = Path(source_path)
    with np.load(source_path, allow_pickle=False) as loaded:
        required = {"state_ref", "tau_ref", "tau_ref_hardware_usage"}
        missing = sorted(required - set(loaded.files))
        if missing:
            raise SeedIntegrityError(f"seed file is missing arrays: {missing}")
        usage = str(loaded["tau_ref_hardware_usage"].item())
        states = np.asarray(loaded["state_ref"], dtype=float)
        controls = np.asarray(loaded["tau_ref"], dtype=float)
    if usage != "analysis_only":
        raise SeedIntegrityError("seed tau_ref is not marked analysis_only")
    if states.shape[0] != controls.shape[0] + 1:
        raise SeedIntegrityError("seed state/control horizons are inconsistent")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(controls)):
        raise SeedIntegrityError("seed trajectory contains non-finite values")
    source_steps = controls.shape[0] // (CYCLE_NODES // STEPS_PER_CYCLE)
    return states, controls, {
        "source_step_count": source_steps,
        "tau_ref_hardware_usage": usage,
    }


def build_continuation_seed(
    *,
    problem,
    state,
    target_steps: int,
    source_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    source_xs, source_us, source_info = _load_source_trajectory(source_path)
    with np.load(PERIODIC_TRAJECTORY_PATH, allow_pickle=False) as periodic_loaded:
        periodic_xs = np.asarray(periodic_loaded["state_ref"], dtype=float)
        periodic_us = np.asarray(periodic_loaded["tau_ref"], dtype=float)
    source_steps = int(source_info["source_step_count"])
    source_label = (
        "accepted_fddp_periodic_2step"
        if source_steps == 2
        else f"accepted_fddp_{source_steps}step"
    )
    accepted = True
    if source_steps == 2:
        summary = yaml.safe_load(PERIODIC_SUMMARY_PATH.read_text(encoding="utf-8"))
        accepted = bool(summary.get("acceptance", {}).get("passed"))
    metadata = {
        "source": source_label,
        "source_step_count": source_steps,
        "accepted": accepted,
        "tau_ref_hardware_usage": source_info["tau_ref_hardware_usage"],
        "source_path": str(source_path),
        "seed_method": "accepted_fddp_periodic_start_seam_stitch",
        "seam_state_policy": "translated_periodic_start",
    }
    validate_seed_metadata(metadata, target_steps)
    us_seed = extend_control_seed(
        previous_us=source_us,
        previous_steps=source_steps,
        target_steps=target_steps,
        periodic_us=periodic_us,
    )
    first_extra_cycle = source_steps // STEPS_PER_CYCLE
    source_block = source_xs.copy()
    source_block[-1] = periodic_xs[0].copy()
    source_block[-1, 1] += first_extra_cycle * STEP_FORWARD_Y_M
    state_blocks = [source_block]
    extra_cycles = (target_steps - source_steps) // STEPS_PER_CYCLE
    for extra_cycle in range(extra_cycles):
        block = periodic_xs[1:].copy()
        cycle_index = first_extra_cycle + extra_cycle
        block[:, 1] += cycle_index * STEP_FORWARD_Y_M
        if extra_cycle + 1 < extra_cycles:
            block[-1] = periodic_xs[0].copy()
            block[-1, 1] += (cycle_index + 1) * STEP_FORWARD_Y_M
        state_blocks.append(block)
    xs_seed = np.concatenate(state_blocks, axis=0)
    if xs_seed.shape[0] != us_seed.shape[0] + 1:
        raise RuntimeError("stitched seed returned an invalid state horizon")
    problem.calc([x.copy() for x in xs_seed], [u.copy() for u in us_seed])
    gaps = [
        float(np.linalg.norm(state.diff(xs_seed[k + 1], data.xnext)))
        for k, data in enumerate(problem.runningDatas)
    ]
    metadata["initial_max_shooting_gap_norm"] = max(gaps, default=0.0)
    if (
        not np.all(np.isfinite(xs_seed))
        or metadata["initial_max_shooting_gap_norm"] > 1.0e-3
    ):
        raise SeedIntegrityError(
            "stitched seed is non-finite or has an excessive boundary gap"
        )
    return xs_seed, us_seed, metadata


def _placement_error(actual: pin.SE3, reference: pin.SE3) -> tuple[float, float]:
    return (
        float(np.linalg.norm(actual.translation - reference.translation)),
        float(np.linalg.norm(pin.log3(reference.rotation.T @ actual.rotation))),
    )


def compute_multistep_diagnostics(
    *, model, problem, state, xs, us, metadata, solver
) -> tuple[dict, np.ndarray]:
    xs = np.asarray(xs, dtype=float)
    us = np.asarray(us, dtype=float)
    problem.calc([x.copy() for x in xs], [u.copy() for u in us])
    gaps = [
        float(np.linalg.norm(state.diff(xs[k + 1], data.xnext)))
        for k, data in enumerate(problem.runningDatas)
    ]
    data = model.createData()
    com = np.zeros((xs.shape[0], 3), dtype=float)
    max_support_translation = 0.0
    max_support_rotation = 0.0
    actual_left = []
    actual_right = []
    for node, x in enumerate(xs):
        q = x[: model.nq]
        com[node] = np.asarray(pin.centerOfMass(model, data, q), dtype=float)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        left = data.oMf[metadata["left_frame_id"]].copy()
        right = data.oMf[metadata["right_frame_id"]].copy()
        actual_left.append(left)
        actual_right.append(right)
        phase = metadata["phase_by_node"][min(node, problem.T - 1)]
        if phase.left_contact_active:
            translation, rotation = _placement_error(
                left, metadata["left_references"][node]
            )
            max_support_translation = max(max_support_translation, translation)
            max_support_rotation = max(max_support_rotation, rotation)
        if phase.right_contact_active:
            translation, rotation = _placement_error(
                right, metadata["right_references"][node]
            )
            max_support_translation = max(max_support_translation, translation)
            max_support_rotation = max(max_support_rotation, rotation)

    landing_translation_errors = []
    landing_rotation_errors = []
    swing_clearances = []
    cycle_count = metadata["step_count"] // STEPS_PER_CYCLE
    for cycle_id in range(cycle_count):
        start = cycle_id * CYCLE_NODES
        for side, landing_node, swing_start, swing_end, poses in (
            ("left", start + 90, start + 20, start + 90, actual_left),
            ("right", start + 200, start + 110, start + 180, actual_right),
        ):
            references = (
                metadata["left_references"]
                if side == "left"
                else metadata["right_references"]
            )
            translation, rotation = _placement_error(
                poses[landing_node], references[landing_node]
            )
            landing_translation_errors.append(translation)
            landing_rotation_errors.append(rotation)
            ground_z = references[start].translation[2]
            swing_clearances.append(
                float(
                    max(
                        pose.translation[2] - ground_z
                        for pose in poses[swing_start : swing_end + 1]
                    )
                )
            )

    joints = xs[:, 7 : model.nq]
    lower = model.lowerPositionLimit[7 : model.nq]
    upper = model.upperPositionLimit[7 : model.nq]
    diagnostics = {
        "solver_stop": float(solver.stop),
        "solver_iterations": int(solver.iter),
        "solver_cost": float(solver.cost),
        "shooting_gap_norm": float(max(gaps, default=0.0)),
        "rms_shooting_gap_norm": float(np.sqrt(np.mean(np.square(gaps)))),
        "active_support_translation_error_m": max_support_translation,
        "active_support_rotation_error_rad": max_support_rotation,
        "landing_translation_error_m": float(max(landing_translation_errors)),
        "landing_rotation_error_rad": float(max(landing_rotation_errors)),
        "landing_translation_errors_m": landing_translation_errors,
        "landing_rotation_errors_rad": landing_rotation_errors,
        "landing_count": len(landing_translation_errors),
        "swing_clearance_m": float(min(swing_clearances)),
        "swing_clearances_m": swing_clearances,
        "joint_limit_margin_rad": float(
            min(np.min(joints - lower), np.min(upper - joints))
        ),
        "forward_displacement_m": float(np.max(com[:, 1]) - com[0, 1]),
        "duration_s": float(problem.T * DT),
    }
    periodic = _load_periodic_module()
    contact = periodic.compute_periodic_contact_feasibility(
        problem=problem,
        model=model,
        metadata=metadata,
        xs=xs,
        us=us,
    )
    for row in contact["rows"]:
        node = int(row["node"])
        row["cycle_id"] = int(metadata["cycle_id_by_node"][node])
    diagnostics.update(
        {
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
    )
    return diagnostics, com


def solve_segment(
    *, step_count: int, source_path: Path, max_iterations: int = 500
) -> dict:
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
    x0 = load_x0(X0_PATH, model.nq + model.nv)
    problem, state, metadata = build_multistep_problem(
        model, x0, step_count=step_count
    )
    xs_seed, us_seed, provenance = build_continuation_seed(
        problem=problem,
        state=state,
        target_steps=step_count,
        source_path=source_path,
    )
    solver = configure_solver(crocoddyl.SolverFDDP(problem))
    logger = crocoddyl.CallbackLogger()
    solver.setCallbacks([logger])
    returned_success = bool(
        solver.solve(
            [x.copy() for x in xs_seed],
            [u.copy() for u in us_seed],
            int(max_iterations),
            False,
        )
    )
    xs = np.asarray(solver.xs, dtype=float)
    us = np.asarray(solver.us, dtype=float)
    diagnostics, com = compute_multistep_diagnostics(
        model=model,
        problem=problem,
        state=state,
        xs=xs,
        us=us,
        metadata=metadata,
        solver=solver,
    )
    diagnostics["solver_return_success"] = returned_success
    acceptance = evaluate_segment(diagnostics, step_count=step_count)
    history = [
        {
            "iteration": int(logger.iters[index]),
            "cost": float(logger.costs[index]),
            "stop": float(logger.stops[index]),
            "step_length": float(logger.steps[index]),
            "primal_regularization": float(logger.pregs[index]),
            "dual_regularization": float(logger.dregs[index]),
        }
        for index in range(len(logger.costs))
    ]
    return {
        "step_count": step_count,
        "diagnostics": diagnostics,
        "acceptance": acceptance,
        "exported": bool(acceptance["passed"]),
        "seed_provenance": provenance,
        "iteration_history": history,
        "xs": xs,
        "us": us,
        "com": com,
        "metadata": metadata,
        "model": model,
    }


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


def gated_export_planning_candidate(
    *,
    trajectory: Mapping[str, np.ndarray],
    acceptance: Mapping[str, object],
    candidate_path: Path,
    summary_path: Path,
    final_reference_path: Path,
) -> bool:
    candidate_path = Path(candidate_path)
    summary_path = Path(summary_path)
    final_reference_path = Path(final_reference_path)
    if candidate_path.resolve() == final_reference_path.resolve():
        raise ValueError("planning candidate and final reference paths must differ")
    if acceptance.get("passed") is not True:
        return False
    required = {"schema_version", "acceptance_state", "tau_ref_hardware_usage"}
    missing = sorted(required - set(trajectory))
    if missing:
        raise ValueError(f"planning candidate is missing arrays: {missing}")
    if str(np.asarray(trajectory["schema_version"]).item()) != "fddp-multistep-v1":
        raise ValueError("planning candidate schema is invalid")
    if str(np.asarray(trajectory["acceptance_state"]).item()) != "planning_accepted_contact_pending":
        raise ValueError("planning candidate acceptance state is invalid")
    if str(np.asarray(trajectory["tau_ref_hardware_usage"]).item()) != "analysis_only":
        raise ValueError("planning candidate tau_ref boundary is invalid")

    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_candidate = candidate_path.with_suffix(candidate_path.suffix + ".tmp")
    with temporary_candidate.open("wb") as stream:
        np.savez(stream, **trajectory)
    os.replace(temporary_candidate, candidate_path)
    summary = {
        "schema_version": "fddp-multistep-summary-v1",
        "acceptance_state": "planning_accepted_contact_pending",
        "planning_acceptance": dict(acceptance),
        "final_reference_created": False,
        "claim_boundary": "simulation planning candidate; contact feasibility pending",
        "tau_ref_hardware_usage": "analysis_only",
    }
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary_summary.write_text(
        yaml.safe_dump(_yaml_safe(summary), sort_keys=False), encoding="utf-8"
    )
    os.replace(temporary_summary, summary_path)
    return True


def _trajectory_arrays(result: Mapping[str, object], acceptance_state: str) -> dict:
    model = result["model"]
    xs = np.asarray(result["xs"], dtype=float)
    us = np.asarray(result["us"], dtype=float)
    com = np.asarray(result["com"], dtype=float)
    metadata = result["metadata"]
    phases = list(metadata["phase_by_node"])
    cycle_ids = np.asarray(metadata["cycle_id_by_node"], dtype=np.int64)
    return {
        "schema_version": np.asarray("fddp-multistep-v1"),
        "acceptance_state": np.asarray(acceptance_state),
        "step_count": np.asarray(int(result["step_count"]), dtype=np.int64),
        "t_ref": np.arange(xs.shape[0], dtype=float) * DT,
        "q_ref": xs[:, 7 : model.nq],
        "dq_ref": xs[:, model.nq + 6 :],
        "state_ref": xs,
        "tau_ref": us,
        "com_ref_world": com,
        "left_foot_pose_ref": np.asarray(
            [pose.homogeneous for pose in metadata["left_references"]], dtype=float
        ),
        "right_foot_pose_ref": np.asarray(
            [pose.homogeneous for pose in metadata["right_references"]], dtype=float
        ),
        "phase": np.asarray(
            [phase.name for phase in phases] + [phases[-1].name], dtype="U40"
        ),
        "left_contact_active": np.asarray(
            [phase.left_contact_active for phase in phases] + [True], dtype=np.uint8
        ),
        "right_contact_active": np.asarray(
            [phase.right_contact_active for phase in phases] + [True], dtype=np.uint8
        ),
        "cycle_id": np.concatenate((cycle_ids, cycle_ids[-1:])),
        "seed_provenance": np.asarray(str(result["seed_provenance"]["source"])),
        "solver_stop": np.asarray(result["diagnostics"]["solver_stop"], dtype=float),
        "tau_ref_hardware_usage": np.asarray("analysis_only"),
        "claim_boundary": np.asarray("simulation_only"),
    }


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def _write_intermediate_seed(result: Mapping[str, object], path: Path) -> None:
    arrays = _trajectory_arrays(result, "planning_accepted_seed")
    _write_npz_atomic(path, arrays)
    summary_path = path.with_suffix(".yaml")
    summary_path.write_text(
        yaml.safe_dump(
            _yaml_safe(
                {
                    "schema_version": "fddp-multistep-summary-v1",
                    "acceptance_state": "planning_accepted_seed",
                    "step_count": result["step_count"],
                    "planning_acceptance": result["acceptance"],
                    "tau_ref_hardware_usage": "analysis_only",
                }
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def solve_ladder(
    *,
    targets=(4, 6, 10),
    max_iterations: int = 500,
    log_dir: Path = PROJECT_ROOT / "results" / "logs",
    trajectory_dir: Path = PROJECT_ROOT / "transfer" / "trajectories",
) -> dict:
    targets = tuple(int(value) for value in targets)
    if targets != (4, 6, 10):
        raise ValueError("the approved continuation targets are exactly 4, 6, 10")
    log_dir = Path(log_dir)
    trajectory_dir = Path(trajectory_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_yaml = log_dir / "fddp_multistep_planning_diagnostics.yaml"
    iteration_csv = log_dir / "fddp_multistep_continuation_iterations.csv"
    candidate_path = (
        trajectory_dir / "fddp_multistep_10step_planning_candidate.npz"
    )
    candidate_summary = (
        trajectory_dir
        / "fddp_multistep_10step_planning_candidate_summary.yaml"
    )
    final_reference = trajectory_dir / "fddp_multistep_10step_ref.npz"

    source_path = PERIODIC_TRAJECTORY_PATH
    attempted_targets = []
    accepted_targets = []
    segment_summaries = []
    iteration_rows = []
    final_result = None
    for target in targets:
        attempted_targets.append(target)
        result = solve_segment(
            step_count=target,
            source_path=source_path,
            max_iterations=max_iterations,
        )
        final_result = result
        segment_summaries.append(
            {
                "step_count": target,
                "seed_provenance": result["seed_provenance"],
                "diagnostics": result["diagnostics"],
                "acceptance": result["acceptance"],
            }
        )
        for row in result["iteration_history"]:
            iteration_rows.append({"step_count": target, **row})
        if not result["acceptance"]["passed"]:
            break
        accepted_targets.append(target)
        if target < 10:
            source_path = (
                trajectory_dir
                / f"fddp_multistep_{target}step_accepted_seed.npz"
            )
            _write_intermediate_seed(result, source_path)
        else:
            trajectory = _trajectory_arrays(
                result, "planning_accepted_contact_pending"
            )
            gated_export_planning_candidate(
                trajectory=trajectory,
                acceptance=result["acceptance"],
                candidate_path=candidate_path,
                summary_path=candidate_summary,
                final_reference_path=final_reference,
            )

    summary = {
        "schema_version": "fddp-multistep-planning-run-v1",
        "passed": accepted_targets == list(targets),
        "attempted_targets": attempted_targets,
        "accepted_targets": accepted_targets,
        "segments": segment_summaries,
        "planning_candidate": str(candidate_path),
        "final_reference_created": False,
        "claim_boundary": "simulation planning; contact feasibility pending",
        "tau_ref_hardware_usage": "analysis_only",
    }
    diagnostic_yaml.write_text(
        yaml.safe_dump(_yaml_safe(summary), sort_keys=False), encoding="utf-8"
    )
    fieldnames = sorted(
        {key for row in iteration_rows for key in row}
        or {"step_count", "iteration", "cost", "stop"}
    )
    with iteration_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(iteration_rows)
    return {
        "passed": summary["passed"],
        "attempted_targets": attempted_targets,
        "accepted_targets": accepted_targets,
        "segments": segment_summaries,
        "final_result": final_result,
        "diagnostic_yaml": diagnostic_yaml,
        "iteration_csv": iteration_csv,
        "planning_candidate": candidate_path,
    }

