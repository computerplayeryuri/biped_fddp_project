"""Stage 7B: solve and export the first double-support FDDP standing reference.

This is intentionally a *hold-standing* task. Both feet are ideal rigid 6D contacts
and the reference state is the contact-consistent Stage 6B pose. Therefore the
mathematically correct optimum is a stationary reference. A zero or near-zero final
cost is expected and confirms the FDDP pipeline is working.

The purpose of this first solve is to verify:
  URDF -> Pinocchio free flyer -> rigid contacts -> Crocoddyl SolverFDDP -> export
before attempting non-trivial tasks such as CoM shift, foot unloading and stepping.
"""
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import yaml
import pinocchio as pin
import crocoddyl

PROJECT = Path.home() / "biped_fddp_project"
URDF_PATH = PROJECT / "model" / "biped_v1.urdf"
INIT_PATH = PROJECT / "transfer" / "initialisation" / "fddp_double_support_x0.npz"
TRAJ_DIR = PROJECT / "transfer" / "trajectories"
SUMMARY_PATH = PROJECT / "transfer" / "trajectories" / "stand_ref_summary.yaml"
TRAJ_PATH = PROJECT / "transfer" / "trajectories" / "stand_ref.npz"

LEFT_FRAME = "left_foot_contact_fixed"
RIGHT_FRAME = "right_foot_contact_fixed"
DT = 0.01
HORIZON_NODES = 40
MAX_ITER = 50


def frame_id_or_fail(model: pin.Model, frame_name: str) -> int:
    frame_id = model.getFrameId(frame_name)
    if frame_id >= model.nframes:
        raise RuntimeError(f"Frame not found: {frame_name}")
    return frame_id


def make_contact_6d(state, frame_id: int, reference: pin.SE3, nu: int):
    """Support both common ContactModel6D Python binding signatures."""
    gains = np.array([0.0, 0.0])
    diagnostics = []

    try:
        contact = crocoddyl.ContactModel6D(
            state,
            frame_id,
            reference,
            pin.LOCAL_WORLD_ALIGNED,
            nu,
            gains,
        )
        return contact, "state, frame_id, SE3, LOCAL_WORLD_ALIGNED, nu, gains"
    except Exception as exc:
        diagnostics.append(f"signature A: {type(exc).__name__}: {exc}")

    try:
        frame_placement = crocoddyl.FramePlacement(frame_id, reference)
        contact = crocoddyl.ContactModel6D(state, frame_placement, nu, gains)
        return contact, "state, FramePlacement(frame_id, SE3), nu, gains"
    except Exception as exc:
        diagnostics.append(f"signature B: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "Could not create ContactModel6D.\n  " + "\n  ".join(diagnostics)
    )


def make_state_residual(state, xref: np.ndarray, nu: int):
    try:
        return crocoddyl.ResidualModelState(state, xref, nu), "state, xref, nu"
    except Exception:
        return crocoddyl.ResidualModelState(state, xref), "state, xref"


def make_control_residual(state, nu: int):
    uref = np.zeros(nu)
    try:
        return crocoddyl.ResidualModelControl(state, uref, nu), "state, uref, nu"
    except Exception:
        return crocoddyl.ResidualModelControl(state, uref), "state, uref"


def make_action_model(
    state,
    actuation,
    left_id: int,
    right_id: int,
    left_reference: pin.SE3,
    right_reference: pin.SE3,
    xref: np.ndarray,
    dt: float,
    state_weight: float,
    control_weight: float,
):
    nu = int(actuation.nu)
    contacts = crocoddyl.ContactModelMultiple(state, nu)
    left_contact, contact_signature = make_contact_6d(
        state, left_id, left_reference, nu
    )
    right_contact, _ = make_contact_6d(
        state, right_id, right_reference, nu
    )
    contacts.addContact("left_foot", left_contact, True)
    contacts.addContact("right_foot", right_contact, True)

    costs = crocoddyl.CostModelSum(state, nu)
    state_residual, state_signature = make_state_residual(state, xref, nu)
    control_residual, control_signature = make_control_residual(state, nu)
    costs.addCost(
        "state_regularisation",
        crocoddyl.CostModelResidual(state, state_residual),
        state_weight,
    )
    costs.addCost(
        "control_regularisation",
        crocoddyl.CostModelResidual(state, control_residual),
        control_weight,
    )

    differential = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state,
        actuation,
        contacts,
        costs,
        0.0,
        True,
    )
    integrated = crocoddyl.IntegratedActionModelEuler(differential, dt)
    return integrated, contact_signature, state_signature, control_signature


def support_errors(
    model: pin.Model,
    q: np.ndarray,
    left_id: int,
    right_id: int,
    left_ref: pin.SE3,
    right_ref: pin.SE3,
) -> tuple[float, float]:
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    left_error = float(np.linalg.norm(data.oMf[left_id].translation - left_ref.translation))
    right_error = float(np.linalg.norm(data.oMf[right_id].translation - right_ref.translation))
    return left_error, right_error


def main() -> None:
    if not URDF_PATH.exists():
        raise FileNotFoundError(f"URDF not found: {URDF_PATH}")
    if not INIT_PATH.exists():
        raise FileNotFoundError(
            f"Initialisation not found: {INIT_PATH}\nRun Stage 6B first."
        )

    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    archive = np.load(INIT_PATH)
    q0 = np.asarray(archive["q0"], dtype=float)
    v0 = np.asarray(archive["v0"], dtype=float)
    x0 = np.asarray(archive["x0"], dtype=float)

    left_reference = pin.SE3(
        np.asarray(archive["left_contact_rotation"], dtype=float),
        np.asarray(archive["left_contact_translation"], dtype=float),
    )
    right_reference = pin.SE3(
        np.asarray(archive["right_contact_rotation"], dtype=float),
        np.asarray(archive["right_contact_translation"], dtype=float),
    )

    model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
    if x0.shape != (model.nq + model.nv,):
        raise RuntimeError(
            f"x0 shape {x0.shape} does not match expected {(model.nq + model.nv,)}"
        )

    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    nu = int(actuation.nu)
    left_id = frame_id_or_fail(model, LEFT_FRAME)
    right_id = frame_id_or_fail(model, RIGHT_FRAME)

    running_model, contact_signature, state_signature, control_signature = make_action_model(
        state,
        actuation,
        left_id,
        right_id,
        left_reference,
        right_reference,
        x0,
        DT,
        state_weight=10.0,
        control_weight=1e-3,
    )
    terminal_model, _, _, _ = make_action_model(
        state,
        actuation,
        left_id,
        right_id,
        left_reference,
        right_reference,
        x0,
        0.0,
        state_weight=100.0,
        control_weight=0.0,
    )

    problem = crocoddyl.ShootingProblem(
        x0,
        [running_model] * HORIZON_NODES,
        terminal_model,
    )

    # The first reference is a contact-consistent hold. The constant state / zero
    # control trajectory is both a physically meaningful and very safe initial guess.
    xs_init = [x0.copy() for _ in range(HORIZON_NODES + 1)]
    us_init = [np.zeros(nu) for _ in range(HORIZON_NODES)]

    problem.calc(xs_init, us_init)
    initial_cost = (
        sum(float(data.cost) for data in problem.runningDatas)
        + float(problem.terminalData.cost)
    )

    solver = crocoddyl.SolverFDDP(problem)
    solver.th_stop = 1e-9
    success = bool(solver.solve(xs_init, us_init, MAX_ITER, False))

    xs = np.asarray(solver.xs, dtype=float)
    us = np.asarray(solver.us, dtype=float)
    if xs.shape != (HORIZON_NODES + 1, model.nq + model.nv):
        raise RuntimeError(
            f"Unexpected xs shape {xs.shape}; expected "
            f"{(HORIZON_NODES + 1, model.nq + model.nv)}"
        )
    if us.shape != (HORIZON_NODES, nu):
        raise RuntimeError(
            f"Unexpected us shape {us.shape}; expected {(HORIZON_NODES, nu)}"
        )

    problem.calc([x.copy() for x in xs], [u.copy() for u in us])
    final_cost = (
        sum(float(data.cost) for data in problem.runningDatas)
        + float(problem.terminalData.cost)
    )

    # Pinocchio free-flyer layout: first 7 q entries are root translation/quaternion;
    # first 6 v entries are root spatial velocity.
    q_ref = xs[:, 7:model.nq]
    dq_ref = xs[:, model.nq + 6:]
    t_ref = np.arange(HORIZON_NODES + 1, dtype=float) * DT
    phase = np.full(HORIZON_NODES + 1, "double_support", dtype="U16")

    max_left_error = 0.0
    max_right_error = 0.0
    for x in xs:
        left_error, right_error = support_errors(
            model, x[:model.nq], left_id, right_id, left_reference, right_reference
        )
        max_left_error = max(max_left_error, left_error)
        max_right_error = max(max_right_error, right_error)

    if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(us))):
        raise RuntimeError("FDDP output contains non-finite values.")

    np.savez_compressed(
        TRAJ_PATH,
        time_s=t_ref,
        xs=xs,
        us=us,
        q_ref=q_ref,
        dq_ref=dq_ref,
        contact_phase=phase,
        x0=x0,
        dt_s=np.array(DT),
        horizon_nodes=np.array(HORIZON_NODES),
        joint_names=np.array(
            [
                "left_hip_pitch_joint",
                "left_knee_pitch_joint",
                "left_ankle_pitch_joint",
                "right_hip_pitch_joint",
                "right_knee_pitch_joint",
                "right_ankle_pitch_joint",
            ],
            dtype="U32",
        ),
    )

    summary = {
        "stage": "7B_double_support_fddp_stand_solve",
        "task": "ideal_rigid_double_support_hold",
        "purpose": (
            "Validate the full Crocoddyl SolverFDDP pipeline and export the first "
            "contact-consistent standing reference. This task is intentionally "
            "stationary; a near-zero optimal cost is expected."
        ),
        "model": {
            "urdf": str(URDF_PATH),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "ndx": int(state.ndx),
            "nu": int(nu),
        },
        "solver": {
            "name": "SolverFDDP",
            "max_iterations": int(MAX_ITER),
            "success": success,
            "stop_criterion": float(solver.stop),
            "initial_cost": initial_cost,
            "final_cost": final_cost,
        },
        "contacts": {
            "left_frame": LEFT_FRAME,
            "right_frame": RIGHT_FRAME,
            "binding_signature": contact_signature,
            "max_left_translation_error_m": max_left_error,
            "max_right_translation_error_m": max_right_error,
        },
        "discretisation": {
            "dt_s": DT,
            "horizon_nodes": HORIZON_NODES,
            "horizon_s": HORIZON_NODES * DT,
        },
        "costs": {
            "running_state_weight": 10.0,
            "running_control_weight": 0.001,
            "terminal_state_weight": 100.0,
            "state_residual_signature": state_signature,
            "control_residual_signature": control_signature,
        },
        "outputs": {
            "trajectory_npz": str(TRAJ_PATH),
            "q_ref_shape": [int(v) for v in q_ref.shape],
            "dq_ref_shape": [int(v) for v in dq_ref.shape],
            "us_shape": [int(v) for v in us.shape],
        },
    }
    with SUMMARY_PATH.open("w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    q_range_deg = np.rad2deg(q_ref.max(axis=0) - q_ref.min(axis=0))
    max_tau = float(np.max(np.abs(us))) if us.size else 0.0

    print("===== STAGE 7B: DOUBLE-SUPPORT FDDP STANDING SOLVE =====")
    print("Task: ideal rigid double-support hold standing")
    print("This first FDDP task is intentionally stationary.")
    print("A near-zero final cost and near-zero joint-motion range are expected.")
    print(f"URDF: {URDF_PATH}")
    print(f"Horizon: {HORIZON_NODES} nodes x {DT:.3f} s = {HORIZON_NODES * DT:.3f} s")
    print(f"Solver success: {success}")
    print(f"Solver stop criterion: {solver.stop:.3e}")
    print(f"Initial cost: {initial_cost:.10f}")
    print(f"Final cost:   {final_cost:.10f}")
    print(f"Maximum |virtual torque|: {max_tau:.6f} N m")
    print(
        "Maximum left/right support-frame translation error: "
        f"{max_left_error:.3e} / {max_right_error:.3e} m"
    )
    print(
        "Joint range over reference (deg): "
        + ", ".join(f"{value:.6f}" for value in q_range_deg)
    )
    print("\nSaved:")
    print(f"  {TRAJ_PATH}")
    print(f"  {SUMMARY_PATH}")
    print("\nDOUBLE-SUPPORT FDDP STANDING SOLVE PASSED")


if __name__ == "__main__":
    main()
