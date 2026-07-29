"""Stage 7A: build and audit the first Crocoddyl double-support FDDP problem.

This file deliberately does NOT run a long optimisation yet. It creates the exact
contact-constrained optimal-control problem that FDDP will solve in the next stage,
and performs one finite dynamics/cost evaluation. This avoids hiding Crocoddyl API
or contact-frame errors inside a solver failure.
"""
from __future__ import annotations

from pathlib import Path
import textwrap
import numpy as np
import yaml
import pinocchio as pin
import crocoddyl

PROJECT = Path.home() / "biped_fddp_project"
URDF_PATH = PROJECT / "model" / "biped_v1.urdf"
INIT_PATH = PROJECT / "transfer" / "initialisation" / "fddp_double_support_x0.npz"
CONTACT_PATH = PROJECT / "transfer" / "initialisation" / "fddp_double_support_contacts.yaml"
OUT_PATH = PROJECT / "transfer" / "initialisation" / "fddp_double_support_problem_manifest.yaml"

LEFT_FRAME = "left_foot_contact_fixed"
RIGHT_FRAME = "right_foot_contact_fixed"
DT = 0.01
HORIZON_NODES = 40


def frame_id_or_fail(model: pin.Model, frame_name: str) -> int:
    frame_id = model.getFrameId(frame_name)
    if frame_id >= model.nframes:
        raise RuntimeError(f"Frame not found: {frame_name}")
    return frame_id


def make_contact_6d(state, frame_id: int, reference: pin.SE3, nu: int):
    """Support both common Python binding signatures for ContactModel6D."""
    gains = np.array([0.0, 0.0])
    attempts = []

    # Current / widely used binding: state, frame_id, SE3, reference-frame, nu, gains
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
    except Exception as exc:  # Keep diagnostics for older bindings.
        attempts.append(f"signature A: {type(exc).__name__}: {exc}")

    # Older binding: FramePlacement wrapper.
    try:
        frame_placement = crocoddyl.FramePlacement(frame_id, reference)
        contact = crocoddyl.ContactModel6D(state, frame_placement, nu, gains)
        return contact, "state, FramePlacement(frame_id, SE3), nu, gains"
    except Exception as exc:
        attempts.append(f"signature B: {type(exc).__name__}: {exc}")

    joined = "\n  ".join(attempts)
    raise RuntimeError(
        "Could not create ContactModel6D with either supported Python binding.\n"
        f"  {joined}\n\n"
        "Copy this full error into the chat; do not modify the file manually."
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


def main() -> None:
    for path in (URDF_PATH, INIT_PATH, CONTACT_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Required input does not exist: {path}\n"
                "Run Stage 6B first and ensure the files are in transfer/initialisation/."
            )

    archive = np.load(INIT_PATH)
    x0 = np.asarray(archive["x0"], dtype=float)
    left_translation = np.asarray(archive["left_contact_translation"], dtype=float)
    left_rotation = np.asarray(archive["left_contact_rotation"], dtype=float)
    right_translation = np.asarray(archive["right_contact_translation"], dtype=float)
    right_rotation = np.asarray(archive["right_contact_rotation"], dtype=float)

    model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
    if x0.shape != (model.nq + model.nv,):
        raise RuntimeError(
            f"x0 has shape {x0.shape}; expected {(model.nq + model.nv,)}."
        )

    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    nu = int(actuation.nu)

    left_id = frame_id_or_fail(model, LEFT_FRAME)
    right_id = frame_id_or_fail(model, RIGHT_FRAME)
    left_reference = pin.SE3(left_rotation, left_translation)
    right_reference = pin.SE3(right_rotation, right_translation)

    contacts = crocoddyl.ContactModelMultiple(state, nu)
    left_contact, left_signature = make_contact_6d(
        state, left_id, left_reference, nu
    )
    right_contact, right_signature = make_contact_6d(
        state, right_id, right_reference, nu
    )
    contacts.addContact("left_foot", left_contact, True)
    contacts.addContact("right_foot", right_contact, True)

    costs = crocoddyl.CostModelSum(state, nu)
    state_residual, state_signature = make_state_residual(state, x0, nu)
    control_residual, control_signature = make_control_residual(state, nu)

    # Keep the first stand problem deliberately simple. The state term holds the
    # contact-consistent posture; the control term produces a smooth, finite torque
    # solution. Task-specific costs are added only after this construction is verified.
    state_cost = crocoddyl.CostModelResidual(state, state_residual)
    control_cost = crocoddyl.CostModelResidual(state, control_residual)
    costs.addCost("state_regularisation", state_cost, 10.0)
    costs.addCost("control_regularisation", control_cost, 1e-3)

    differential = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state,
        actuation,
        contacts,
        costs,
        0.0,
        True,
    )
    running_model = crocoddyl.IntegratedActionModelEuler(differential, DT)
    terminal_model = crocoddyl.IntegratedActionModelEuler(differential, 0.0)

    problem = crocoddyl.ShootingProblem(
        x0,
        [running_model] * HORIZON_NODES,
        terminal_model,
    )

    # One local evaluation establishes that contact dynamics and costs are finite.
    action_data = running_model.createData()
    u0 = np.zeros(nu)
    running_model.calc(action_data, x0, u0)
    finite_cost = bool(np.isfinite(action_data.cost))
    finite_next = bool(np.all(np.isfinite(action_data.xnext)))

    if not (finite_cost and finite_next):
        raise RuntimeError(
            "The constructed action model returned non-finite cost or xnext.\n"
            f"cost={action_data.cost}, finite_xnext={finite_next}"
        )

    manifest = {
        "stage": "7A_double_support_fddp_problem_build",
        "urdf": str(URDF_PATH),
        "initialisation_file": str(INIT_PATH),
        "contact_file": str(CONTACT_PATH),
        "model": {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "ndx": int(state.ndx),
            "nu": int(nu),
        },
        "contacts": {
            "left_frame": LEFT_FRAME,
            "left_frame_id": int(left_id),
            "right_frame": RIGHT_FRAME,
            "right_frame_id": int(right_id),
            "reference_frame": "LOCAL_WORLD_ALIGNED",
            "contact_signature": left_signature,
        },
        "discretisation": {
            "dt_s": DT,
            "horizon_nodes": HORIZON_NODES,
            "horizon_s": DT * HORIZON_NODES,
        },
        "costs": {
            "state_regularisation_weight": 10.0,
            "control_regularisation_weight": 0.001,
            "state_residual_signature": state_signature,
            "control_residual_signature": control_signature,
        },
        "one_step_evaluation": {
            "cost": float(action_data.cost),
            "finite_cost": finite_cost,
            "finite_xnext": finite_next,
        },
        "notes": [
            "This file verifies the first double-support FDDP problem before optimisation.",
            "Both feet are ideal rigid 6D contacts at the Stage 6B world-frame references.",
            "The next stage uses this same construction to run SolverFDDP and export stand_ref.npz.",
        ],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    doc = (getattr(crocoddyl.ContactModel6D, "__doc__", "") or "").strip()
    doc_preview = " ".join(doc.split())[:220]

    print("===== STAGE 7A: DOUBLE-SUPPORT FDDP PROBLEM BUILD =====")
    print(f"Crocoddyl module: {getattr(crocoddyl, '__file__', 'unknown')}")
    print(f"URDF: {URDF_PATH}")
    print(
        f"Model dimensions: nq={model.nq}, nv={model.nv}, "
        f"ndx={state.ndx}, nu={nu}"
    )
    print(
        f"Horizon: {HORIZON_NODES} nodes x {DT:.3f} s = "
        f"{HORIZON_NODES * DT:.3f} s"
    )
    print("\nRigid support contacts:")
    print(f"  left : {LEFT_FRAME}  (frame id {left_id})")
    print(f"  right: {RIGHT_FRAME} (frame id {right_id})")
    print(f"  ContactModel6D binding used: {left_signature}")
    print("\nCosts:")
    print(f"  state regularisation: weight=10.0 ({state_signature})")
    print(f"  control regularisation: weight=0.001 ({control_signature})")
    print("\nOne-step action evaluation:")
    print(f"  cost={action_data.cost:.8f}")
    print(f"  finite xnext={finite_next}")
    print("\nContactModel6D docstring preview:")
    print(textwrap.fill(doc_preview if doc_preview else "<not available>", width=88))
    print("\nSaved:")
    print(f"  {OUT_PATH}")
    print("\nDOUBLE-SUPPORT FDDP PROBLEM BUILD PASSED")


if __name__ == "__main__":
    main()
