#!/usr/bin/env python3
"""
STAGE 7C DIAGNOSTIC-4 — 3D-contact static-equilibrium seed audit.

Purpose
-------
Test the likely reason why 03i failed at its first 0 mm continuation stage:

    Two ContactModel3D foot contacts constrain foot translation but do not
    constrain foot orientation / ground moment. Therefore, x0 with u=0 may
    not be a static equilibrium even though the CoM target is zero.

This script builds exactly the 0 mm 3D-contact problem and compares:
  A) repeated x0 + zero controls;
  B) repeated x0 + quasiStatic controls;
  C) explicit dynamics rollout of the quasiStatic controls;
  D) SolverFDDP started from the quasiStatic static seed.

It does not overwrite trajectory reference files.
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import yaml
import pinocchio as pin
import crocoddyl


DT = 0.010
HORIZON_NODES = 120
MAX_ITER = 100

LEFT_FOOT_FRAME = "left_foot_contact_fixed"
RIGHT_FOOT_FRAME = "right_foot_contact_fixed"

CONTACT_GAINS_KP_KD = np.array([0.0, 50.0], dtype=float)

RUNNING_STATE_WEIGHT = 0.20
RUNNING_CONTROL_WEIGHT = 1e-3
TERMINAL_STATE_WEIGHT = 10.0
COM_TRACKING_WEIGHT_Y = 1.0e4
COM_TRACKING_WEIGHT_XZ = 2.0e2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = PROJECT_ROOT / "model" / "biped_v1.urdf"
X0_PATH = PROJECT_ROOT / "transfer" / "initialisation" / "fddp_double_support_x0.npz"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "transfer"
    / "initialisation"
    / "three_d_contact_static_seed_audit.yaml"
)


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
                f"No x0-like key in {path}; available: {loaded.files}"
            )

    if x0.size != expected_size:
        raise RuntimeError(f"x0 size {x0.size}; expected {expected_size}.")
    return x0


def total_cost(problem, xs, us) -> float:
    problem.calc(xs, us)
    return float(
        sum(float(data.cost) for data in problem.runningDatas)
        + float(problem.terminalData.cost)
    )


def get_support_references(model, x0, left_frame_id, right_frame_id):
    data = model.createData()
    pin.forwardKinematics(model, data, x0[:model.nq])
    pin.updateFramePlacements(model, data)
    return (
        data.oMf[left_frame_id].translation.copy(),
        data.oMf[right_frame_id].translation.copy(),
        data.oMf[left_frame_id].rotation.copy(),
        data.oMf[right_frame_id].rotation.copy(),
    )


def make_contact_3d(state, nu, frame_id, translation_reference):
    return crocoddyl.ContactModel3D(
        state,
        frame_id,
        translation_reference,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        nu,
        CONTACT_GAINS_KP_KD,
    )


def build_zero_shift_problem(
    model,
    x0,
    left_frame_id,
    right_frame_id,
    left_translation_reference,
    right_translation_reference,
):
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    nu = actuation.nu

    data = model.createData()
    initial_com = np.asarray(
        pin.centerOfMass(model, data, x0[:model.nq]), dtype=float
    ).reshape(3)

    def make_action(state_weight, control_weight, dt):
        contacts = crocoddyl.ContactModelMultiple(state, nu)
        contacts.addContact(
            LEFT_FOOT_FRAME,
            make_contact_3d(
                state, nu, left_frame_id, left_translation_reference
            ),
        )
        contacts.addContact(
            RIGHT_FOOT_FRAME,
            make_contact_3d(
                state, nu, right_frame_id, right_translation_reference
            ),
        )

        costs = crocoddyl.CostModelSum(state, nu)

        state_residual = crocoddyl.ResidualModelState(state, x0, nu)
        costs.addCost(
            "state_regularisation",
            crocoddyl.CostModelResidual(state, state_residual),
            state_weight,
        )

        control_residual = crocoddyl.ResidualModelControl(state, np.zeros(nu))
        costs.addCost(
            "control_regularisation",
            crocoddyl.CostModelResidual(state, control_residual),
            control_weight,
        )

        com_residual = crocoddyl.ResidualModelCoMPosition(
            state, initial_com, nu
        )
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
            0.0,
            True,
        )
        return crocoddyl.IntegratedActionModelEuler(differential, dt)

    running = [
        make_action(RUNNING_STATE_WEIGHT, RUNNING_CONTROL_WEIGHT, DT)
        for _ in range(HORIZON_NODES)
    ]
    terminal = make_action(TERMINAL_STATE_WEIGHT, 0.0, 0.0)
    problem = crocoddyl.ShootingProblem(x0, running, terminal)
    return problem, state, nu, initial_com


def calculate_gaps(problem, state, xs, us):
    problem.calc(xs, us)
    values = []
    for k, data in enumerate(problem.runningDatas):
        gap = np.asarray(
            state.diff(xs[k + 1], data.xnext), dtype=float
        ).reshape(-1)
        values.append(float(np.linalg.norm(gap)))

    values = np.asarray(values, dtype=float)
    return {
        "max_gap_norm": float(np.max(values)),
        "rms_gap_norm": float(np.sqrt(np.mean(values**2))),
        "first_gap_norm": float(values[0]),
        "last_gap_norm": float(values[-1]),
    }


def quasi_static_controls(problem, xs, nu):
    """
    Current user binding exposes quasiStatic(xs) -> controls.
    Try a running-node list first, then full state list for compatibility.
    """
    attempts = [
        ("running_nodes_only", [np.asarray(x).copy() for x in xs[:-1]]),
        ("full_state_sequence", [np.asarray(x).copy() for x in xs]),
    ]

    errors = []
    for mode, argument in attempts:
        try:
            output = problem.quasiStatic(argument)
            controls = [
                np.asarray(u, dtype=float).reshape(-1).copy()
                for u in output
            ]
            if len(controls) != HORIZON_NODES:
                raise RuntimeError(
                    f"returned {len(controls)} controls; expected {HORIZON_NODES}"
                )
            if any(u.shape != (nu,) for u in controls):
                raise RuntimeError("returned a control with unexpected shape")
            return controls, mode
        except Exception as exc:
            errors.append(f"{mode}: {exc}")

    raise RuntimeError(" | ".join(errors))


def explicit_rollout(problem, us):
    xs = [np.asarray(problem.x0, dtype=float).copy()]
    for k, u in enumerate(us):
        model_k = problem.runningModels[k]
        data_k = problem.runningDatas[k]
        model_k.calc(data_k, xs[-1], u)
        xs.append(np.asarray(data_k.xnext, dtype=float).copy())
    return xs


def support_and_state_deviation(
    model,
    xs,
    x0,
    left_frame_id,
    right_frame_id,
    left_translation_reference,
    right_translation_reference,
    left_rotation_reference,
    right_rotation_reference,
):
    data = model.createData()
    max_state_diff = 0.0
    max_left_translation = 0.0
    max_right_translation = 0.0
    max_left_rotation = 0.0
    max_right_rotation = 0.0

    for x in xs:
        max_state_diff = max(
            max_state_diff,
            float(np.linalg.norm(pin.difference(model, x0[:model.nq], x[:model.nq]))),
        )

        pin.forwardKinematics(model, data, x[:model.nq])
        pin.updateFramePlacements(model, data)

        max_left_translation = max(
            max_left_translation,
            float(
                np.linalg.norm(
                    data.oMf[left_frame_id].translation
                    - left_translation_reference
                )
            ),
        )
        max_right_translation = max(
            max_right_translation,
            float(
                np.linalg.norm(
                    data.oMf[right_frame_id].translation
                    - right_translation_reference
                )
            ),
        )
        max_left_rotation = max(
            max_left_rotation,
            float(
                np.linalg.norm(
                    pin.log3(
                        left_rotation_reference.T
                        @ data.oMf[left_frame_id].rotation
                    )
                )
            ),
        )
        max_right_rotation = max(
            max_right_rotation,
            float(
                np.linalg.norm(
                    pin.log3(
                        right_rotation_reference.T
                        @ data.oMf[right_frame_id].rotation
                    )
                )
            ),
        )

    return {
        "max_configuration_difference_norm": float(max_state_diff),
        "max_left_translation_error_m": float(max_left_translation),
        "max_right_translation_error_m": float(max_right_translation),
        "max_left_rotation_error_rad": float(max_left_rotation),
        "max_right_rotation_error_rad": float(max_right_rotation),
    }


def safe_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def main():
    print("===== STAGE 7C DIAGNOSTIC-4: 3D-CONTACT STATIC SEED AUDIT =====")
    print("No reference trajectory is overwritten.")
    print(f"URDF: {URDF_PATH}")
    print(f"Contact gains [Kp, Kd]: {CONTACT_GAINS_KP_KD.tolist()}")

    model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
    if model.nq != 13 or model.nv != 12:
        raise RuntimeError(
            f"Unexpected model dimensions nq={model.nq}, nv={model.nv}"
        )

    x0 = load_x0(X0_PATH, model.nq + model.nv)
    left_frame_id = model.getFrameId(LEFT_FOOT_FRAME)
    right_frame_id = model.getFrameId(RIGHT_FOOT_FRAME)
    (
        left_translation_reference,
        right_translation_reference,
        left_rotation_reference,
        right_rotation_reference,
    ) = get_support_references(model, x0, left_frame_id, right_frame_id)

    problem, state, nu, initial_com = build_zero_shift_problem(
        model,
        x0,
        left_frame_id,
        right_frame_id,
        left_translation_reference,
        right_translation_reference,
    )

    xs_static = [x0.copy() for _ in range(HORIZON_NODES + 1)]
    us_zero = [np.zeros(nu, dtype=float) for _ in range(HORIZON_NODES)]

    zero_cost = total_cost(problem, xs_static, us_zero)
    zero_gaps = calculate_gaps(problem, state, xs_static, us_zero)

    print()
    print("A. Repeated x0 with zero controls")
    print(f"  cost: {zero_cost:.10e}")
    print(
        "  gaps max/RMS: "
        f"{zero_gaps['max_gap_norm']:.3e} / {zero_gaps['rms_gap_norm']:.3e}"
    )

    quasi_controls, binding_mode = quasi_static_controls(problem, xs_static, nu)
    quasi_cost = total_cost(problem, xs_static, quasi_controls)
    quasi_gaps = calculate_gaps(problem, state, xs_static, quasi_controls)
    max_abs_quasi_u = float(
        max(np.max(np.abs(u)) for u in quasi_controls)
    )
    rms_quasi_u = float(
        np.sqrt(
            np.mean(
                np.concatenate([u.reshape(-1) for u in quasi_controls]) ** 2
            )
        )
    )

    print()
    print("B. Repeated x0 with quasiStatic controls")
    print(f"  quasiStatic binding state-list mode: {binding_mode}")
    print(f"  control max/RMS: {max_abs_quasi_u:.6f} / {rms_quasi_u:.6f}")
    print(f"  cost: {quasi_cost:.10e}")
    print(
        "  gaps max/RMS: "
        f"{quasi_gaps['max_gap_norm']:.3e} / {quasi_gaps['rms_gap_norm']:.3e}"
    )

    rollout_xs = explicit_rollout(problem, quasi_controls)
    rollout_cost = total_cost(problem, rollout_xs, quasi_controls)
    rollout_gaps = calculate_gaps(problem, state, rollout_xs, quasi_controls)
    rollout_deviation = support_and_state_deviation(
        model,
        rollout_xs,
        x0,
        left_frame_id,
        right_frame_id,
        left_translation_reference,
        right_translation_reference,
        left_rotation_reference,
        right_rotation_reference,
    )

    print()
    print("C. Dynamics rollout from x0 under quasiStatic controls")
    print(f"  cost: {rollout_cost:.10e}")
    print(
        "  gaps max/RMS: "
        f"{rollout_gaps['max_gap_norm']:.3e} / {rollout_gaps['rms_gap_norm']:.3e}"
    )
    print(
        "  max configuration deviation / L-R foot translation (mm): "
        f"{rollout_deviation['max_configuration_difference_norm']:.3e} / "
        f"{rollout_deviation['max_left_translation_error_m'] * 1000.0:.4f} / "
        f"{rollout_deviation['max_right_translation_error_m'] * 1000.0:.4f}"
    )
    print(
        "  max L-R foot rotation (deg): "
        f"{np.rad2deg(rollout_deviation['max_left_rotation_error_rad']):.4f} / "
        f"{np.rad2deg(rollout_deviation['max_right_rotation_error_rad']):.4f}"
    )

    solver = crocoddyl.SolverFDDP(problem)
    solver.th_stop = 1e-7
    solver_return = bool(
        solver.solve(
            [x.copy() for x in xs_static],
            [u.copy() for u in quasi_controls],
            MAX_ITER,
            False,
        )
    )
    stop_passed = bool(float(solver.stop) <= float(solver.th_stop))
    accepted = bool(solver_return or stop_passed)

    solver_xs = [np.asarray(x, dtype=float).copy() for x in solver.xs]
    solver_us = [np.asarray(u, dtype=float).copy() for u in solver.us]
    solver_cost = total_cost(problem, solver_xs, solver_us)
    solver_gaps = calculate_gaps(problem, state, solver_xs, solver_us)
    solver_deviation = support_and_state_deviation(
        model,
        solver_xs,
        x0,
        left_frame_id,
        right_frame_id,
        left_translation_reference,
        right_translation_reference,
        left_rotation_reference,
        right_rotation_reference,
    )

    print()
    print("D. SolverFDDP from static quasiStatic seed")
    print(f"  solver return / stop / accepted: {solver_return} / {solver.stop:.3e} / {accepted}")
    print(f"  cost: {solver_cost:.10e}")
    print(
        "  gaps max/RMS: "
        f"{solver_gaps['max_gap_norm']:.3e} / {solver_gaps['rms_gap_norm']:.3e}"
    )
    print(
        "  max configuration deviation / L-R foot translation (mm): "
        f"{solver_deviation['max_configuration_difference_norm']:.3e} / "
        f"{solver_deviation['max_left_translation_error_m'] * 1000.0:.4f} / "
        f"{solver_deviation['max_right_translation_error_m'] * 1000.0:.4f}"
    )
    print(
        "  max L-R foot rotation (deg): "
        f"{np.rad2deg(solver_deviation['max_left_rotation_error_rad']):.4f} / "
        f"{np.rad2deg(solver_deviation['max_right_rotation_error_rad']):.4f}"
    )

    interpretation = (
        "If section B and C remain close to x0, and section D is accepted, "
        "the revised FDDP continuation should initialise its 0 mm stage with "
        "the quasi-static controls rather than zeros."
        if accepted and rollout_deviation["max_configuration_difference_norm"] < 1e-3
        else
        "The 3D translation-contact static state is not a suitable FDDP seed "
        "under the current model/cost formulation. Do not continue with 03i."
    )

    summary = {
        "stage": "7C_3d_contact_static_seed_audit",
        "purpose": (
            "Determine whether 3D translational support contacts require "
            "non-zero static control input at x0, and whether that seed permits "
            "a stable zero-shift FDDP solve."
        ),
        "model": {
            "urdf": str(URDF_PATH),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(nu),
        },
        "contact_model": {
            "type": "two ContactModel3D translational contacts",
            "gains_Kp_Kd": [float(v) for v in CONTACT_GAINS_KP_KD],
        },
        "zero_control_static_seed": {
            "cost": safe_float(zero_cost),
            "shooting_gaps": zero_gaps,
        },
        "quasi_static_static_seed": {
            "binding_state_list_mode": binding_mode,
            "control_max_abs": safe_float(max_abs_quasi_u),
            "control_rms": safe_float(rms_quasi_u),
            "cost": safe_float(quasi_cost),
            "shooting_gaps": quasi_gaps,
        },
        "quasi_static_rollout": {
            "cost": safe_float(rollout_cost),
            "shooting_gaps": rollout_gaps,
            **{key: safe_float(value) for key, value in rollout_deviation.items()},
        },
        "fddp_from_quasi_static_static_seed": {
            "max_iterations": int(MAX_ITER),
            "return_success": bool(solver_return),
            "stop_criterion": safe_float(solver.stop),
            "threshold": float(solver.th_stop),
            "accepted_convergence": bool(accepted),
            "cost": safe_float(solver_cost),
            "shooting_gaps": solver_gaps,
            **{key: safe_float(value) for key, value in solver_deviation.items()},
        },
        "interpretation": interpretation,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        yaml.safe_dump(summary, sort_keys=False),
        encoding="utf-8",
    )

    print()
    print("Interpretation:")
    print(f"  {interpretation}")
    print(f"Saved:\n  {OUTPUT_PATH}")
    print("3D-CONTACT STATIC SEED AUDIT COMPLETED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n3D-CONTACT STATIC SEED AUDIT FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
