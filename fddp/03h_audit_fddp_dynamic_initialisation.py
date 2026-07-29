#!/usr/bin/env python3
"""
STAGE 7C DIAGNOSTIC-3 — Dynamic initialisation audit for FDDP.

This script does NOT run SolverFDDP and does NOT overwrite any reference
trajectory.

It evaluates why the prior FDDP solve diverged:
  1) The Stage 7C kinematic warm-start provides xs_init but uses us_init = 0.
  2) A low trajectory cost does not prove dynamic feasibility.
  3) Crocoddyl ShootingProblem.quasiStatic() can compute contact-consistent
     quasi-static controls for an input state trajectory.
  4) ShootingProblem.rollout_us() then produces a dynamically feasible state
     rollout from x0 under those controls.

For each forward/backward warm-start, and for two contact-gain cases:
  - zero:     [Kp, Kd] = [0, 0]
  - moderate: [Kp, Kd] = [400, 40]

the script records:
  - max/RMS shooting gap with zero controls;
  - max/RMS shooting gap after quasiStatic controls;
  - max quasi-static control magnitude;
  - CoM tracking and support errors of the feasible rollout.

No generated .npz reference is replaced.
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

LEFT_FOOT_FRAME = "left_foot_contact_fixed"
RIGHT_FOOT_FRAME = "right_foot_contact_fixed"

CONTACT_GAIN_CASES = {
    "zero": np.array([0.0, 0.0], dtype=float),
    "moderate": np.array([400.0, 40.0], dtype=float),
}

RUNNING_STATE_WEIGHT = 0.05
RUNNING_CONTROL_WEIGHT = 1e-3
TERMINAL_STATE_WEIGHT = 5.0
COM_TRACKING_WEIGHT_Y = 2.0e4
COM_TRACKING_WEIGHT_XZ = 2.0e2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = PROJECT_ROOT / "model" / "biped_v1.urdf"
X0_PATH = PROJECT_ROOT / "transfer" / "initialisation" / "fddp_double_support_x0.npz"
TRAJECTORY_DIR = PROJECT_ROOT / "transfer" / "trajectories"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "transfer"
    / "initialisation"
    / "double_support_dynamic_seed_audit.yaml"
)


def evaluate_total_cost(problem, xs, us):
    """Compatibility-safe total cost evaluation."""
    problem.calc(xs, us)
    return (
        sum(float(data.cost) for data in problem.runningDatas)
        + float(problem.terminalData.cost)
    )


def load_x0(path: Path, expected_size: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing initial-state file: {path}")

    with np.load(path) as loaded:
        for key in ("x0", "state", "initial_state"):
            if key in loaded.files:
                x0 = np.asarray(loaded[key], dtype=float).reshape(-1)
                break
        else:
            raise RuntimeError(
                f"Could not find x0 in {path}; keys available: {loaded.files}"
            )

    if x0.size != expected_size:
        raise RuntimeError(f"x0 size {x0.size}; expected {expected_size}.")
    return x0


def load_warmstart(direction: str, expected_x_shape, expected_u_shape):
    path = TRAJECTORY_DIR / f"com_shift_{direction}_warmstart.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing warm-start: {path}\n"
            "Run: python fddp/03f_build_double_support_com_shift_warmstart.py"
        )

    with np.load(path) as loaded:
        required = ("xs_init", "us_init", "com_y_target", "t_ref")
        missing = [key for key in required if key not in loaded.files]
        if missing:
            raise RuntimeError(f"{path} missing keys {missing}; has {loaded.files}")

        xs_init = np.asarray(loaded["xs_init"], dtype=float)
        us_init = np.asarray(loaded["us_init"], dtype=float)
        com_y_target = np.asarray(loaded["com_y_target"], dtype=float).reshape(-1)
        t_ref = np.asarray(loaded["t_ref"], dtype=float).reshape(-1)

    if xs_init.shape != expected_x_shape:
        raise RuntimeError(f"{path}: xs_init {xs_init.shape}; expected {expected_x_shape}")
    if us_init.shape != expected_u_shape:
        raise RuntimeError(f"{path}: us_init {us_init.shape}; expected {expected_u_shape}")
    if com_y_target.shape != (HORIZON_NODES + 1,):
        raise RuntimeError(f"{path}: invalid com_y_target shape {com_y_target.shape}")
    if t_ref.shape != (HORIZON_NODES + 1,):
        raise RuntimeError(f"{path}: invalid t_ref shape {t_ref.shape}")

    return xs_init, us_init, com_y_target, t_ref, path


def support_references(model, x0, left_frame_id, right_frame_id):
    data = model.createData()
    pin.forwardKinematics(model, data, x0[: model.nq])
    pin.updateFramePlacements(model, data)
    return data.oMf[left_frame_id].copy(), data.oMf[right_frame_id].copy()


def make_contact_model(state, nu, frame_id, reference, gains):
    return crocoddyl.ContactModel6D(
        state,
        frame_id,
        reference,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        nu,
        gains,
    )


def make_action_model(
    state,
    actuation,
    x_ref,
    com_target,
    left_frame_id,
    right_frame_id,
    left_reference,
    right_reference,
    gains,
    state_weight,
    control_weight,
    dt,
):
    nu = actuation.nu

    contacts = crocoddyl.ContactModelMultiple(state, nu)
    contacts.addContact(
        LEFT_FOOT_FRAME,
        make_contact_model(state, nu, left_frame_id, left_reference, gains),
    )
    contacts.addContact(
        RIGHT_FOOT_FRAME,
        make_contact_model(state, nu, right_frame_id, right_reference, gains),
    )

    costs = crocoddyl.CostModelSum(state, nu)

    residual_state = crocoddyl.ResidualModelState(state, x_ref, nu)
    costs.addCost(
        "state_regularisation",
        crocoddyl.CostModelResidual(state, residual_state),
        state_weight,
    )

    residual_control = crocoddyl.ResidualModelControl(state, np.zeros(nu))
    costs.addCost(
        "control_regularisation",
        crocoddyl.CostModelResidual(state, residual_control),
        control_weight,
    )

    residual_com = crocoddyl.ResidualModelCoMPosition(state, com_target, nu)
    activation_com = crocoddyl.ActivationModelWeightedQuad(
        np.array(
            [COM_TRACKING_WEIGHT_XZ, COM_TRACKING_WEIGHT_Y, COM_TRACKING_WEIGHT_XZ],
            dtype=float,
        )
    )
    costs.addCost(
        "com_tracking",
        crocoddyl.CostModelResidual(state, activation_com, residual_com),
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


def make_problem(
    model,
    x0,
    com_y_target,
    left_frame_id,
    right_frame_id,
    left_reference,
    right_reference,
    gains,
):
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    initial_com = compute_com(model, x0[None, :])[0]

    running_models = []
    for k in range(HORIZON_NODES):
        target = initial_com.copy()
        target[1] = com_y_target[k]
        running_models.append(
            make_action_model(
                state,
                actuation,
                x0,
                target,
                left_frame_id,
                right_frame_id,
                left_reference,
                right_reference,
                gains,
                RUNNING_STATE_WEIGHT,
                RUNNING_CONTROL_WEIGHT,
                DT,
            )
        )

    terminal_target = initial_com.copy()
    terminal_target[1] = com_y_target[-1]
    terminal_model = make_action_model(
        state,
        actuation,
        x0,
        terminal_target,
        left_frame_id,
        right_frame_id,
        left_reference,
        right_reference,
        gains,
        TERMINAL_STATE_WEIGHT,
        0.0,
        0.0,
    )

    return crocoddyl.ShootingProblem(x0, running_models, terminal_model)


def calculate_gaps(problem, state, xs, us):
    """
    The shooting gap at node k is state.diff(x_{k+1}, f(x_k, u_k)).
    It is separate from the scalar cost.
    """
    problem.calc(xs, us)
    gap_norms = []

    for k, data in enumerate(problem.runningDatas):
        gap = np.asarray(state.diff(xs[k + 1], data.xnext), dtype=float).reshape(-1)
        gap_norms.append(float(np.linalg.norm(gap)))

    values = np.asarray(gap_norms, dtype=float)
    return {
        "max_gap_norm": float(np.max(values)),
        "rms_gap_norm": float(np.sqrt(np.mean(values * values))),
        "first_gap_norm": float(values[0]),
        "last_gap_norm": float(values[-1]),
    }


def rollout_controls(problem, us):
    """Binding-independent forward rollout through Crocoddyl running models."""
    xs = [np.asarray(problem.x0, dtype=float).copy()]
    for k, u in enumerate(us):
        running_model = problem.runningModels[k]
        running_data = problem.runningDatas[k]
        running_model.calc(running_data, xs[-1], u)
        xs.append(np.asarray(running_data.xnext, dtype=float).copy())
    return xs


def compute_com(model, xs):
    data = model.createData()
    output = np.zeros((len(xs), 3), dtype=float)
    for k, x in enumerate(xs):
        output[k] = np.asarray(
            pin.centerOfMass(model, data, np.asarray(x)[: model.nq]),
            dtype=float,
        ).reshape(3)
    return output


def pose_errors(reference, current):
    translation_error = float(np.linalg.norm(current.translation - reference.translation))
    rotation_error = float(
        np.linalg.norm(pin.log3(reference.rotation.T @ current.rotation))
    )
    return translation_error, rotation_error


def support_errors(
    model,
    xs,
    left_frame_id,
    right_frame_id,
    left_reference,
    right_reference,
):
    data = model.createData()
    left_t_values, right_t_values = [], []
    left_r_values, right_r_values = [], []

    for x in xs:
        pin.forwardKinematics(model, data, np.asarray(x)[: model.nq])
        pin.updateFramePlacements(model, data)
        lt, lr = pose_errors(left_reference, data.oMf[left_frame_id])
        rt, rr = pose_errors(right_reference, data.oMf[right_frame_id])
        left_t_values.append(lt)
        right_t_values.append(rt)
        left_r_values.append(lr)
        right_r_values.append(rr)

    return {
        "max_left_translation_error_m": float(max(left_t_values)),
        "max_right_translation_error_m": float(max(right_t_values)),
        "max_left_rotation_error_rad": float(max(left_r_values)),
        "max_right_rotation_error_rad": float(max(right_r_values)),
    }


def evaluate_case(
    direction,
    case_name,
    gains,
    model,
    x0,
    xs_warm,
    com_y_target,
    left_frame_id,
    right_frame_id,
    left_reference,
    right_reference,
):
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    nu = actuation.nu

    problem = make_problem(
        model,
        x0,
        com_y_target,
        left_frame_id,
        right_frame_id,
        left_reference,
        right_reference,
        gains,
    )

    xs_seed = [row.copy() for row in xs_warm]
    us_zero = [np.zeros(nu, dtype=float) for _ in range(HORIZON_NODES)]

    zero_cost = evaluate_total_cost(problem, xs_seed, us_zero)
    zero_gaps = calculate_gaps(problem, state, xs_seed, us_zero)

    try:
        # This Crocoddyl Python binding exposes quasiStatic(xs) and returns the
        # T control vectors directly. Different builds accept either T states
        # (running nodes only) or T+1 states, so try the standard T-state call
        # first and retain the successful mode in the report.
        try:
            returned_controls = problem.quasiStatic(
                [np.asarray(x, dtype=float).copy() for x in xs_seed[:-1]]
            )
            quasi_static_state_list_mode = "running_nodes_only"
        except Exception:
            returned_controls = problem.quasiStatic(
                [np.asarray(x, dtype=float).copy() for x in xs_seed]
            )
            quasi_static_state_list_mode = "full_state_sequence"

        us_quasi = [
            np.asarray(u, dtype=float).reshape(-1).copy()
            for u in returned_controls
        ]
        if len(us_quasi) != HORIZON_NODES:
            raise RuntimeError(
                f"quasiStatic returned {len(us_quasi)} controls; "
                f"expected {HORIZON_NODES}."
            )
        if any(u.shape != (nu,) for u in us_quasi):
            raise RuntimeError(
                "quasiStatic returned a control vector with an unexpected shape."
            )
    except Exception as exc:
        return {
            "contact_gains_Kp_Kd": [float(v) for v in gains],
            "zero_control": {
                "cost": float(zero_cost),
                "shooting_gaps": zero_gaps,
            },
            "quasi_static": {
                "available": False,
                "error": str(exc),
            },
        }
    quasi_cost = evaluate_total_cost(problem, xs_seed, us_quasi)
    quasi_gaps = calculate_gaps(problem, state, xs_seed, us_quasi)
    max_abs_u = float(max(np.max(np.abs(u)) for u in us_quasi))
    rms_u = float(
        np.sqrt(np.mean(np.concatenate([u.reshape(-1) for u in us_quasi]) ** 2))
    )

    rollout_xs = rollout_controls(problem, us_quasi)
    if len(rollout_xs) != HORIZON_NODES + 1:
        raise RuntimeError(
            f"{direction}/{case_name}: rollout returned {len(rollout_xs)} states; "
            f"expected {HORIZON_NODES + 1}."
        )

    rollout_cost = evaluate_total_cost(problem, rollout_xs, us_quasi)
    rollout_gaps = calculate_gaps(problem, state, rollout_xs, us_quasi)
    rollout_com = compute_com(model, rollout_xs)
    rollout_supports = support_errors(
        model,
        rollout_xs,
        left_frame_id,
        right_frame_id,
        left_reference,
        right_reference,
    )

    initial_com_y = float(rollout_com[0, 1])
    if direction == "forward":
        achieved_shift = float(np.max(rollout_com[:, 1]) - initial_com_y)
    else:
        achieved_shift = float(np.min(rollout_com[:, 1]) - initial_com_y)

    rollout_tracking_error = float(
        np.max(np.abs(rollout_com[:, 1] - com_y_target))
    )
    rollout_return_error = float(abs(rollout_com[-1, 1] - initial_com_y))

    return {
        "contact_gains_Kp_Kd": [float(v) for v in gains],
        "zero_control": {
            "cost": float(zero_cost),
            "shooting_gaps": zero_gaps,
        },
        "quasi_static": {
            "available": True,
            "cost_on_kinematic_warmstart": float(quasi_cost),
            "shooting_gaps_on_kinematic_warmstart": quasi_gaps,
            "max_abs_control": max_abs_u,
            "rms_control": rms_u,
        },
        "rollout_from_quasi_static_controls": {
            "cost": float(rollout_cost),
            "shooting_gaps": rollout_gaps,
            "achieved_peak_shift_y_m": achieved_shift,
            "peak_abs_com_y_tracking_error_m": rollout_tracking_error,
            "final_com_y_return_error_m": rollout_return_error,
            **rollout_supports,
        },
    }


def main():
    print("===== STAGE 7C DIAGNOSTIC-3: FDDP DYNAMIC INITIALISATION AUDIT =====")
    print("This script does not call SolverFDDP and does not overwrite trajectories.")
    print(f"URDF: {URDF_PATH}")
    print("Cases: zero controls vs quasiStatic controls; zero vs moderate contact gains.")

    if not URDF_PATH.exists():
        raise FileNotFoundError(f"URDF not found: {URDF_PATH}")

    model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
    if model.nq != 13 or model.nv != 12:
        raise RuntimeError(
            f"Unexpected model dimensions nq={model.nq}, nv={model.nv}; "
            "expected nq=13, nv=12."
        )

    x0 = load_x0(X0_PATH, model.nq + model.nv)
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)

    left_frame_id = model.getFrameId(LEFT_FOOT_FRAME)
    right_frame_id = model.getFrameId(RIGHT_FOOT_FRAME)
    if left_frame_id >= model.nframes or right_frame_id >= model.nframes:
        raise RuntimeError(
            f"Invalid support frames: left={left_frame_id}, right={right_frame_id}"
        )

    left_reference, right_reference = support_references(
        model, x0, left_frame_id, right_frame_id
    )

    expected_x_shape = (HORIZON_NODES + 1, model.nq + model.nv)
    expected_u_shape = (HORIZON_NODES, actuation.nu)

    output = {
        "stage": "7C_dynamic_initialisation_audit",
        "purpose": (
            "Compare the kinematic warm-start with zero controls against "
            "quasi-static controls and a dynamically rolled-out seed."
        ),
        "model": {
            "urdf": str(URDF_PATH),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(actuation.nu),
        },
        "discretisation": {
            "dt_s": float(DT),
            "horizon_nodes": int(HORIZON_NODES),
        },
        "directions": {},
    }

    for direction in ("forward", "backward"):
        xs_warm, us_warm, com_y_target, _, warm_path = load_warmstart(
            direction, expected_x_shape, expected_u_shape
        )

        if not np.allclose(us_warm, 0.0):
            print(f"Warning: {direction} warm-start us_init is not exactly zero.")

        print()
        print(f"----- {direction.upper()} -----")
        print(f"Warm-start source: {warm_path}")

        direction_summary = {
            "warmstart_source": str(warm_path),
            "warmstart_us_is_zero": bool(np.allclose(us_warm, 0.0)),
            "gain_cases": {},
        }

        for case_name, gains in CONTACT_GAIN_CASES.items():
            result = evaluate_case(
                direction,
                case_name,
                gains,
                model,
                x0,
                xs_warm,
                com_y_target,
                left_frame_id,
                right_frame_id,
                left_reference,
                right_reference,
            )
            direction_summary["gain_cases"][case_name] = result

            print(f"  [{case_name}] gains={gains.tolist()}")
            print(
                "    zero-u gaps max/RMS: "
                f"{result['zero_control']['shooting_gaps']['max_gap_norm']:.3e} / "
                f"{result['zero_control']['shooting_gaps']['rms_gap_norm']:.3e}"
            )

            quasi = result["quasi_static"]
            if not quasi["available"]:
                print(f"    quasiStatic: FAILED: {quasi['error']}")
                continue

            print(
                "    quasiStatic u max/RMS: "
                f"{quasi['max_abs_control']:.4f} / {quasi['rms_control']:.4f}"
            )
            print(
                "    quasi-u gaps max/RMS: "
                f"{quasi['shooting_gaps_on_kinematic_warmstart']['max_gap_norm']:.3e} / "
                f"{quasi['shooting_gaps_on_kinematic_warmstart']['rms_gap_norm']:.3e}"
            )

            rollout = result["rollout_from_quasi_static_controls"]
            print(
                "    rollout gaps max/RMS: "
                f"{rollout['shooting_gaps']['max_gap_norm']:.3e} / "
                f"{rollout['shooting_gaps']['rms_gap_norm']:.3e}"
            )
            print(
                "    rollout CoM shift / tracking / return (mm): "
                f"{rollout['achieved_peak_shift_y_m'] * 1000.0:+.3f} / "
                f"{rollout['peak_abs_com_y_tracking_error_m'] * 1000.0:.3f} / "
                f"{rollout['final_com_y_return_error_m'] * 1000.0:.3f}"
            )
            print(
                "    rollout max L/R foot translation error (mm): "
                f"{rollout['max_left_translation_error_m'] * 1000.0:.4f} / "
                f"{rollout['max_right_translation_error_m'] * 1000.0:.4f}"
            )

        output["directions"][direction] = direction_summary

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(yaml.safe_dump(output, sort_keys=False), encoding="utf-8")

    print()
    print("Interpretation rule:")
    print("  - Dynamic rollout gaps should be near numerical precision.")
    print("  - A low kinematic-seed cost with large zero-u gaps confirms that cost alone")
    print("    is not a feasibility test.")
    print("  - The next FDDP script will use the gain case whose quasi-static rollout")
    print("    remains best behaved, with a dynamics-feasible rollout as its initial xs.")
    print()
    print(f"Saved:\n  {OUTPUT_PATH}")
    print("DYNAMIC INITIALISATION AUDIT PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nDYNAMIC INITIALISATION AUDIT FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
