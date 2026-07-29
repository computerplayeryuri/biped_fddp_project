#!/usr/bin/env python3
"""
STAGE 7C-VALIDATION — Damped 6D-contact FDDP micro-shift.

Purpose
-------
Validate the corrected non-static FDDP formulation before attempting a full
±6 mm CoM transfer.

The derivative audit demonstrated:
  - JMinvJt_damping = 0.0 gives an inaccurate Fx derivative;
  - JMinvJt_damping = 1e-8 gives accurate Fx and Fu derivatives.

This script therefore uses:
  * two 6D flat-foot contacts referenced to the measured x0 world poses;
  * JMinvJt_damping = 1e-8;
  * zero Baumgarte gains (the seed is already contact-consistent);
  * a +0.5 mm CoM_y shift-and-return task;
  * a scaled version of the validated kinematic warm-start as BOTH
    the FDDP state seed and the state-tracking cost reference.

It does not overwrite com_shift_forward_ref.npz or any full-scale reference.
It writes a diagnostic summary only:
  transfer/initialisation/damped_6d_microshift_validation.yaml
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import yaml
import pinocchio as pin
import crocoddyl


# ------------------------------- Configuration -------------------------------
DT = 0.010
HORIZON_NODES = 120
MAX_ITER = 250

# Chosen from the derivative audit: much more accurate than 1e-10 while being
# much smaller than 1e-6, so the constrained dynamics are minimally perturbed.
JM_INV_JT_DAMPING = 1.0e-8

# A small, controlled first non-static target.
MICRO_SHIFT_M = 0.0005  # +0.5 mm
WARMSTART_FULL_SHIFT_M = 0.006  # Stage 03f warm-start amplitude

LEFT_FOOT_FRAME = "left_foot_contact_fixed"
RIGHT_FOOT_FRAME = "right_foot_contact_fixed"

# Contact references are exact x0 placements, so do not inject a second
# proportional contact stabilisation effect here.
CONTACT_GAINS_KP_KD = np.array([0.0, 0.0], dtype=float)

# Costs. State tracking follows the scaled kinematic warm-start rather than
# pulling every node back to x0. CoM tracking remains the primary task term.
RUNNING_STATE_TRACKING_WEIGHT = 2.0
RUNNING_CONTROL_WEIGHT = 1.0e-3
TERMINAL_STATE_TRACKING_WEIGHT = 10.0
COM_TRACKING_WEIGHT_Y = 5.0e4
COM_TRACKING_WEIGHT_XZ = 2.0e2

# Integrity gates.
MAX_SUPPORT_TRANSLATION_ERROR_M = 5.0e-4     # 0.5 mm
MAX_SUPPORT_ROTATION_ERROR_RAD = np.deg2rad(0.5)
MIN_JOINT_LIMIT_MARGIN_RAD = 1.0e-2          # 0.57 deg
MIN_ACHIEVED_SHIFT_M = 2.0e-4                # 0.2 mm
MAX_COM_Y_TRACKING_ERROR_M = 5.0e-4          # 0.5 mm
MAX_FINAL_COM_Y_RETURN_ERROR_M = 2.0e-4      # 0.2 mm
MAX_FINAL_GAP_NORM = 1.0e-7

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = PROJECT_ROOT / "model" / "biped_v1.urdf"
X0_PATH = PROJECT_ROOT / "transfer" / "initialisation" / "fddp_double_support_x0.npz"
WARMSTART_PATH = PROJECT_ROOT / "transfer" / "trajectories" / "com_shift_forward_warmstart.npz"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "transfer"
    / "initialisation"
    / "damped_6d_microshift_validation.yaml"
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
                f"No x0-like key in {path}; available keys: {loaded.files}"
            )
    if x0.size != expected_size:
        raise RuntimeError(f"x0 size {x0.size}; expected {expected_size}.")
    return x0


def load_full_warmstart(path: Path, expected_x_shape: tuple[int, int]):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing warm-start: {path}\n"
            "Run: python fddp/03f_build_double_support_com_shift_warmstart.py"
        )
    with np.load(path) as loaded:
        required = ("xs_init", "com_y_target", "t_ref")
        missing = [name for name in required if name not in loaded.files]
        if missing:
            raise RuntimeError(f"{path} is missing {missing}; keys: {loaded.files}")
        xs = np.asarray(loaded["xs_init"], dtype=float)
        com_y_target = np.asarray(loaded["com_y_target"], dtype=float).reshape(-1)
        t_ref = np.asarray(loaded["t_ref"], dtype=float).reshape(-1)

    if xs.shape != expected_x_shape:
        raise RuntimeError(f"warm-start xs shape {xs.shape}; expected {expected_x_shape}.")
    if com_y_target.shape != (HORIZON_NODES + 1,):
        raise RuntimeError("unexpected warm-start com_y_target shape")
    if t_ref.shape != (HORIZON_NODES + 1,):
        raise RuntimeError("unexpected warm-start t_ref shape")
    return xs, com_y_target, t_ref


def support_references(model, q0, left_frame_id, right_frame_id):
    data = model.createData()
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)
    return data.oMf[left_frame_id].copy(), data.oMf[right_frame_id].copy()


def make_scaled_warmstart(model, x0, xs_full, ratio):
    """
    Scale q on the Pinocchio configuration manifold, rather than scaling raw
    quaternion components. Scale tangent velocity linearly.

    x = [q(13), v(12)] and the full warm-start has the same layout.
    """
    q0 = x0[: model.nq]
    xs_scaled = np.zeros_like(xs_full)

    for k, x_full in enumerate(xs_full):
        q_full = x_full[: model.nq]
        v_full = x_full[model.nq :]

        dq_tangent = np.asarray(
            pin.difference(model, q0, q_full), dtype=float
        ).reshape(-1)
        q_scaled = np.asarray(
            pin.integrate(model, q0, ratio * dq_tangent), dtype=float
        ).reshape(-1)
        v_scaled = ratio * v_full

        xs_scaled[k, : model.nq] = q_scaled
        xs_scaled[k, model.nq :] = v_scaled

    # Preserve exact initial x0.
    xs_scaled[0] = x0.copy()
    return xs_scaled


def compute_com(model, xs):
    data = model.createData()
    values = np.zeros((xs.shape[0], 3), dtype=float)
    for k, x in enumerate(xs):
        values[k] = np.asarray(
            pin.centerOfMass(model, data, x[: model.nq]), dtype=float
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

    residual_state = crocoddyl.ResidualModelState(state, x_reference, nu)
    costs.addCost(
        "warmstart_state_tracking",
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
        crocoddyl.CostModelResidual(state, activation_com, residual_com),
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
        com_target = initial_com.copy()
        com_target[1] = com_y_target[k]
        running_models.append(
            build_action(
                state,
                actuation,
                xs_reference[k],
                com_target,
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


def gap_summary(problem, state, xs, us):
    problem.calc(xs, us)
    gaps = []
    for k, data in enumerate(problem.runningDatas):
        gap = np.asarray(
            state.diff(xs[k + 1], data.xnext), dtype=float
        ).reshape(-1)
        gaps.append(float(np.linalg.norm(gap)))
    gaps = np.asarray(gaps, dtype=float)
    return {
        "max_gap_norm": float(np.max(gaps)),
        "rms_gap_norm": float(np.sqrt(np.mean(gaps**2))),
        "first_gap_norm": float(gaps[0]),
        "last_gap_norm": float(gaps[-1]),
    }


def support_error_summary(model, xs, left_frame_id, right_frame_id, left_ref, right_ref):
    data = model.createData()
    left_t_max = right_t_max = 0.0
    left_r_max = right_r_max = 0.0

    for x in xs:
        pin.forwardKinematics(model, data, x[: model.nq])
        pin.updateFramePlacements(model, data)

        left_t_max = max(
            left_t_max,
            float(np.linalg.norm(data.oMf[left_frame_id].translation - left_ref.translation)),
        )
        right_t_max = max(
            right_t_max,
            float(np.linalg.norm(data.oMf[right_frame_id].translation - right_ref.translation)),
        )
        left_r_max = max(
            left_r_max,
            float(np.linalg.norm(pin.log3(left_ref.rotation.T @ data.oMf[left_frame_id].rotation))),
        )
        right_r_max = max(
            right_r_max,
            float(np.linalg.norm(pin.log3(right_ref.rotation.T @ data.oMf[right_frame_id].rotation))),
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


def main():
    print("===== STAGE 7C VALIDATION: DAMPED 6D-CONTACT FDDP MICRO-SHIFT =====")
    print("This is a +0.5 mm non-static validation only; no reference file is exported.")
    print(f"URDF: {URDF_PATH}")
    print(f"JMinvJt damping: {JM_INV_JT_DAMPING:.0e}")
    print(f"Contact gains [Kp, Kd]: {CONTACT_GAINS_KP_KD.tolist()}")
    print(f"Target peak CoM_y shift: {MICRO_SHIFT_M * 1000.0:+.3f} mm")

    model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
    if model.nq != 13 or model.nv != 12:
        raise RuntimeError(f"Unexpected model nq={model.nq}, nv={model.nv}")

    x0 = load_x0(X0_PATH, model.nq + model.nv)
    state_dim = model.nq + model.nv
    full_warm_xs, full_warm_target, t_ref = load_full_warmstart(
        WARMSTART_PATH,
        (HORIZON_NODES + 1, state_dim),
    )

    ratio = MICRO_SHIFT_M / WARMSTART_FULL_SHIFT_M
    xs_seed_array = make_scaled_warmstart(model, x0, full_warm_xs, ratio)

    initial_y = float(full_warm_target[0])
    micro_target = initial_y + ratio * (full_warm_target - initial_y)

    left_frame_id = model.getFrameId(LEFT_FOOT_FRAME)
    right_frame_id = model.getFrameId(RIGHT_FOOT_FRAME)
    left_ref, right_ref = support_references(
        model, x0[:model.nq], left_frame_id, right_frame_id
    )

    problem, state, nu = build_problem(
        model,
        x0,
        xs_seed_array,
        micro_target,
        left_frame_id,
        right_frame_id,
        left_ref,
        right_ref,
    )
    xs_seed = [x.copy() for x in xs_seed_array]
    us_seed = [np.zeros(nu, dtype=float) for _ in range(HORIZON_NODES)]

    initial_cost = total_cost(problem, xs_seed, us_seed)
    initial_gaps = gap_summary(problem, state, xs_seed, us_seed)

    solver = crocoddyl.SolverFDDP(problem)
    solver.th_stop = 1e-7
    returned_success = bool(solver.solve(xs_seed, us_seed, MAX_ITER, False))
    stop_passed = bool(float(solver.stop) <= float(solver.th_stop))
    accepted_convergence = bool(returned_success or stop_passed)

    xs = np.asarray(solver.xs, dtype=float)
    us = np.asarray(solver.us, dtype=float)

    if xs.shape != (HORIZON_NODES + 1, state_dim):
        raise RuntimeError(f"Unexpected xs output shape: {xs.shape}")
    if us.shape != (HORIZON_NODES, nu):
        raise RuntimeError(f"Unexpected us output shape: {us.shape}")

    final_cost = total_cost(
        problem,
        [x.copy() for x in xs],
        [u.copy() for u in us],
    )
    final_gaps = gap_summary(
        problem,
        state,
        [x.copy() for x in xs],
        [u.copy() for u in us],
    )
    com = compute_com(model, xs)

    initial_com_y = float(com[0, 1])
    achieved_shift = float(np.max(com[:, 1]) - initial_com_y)
    tracking_error = float(np.max(np.abs(com[:, 1] - micro_target)))
    return_error = float(abs(com[-1, 1] - initial_com_y))

    supports = support_error_summary(
        model,
        xs,
        left_frame_id,
        right_frame_id,
        left_ref,
        right_ref,
    )
    limits = joint_limit_summary(model, xs)

    print()
    print("----- DAMPED 6D FDDP MICRO-SHIFT RESULT -----")
    print(f"Solver return value: {returned_success}")
    print(f"Solver stop criterion: {solver.stop:.3e}")
    print(f"Stopping criterion passed: {stop_passed}")
    print(f"Solver accepted: {accepted_convergence}")
    print(f"Initial / final cost: {initial_cost:.10e} / {final_cost:.10e}")
    print(
        "Initial gaps max/RMS: "
        f"{initial_gaps['max_gap_norm']:.3e} / {initial_gaps['rms_gap_norm']:.3e}"
    )
    print(
        "Final gaps max/RMS: "
        f"{final_gaps['max_gap_norm']:.3e} / {final_gaps['rms_gap_norm']:.3e}"
    )
    print(
        "Requested / achieved CoM_y shift: "
        f"{MICRO_SHIFT_M * 1000.0:+.4f} / {achieved_shift * 1000.0:+.4f} mm"
    )
    print(f"Peak |CoM_y tracking error|: {tracking_error * 1000.0:.5f} mm")
    print(f"Final CoM_y return error: {return_error * 1000.0:.5f} mm")
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
    if not accepted_convergence:
        failures.append("SolverFDDP did not satisfy convergence criterion")
    if not np.isfinite(final_cost):
        failures.append("final cost is non-finite")
    if final_gaps["max_gap_norm"] > MAX_FINAL_GAP_NORM:
        failures.append(
            f"final max shooting gap {final_gaps['max_gap_norm']:.3e} exceeds "
            f"{MAX_FINAL_GAP_NORM:.1e}"
        )
    if achieved_shift < MIN_ACHIEVED_SHIFT_M:
        failures.append("insufficient forward CoM shift")
    if tracking_error > MAX_COM_Y_TRACKING_ERROR_M:
        failures.append("CoM tracking error above threshold")
    if return_error > MAX_FINAL_COM_Y_RETURN_ERROR_M:
        failures.append("final CoM return error above threshold")
    if supports["max_left_translation_error_m"] > MAX_SUPPORT_TRANSLATION_ERROR_M:
        failures.append("left foot translation error above threshold")
    if supports["max_right_translation_error_m"] > MAX_SUPPORT_TRANSLATION_ERROR_M:
        failures.append("right foot translation error above threshold")
    if supports["max_left_rotation_error_rad"] > MAX_SUPPORT_ROTATION_ERROR_RAD:
        failures.append("left foot rotation error above threshold")
    if supports["max_right_rotation_error_rad"] > MAX_SUPPORT_ROTATION_ERROR_RAD:
        failures.append("right foot rotation error above threshold")
    if not limits["within_position_limits"]:
        failures.append("joint position-limit violation")
    elif not limits["has_safety_margin"]:
        failures.append("joint-limit safety margin below threshold")

    summary = {
        "stage": "7C_damped_6d_contact_fddp_microshift_validation",
        "purpose": (
            "Validate the corrected damped 6D-contact FDDP formulation on a "
            "+0.5 mm CoM_y shift before any full ±6 mm continuation run."
        ),
        "model": {
            "urdf": str(URDF_PATH),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "ndx": int(state.ndx),
            "nu": int(nu),
        },
        "contact_dynamics": {
            "type": "two 6D flat-foot contacts",
            "JMinvJt_damping": float(JM_INV_JT_DAMPING),
            "baumgarte_gains_Kp_Kd": [float(v) for v in CONTACT_GAINS_KP_KD],
            "reference_mode": "exact world foot poses at x0",
        },
        "task": {
            "direction": "forward",
            "target_peak_shift_y_m": float(MICRO_SHIFT_M),
            "warmstart_source": str(WARMSTART_PATH),
            "warmstart_scale_ratio": float(ratio),
        },
        "solver": {
            "name": "SolverFDDP",
            "max_iterations": int(MAX_ITER),
            "return_success": bool(returned_success),
            "stop_criterion": float(solver.stop),
            "threshold": float(solver.th_stop),
            "accepted_convergence": bool(accepted_convergence),
            "initial_cost": float(initial_cost),
            "final_cost": float(final_cost),
            "initial_shooting_gaps": initial_gaps,
            "final_shooting_gaps": final_gaps,
        },
        "result": {
            "achieved_peak_shift_y_m": float(achieved_shift),
            "peak_abs_com_y_tracking_error_m": float(tracking_error),
            "final_com_y_return_error_m": float(return_error),
            **supports,
            "joint_limits": limits,
        },
        "integrity_passed": bool(not failures),
        "failures": failures,
        "next_step": (
            "Use the identical damped 6D contact model and warmstart-tracking "
            "cost formulation in a full amplitude continuation."
            if not failures
            else
            "Do not scale to 6 mm. Inspect this diagnostic result first."
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")

    if failures:
        print("Micro-shift integrity: FAILED")
        for item in failures:
            print(f"  - {item}")
    else:
        print("Micro-shift integrity: PASSED")
    print(f"Saved:\n  {OUTPUT_PATH}")
    print("DAMPED 6D MICRO-SHIFT VALIDATION COMPLETED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nDAMPED 6D MICRO-SHIFT VALIDATION FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
