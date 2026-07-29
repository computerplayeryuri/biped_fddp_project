#!/usr/bin/env python3
"""
STAGE 7C DIAGNOSTIC-6 — Bilateral 6D-contact dynamics derivative audit.

Purpose
-------
The previous damping audit showed that the 6D flat-foot constrained forward
dynamics are statically stable even with JMinvJt_damping=0. Therefore damping
alone does not explain the non-static FDDP divergence.

FDDP depends on calcDiff(), not only calc(). This script checks whether the
analytic discrete-time derivatives supplied by Crocoddyl's integrated
two-foot 6D contact model agree with finite differences around x0.

For every damping case it evaluates:
  - calc() and calcDiff() success / finiteness;
  - analytic Fx and Fu;
  - finite-difference Fx and Fu on the Pinocchio manifold;
  - absolute and relative errors;
  - rank and singular values of Fu.

It does NOT run SolverFDDP and does NOT overwrite trajectories.
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import yaml
import pinocchio as pin
import crocoddyl


DT = 0.010
FD_EPS_STATE = 1.0e-7
FD_EPS_CONTROL = 1.0e-7

LEFT_FOOT_FRAME = "left_foot_contact_fixed"
RIGHT_FOOT_FRAME = "right_foot_contact_fixed"
DAMPING_CASES = [0.0, 1e-10, 1e-8, 1e-6]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = PROJECT_ROOT / "model" / "biped_v1.urdf"
X0_PATH = PROJECT_ROOT / "transfer" / "initialisation" / "fddp_double_support_x0.npz"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "transfer"
    / "initialisation"
    / "six_d_contact_derivative_audit.yaml"
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
            raise RuntimeError(f"No x0-like key in {path}; keys: {loaded.files}")
    if x0.size != expected_size:
        raise RuntimeError(f"x0 size {x0.size}; expected {expected_size}")
    return x0


def get_support_placements(model, x0, left_frame_id, right_frame_id):
    data = model.createData()
    pin.forwardKinematics(model, data, x0[: model.nq])
    pin.updateFramePlacements(model, data)
    return data.oMf[left_frame_id].copy(), data.oMf[right_frame_id].copy()


def build_action(model, x0, left_frame_id, right_frame_id, left_ref, right_ref, damping):
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    nu = actuation.nu

    contacts = crocoddyl.ContactModelMultiple(state, nu)
    zero_gains = np.zeros(2, dtype=float)

    contacts.addContact(
        LEFT_FOOT_FRAME,
        crocoddyl.ContactModel6D(
            state,
            left_frame_id,
            left_ref,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            nu,
            zero_gains,
        ),
    )
    contacts.addContact(
        RIGHT_FOOT_FRAME,
        crocoddyl.ContactModel6D(
            state,
            right_frame_id,
            right_ref,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            nu,
            zero_gains,
        ),
    )

    costs = crocoddyl.CostModelSum(state, nu)
    differential = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state,
        actuation,
        contacts,
        costs,
        float(damping),
        False,
    )
    return crocoddyl.IntegratedActionModelEuler(differential, DT), state, nu


def matrix_summary(matrix):
    matrix = np.asarray(matrix, dtype=float)
    finite = bool(np.all(np.isfinite(matrix)))
    if not finite:
        return {
            "shape": [int(v) for v in matrix.shape],
            "finite": False,
        }

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    tolerance = (
        max(matrix.shape)
        * np.finfo(float).eps
        * float(singular_values[0])
        if singular_values.size
        else 0.0
    )
    rank = int(np.sum(singular_values > tolerance))
    return {
        "shape": [int(v) for v in matrix.shape],
        "finite": True,
        "frobenius_norm": float(np.linalg.norm(matrix)),
        "rank": rank,
        "svd_tolerance": float(tolerance),
        "singular_values": [float(v) for v in singular_values],
    }


def relative_error(analytic, numerical):
    analytic = np.asarray(analytic, dtype=float)
    numerical = np.asarray(numerical, dtype=float)
    absolute = float(np.linalg.norm(analytic - numerical))
    scale = max(1.0, float(np.linalg.norm(analytic)), float(np.linalg.norm(numerical)))
    return {
        "absolute_frobenius_error": absolute,
        "relative_frobenius_error": float(absolute / scale),
        "max_abs_element_error": float(np.max(np.abs(analytic - numerical))),
    }


def evaluate_action(action, state, x0, u0):
    data = action.createData()
    action.calc(data, x0, u0)
    xnext = np.asarray(data.xnext, dtype=float).copy()
    action.calcDiff(data, x0, u0)
    fx = np.asarray(data.Fx, dtype=float).copy()
    fu = np.asarray(data.Fu, dtype=float).copy()
    return xnext, fx, fu


def finite_difference_derivatives(action, state, x0, u0, xnext0):
    ndx = state.ndx
    nu = u0.size
    fx_fd = np.zeros((ndx, ndx), dtype=float)
    fu_fd = np.zeros((ndx, nu), dtype=float)

    for i in range(ndx):
        dx = np.zeros(ndx, dtype=float)
        dx[i] = FD_EPS_STATE
        x_perturbed = np.asarray(state.integrate(x0, dx), dtype=float)
        data = action.createData()
        action.calc(data, x_perturbed, u0)
        xnext_perturbed = np.asarray(data.xnext, dtype=float)
        fx_fd[:, i] = np.asarray(
            state.diff(xnext0, xnext_perturbed), dtype=float
        ).reshape(-1) / FD_EPS_STATE

    for j in range(nu):
        u_perturbed = u0.copy()
        u_perturbed[j] += FD_EPS_CONTROL
        data = action.createData()
        action.calc(data, x0, u_perturbed)
        xnext_perturbed = np.asarray(data.xnext, dtype=float)
        fu_fd[:, j] = np.asarray(
            state.diff(xnext0, xnext_perturbed), dtype=float
        ).reshape(-1) / FD_EPS_CONTROL

    return fx_fd, fu_fd


def run_case(damping, model, x0, left_frame_id, right_frame_id, left_ref, right_ref):
    result = {"damping": float(damping)}
    try:
        action, state, nu = build_action(
            model, x0, left_frame_id, right_frame_id, left_ref, right_ref, damping
        )
        u0 = np.zeros(nu, dtype=float)

        xnext, fx, fu = evaluate_action(action, state, x0, u0)
        result["calc_and_calcDiff_completed"] = True
        result["xnext_finite"] = bool(np.all(np.isfinite(xnext)))
        result["Fx_analytic"] = matrix_summary(fx)
        result["Fu_analytic"] = matrix_summary(fu)

        fx_fd, fu_fd = finite_difference_derivatives(action, state, x0, u0, xnext)
        result["Fx_finite_difference"] = matrix_summary(fx_fd)
        result["Fu_finite_difference"] = matrix_summary(fu_fd)

        if (
            result["Fx_analytic"]["finite"]
            and result["Fu_analytic"]["finite"]
            and result["Fx_finite_difference"]["finite"]
            and result["Fu_finite_difference"]["finite"]
        ):
            result["Fx_error"] = relative_error(fx, fx_fd)
            result["Fu_error"] = relative_error(fu, fu_fd)
            result["derivatives_numerically_consistent"] = bool(
                result["Fx_error"]["relative_frobenius_error"] < 1e-3
                and result["Fu_error"]["relative_frobenius_error"] < 1e-3
            )
        else:
            result["derivatives_numerically_consistent"] = False

    except Exception as exc:
        result["calc_and_calcDiff_completed"] = False
        result["exception"] = str(exc)
        result["derivatives_numerically_consistent"] = False

    return result


def fmt_error(result, key):
    if key not in result:
        return "n/a"
    return f"{result[key]['relative_frobenius_error']:.3e}"


def main():
    print("===== STAGE 7C DIAGNOSTIC-6: 6D-CONTACT DERIVATIVE AUDIT =====")
    print("This script does not run FDDP and does not overwrite trajectories.")
    print(f"URDF: {URDF_PATH}")
    print(f"Finite-difference eps (state/control): {FD_EPS_STATE:.0e}/{FD_EPS_CONTROL:.0e}")

    model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
    if model.nq != 13 or model.nv != 12:
        raise RuntimeError(f"Unexpected model nq={model.nq}, nv={model.nv}")

    x0 = load_x0(X0_PATH, model.nq + model.nv)
    left_frame_id = model.getFrameId(LEFT_FOOT_FRAME)
    right_frame_id = model.getFrameId(RIGHT_FOOT_FRAME)
    left_ref, right_ref = get_support_placements(
        model, x0, left_frame_id, right_frame_id
    )

    results = []
    for damping in DAMPING_CASES:
        result = run_case(
            damping,
            model,
            x0,
            left_frame_id,
            right_frame_id,
            left_ref,
            right_ref,
        )
        results.append(result)

        print(f"\n----- damping={damping:.0e} -----")
        if not result["calc_and_calcDiff_completed"]:
            print(f"calc/calcDiff FAILED: {result['exception']}")
            continue

        print(f"xnext finite: {result['xnext_finite']}")
        print(
            "Fx relative error (analytic vs FD): "
            f"{fmt_error(result, 'Fx_error')}"
        )
        print(
            "Fu relative error (analytic vs FD): "
            f"{fmt_error(result, 'Fu_error')}"
        )
        print(
            "Fu analytic / FD rank: "
            f"{result['Fu_analytic'].get('rank')} / "
            f"{result['Fu_finite_difference'].get('rank')}"
        )
        print(
            "Derivative consistency (<1e-3 both): "
            f"{result['derivatives_numerically_consistent']}"
        )

    consistent = [
        r for r in results if r.get("derivatives_numerically_consistent", False)
    ]
    recommended = (
        min(consistent, key=lambda r: r["damping"])["damping"]
        if consistent
        else None
    )

    if recommended is None:
        interpretation = (
            "No damping case produced analytic derivatives consistent with "
            "finite differences at the rank-deficient bilateral 6D contact "
            "state. Do not continue FDDP with this redundant contact stack."
        )
    else:
        interpretation = (
            "At least one damping case produced locally consistent derivatives. "
            "The remaining FDDP issue is then trajectory initialisation/cost "
            "continuation rather than a broken local contact derivative."
        )

    summary = {
        "stage": "7C_6d_contact_derivative_audit",
        "purpose": (
            "Check analytic calcDiff derivatives of bilateral 6D contact "
            "dynamics against finite differences before attempting another FDDP solve."
        ),
        "model": {
            "urdf": str(URDF_PATH),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": 6,
        },
        "finite_difference": {
            "state_epsilon": float(FD_EPS_STATE),
            "control_epsilon": float(FD_EPS_CONTROL),
        },
        "results": results,
        "recommended_damping_from_derivative_consistency": (
            float(recommended) if recommended is not None else None
        ),
        "interpretation": interpretation,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")

    print("\nInterpretation:")
    print(f"  {interpretation}")
    print(
        "  recommended_damping_from_derivative_consistency = "
        f"{recommended}"
    )
    print(f"\nSaved:\n  {OUTPUT_PATH}")
    print("6D-CONTACT DERIVATIVE AUDIT COMPLETED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n6D-CONTACT DERIVATIVE AUDIT FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
