#!/usr/bin/env python3
"""
STAGE 7C DIAGNOSTIC-5 — Contact KKT damping audit.

Why this diagnostic is required
-------------------------------
The current biped has redundant/rank-deficient bilateral contact Jacobians:
  - two 6D contacts: rank 8 from 12 rows;
  - two 3D contacts: rank 5 from 6 rows.

All earlier FDDP/contact-dynamics scripts passed JMinvJt_damping=0.0 to
DifferentialActionModelContactFwdDynamics. Crocoddyl's own source explicitly
requires a non-zero damping term for a rank-deficient contact Jacobian.

This script is read-only:
  - does NOT run SolverFDDP;
  - does NOT overwrite any trajectory;
  - tests short zero-input constrained-dynamics rollouts only.

It compares 6D flat-foot contacts and 3D translational contacts across a
small damping sweep, then writes a YAML report. The next FDDP script will use
the best finite *6D flat-foot* damping case, because the physical robot has
finite-area feet and flat-foot support is the intended model.
"""

from __future__ import annotations

from pathlib import Path
import sys
import math
import numpy as np
import yaml
import pinocchio as pin
import crocoddyl


DT = 0.010
ROLLOUT_STEPS = 40  # 0.40 s static-hold audit; no trajectory optimisation

LEFT_FOOT_FRAME = "left_foot_contact_fixed"
RIGHT_FOOT_FRAME = "right_foot_contact_fixed"

# Zero is deliberately retained as a control case. The remaining values are
# operational-space-inertia regularisers for rank-deficient contact KKT solves.
DAMPING_CASES = [0.0, 1e-10, 1e-8, 1e-6, 1e-4]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = PROJECT_ROOT / "model" / "biped_v1.urdf"
X0_PATH = PROJECT_ROOT / "transfer" / "initialisation" / "fddp_double_support_x0.npz"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "transfer"
    / "initialisation"
    / "contact_kkt_damping_audit.yaml"
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
                f"No x0-like key found in {path}; available: {loaded.files}"
            )

    if x0.size != expected_size:
        raise RuntimeError(f"x0 has {x0.size} values; expected {expected_size}.")
    return x0


def numeric_rank(matrix: np.ndarray):
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    tolerance = (
        max(matrix.shape)
        * np.finfo(float).eps
        * float(singular_values[0])
    )
    rank = int(np.sum(singular_values > tolerance))
    return rank, float(tolerance), singular_values


def get_support_references(model, x0, left_frame_id, right_frame_id):
    data = model.createData()
    pin.forwardKinematics(model, data, x0[:model.nq])
    pin.updateFramePlacements(model, data)
    return {
        "left_translation": data.oMf[left_frame_id].translation.copy(),
        "right_translation": data.oMf[right_frame_id].translation.copy(),
        "left_placement": data.oMf[left_frame_id].copy(),
        "right_placement": data.oMf[right_frame_id].copy(),
    }


def make_contacts(
    contact_kind: str,
    state,
    nu: int,
    left_frame_id: int,
    right_frame_id: int,
    refs: dict,
):
    contacts = crocoddyl.ContactModelMultiple(state, nu)
    zero_gains = np.zeros(2, dtype=float)

    if contact_kind == "6D":
        contacts.addContact(
            LEFT_FOOT_FRAME,
            crocoddyl.ContactModel6D(
                state,
                left_frame_id,
                refs["left_placement"],
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
                refs["right_placement"],
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                nu,
                zero_gains,
            ),
        )
    elif contact_kind == "3D":
        contacts.addContact(
            LEFT_FOOT_FRAME,
            crocoddyl.ContactModel3D(
                state,
                left_frame_id,
                refs["left_translation"],
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                nu,
                zero_gains,
            ),
        )
        contacts.addContact(
            RIGHT_FOOT_FRAME,
            crocoddyl.ContactModel3D(
                state,
                right_frame_id,
                refs["right_translation"],
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                nu,
                zero_gains,
            ),
        )
    else:
        raise ValueError(f"Unknown contact kind: {contact_kind}")

    return contacts


def build_action_model(
    contact_kind: str,
    damping: float,
    model: pin.Model,
    x0: np.ndarray,
    left_frame_id: int,
    right_frame_id: int,
    refs: dict,
):
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    nu = actuation.nu
    contacts = make_contacts(
        contact_kind,
        state,
        nu,
        left_frame_id,
        right_frame_id,
        refs,
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


def support_errors(
    model: pin.Model,
    xs: list[np.ndarray],
    left_frame_id: int,
    right_frame_id: int,
    refs: dict,
):
    data = model.createData()
    max_left_translation = 0.0
    max_right_translation = 0.0
    max_left_rotation = 0.0
    max_right_rotation = 0.0

    for x in xs:
        pin.forwardKinematics(model, data, x[:model.nq])
        pin.updateFramePlacements(model, data)

        max_left_translation = max(
            max_left_translation,
            float(
                np.linalg.norm(
                    data.oMf[left_frame_id].translation - refs["left_translation"]
                )
            ),
        )
        max_right_translation = max(
            max_right_translation,
            float(
                np.linalg.norm(
                    data.oMf[right_frame_id].translation - refs["right_translation"]
                )
            ),
        )
        max_left_rotation = max(
            max_left_rotation,
            float(
                np.linalg.norm(
                    pin.log3(
                        refs["left_placement"].rotation.T
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
                        refs["right_placement"].rotation.T
                        @ data.oMf[right_frame_id].rotation
                    )
                )
            ),
        )

    return {
        "max_left_translation_error_m": float(max_left_translation),
        "max_right_translation_error_m": float(max_right_translation),
        "max_left_rotation_error_rad": float(max_left_rotation),
        "max_right_rotation_error_rad": float(max_right_rotation),
    }


def finite_scalar(value) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def run_case(
    contact_kind: str,
    damping: float,
    model: pin.Model,
    x0: np.ndarray,
    left_frame_id: int,
    right_frame_id: int,
    refs: dict,
):
    action_model, state, nu = build_action_model(
        contact_kind,
        damping,
        model,
        x0,
        left_frame_id,
        right_frame_id,
        refs,
    )
    u_zero = np.zeros(nu, dtype=float)
    data = action_model.createData()

    result = {
        "contact_kind": contact_kind,
        "damping": float(damping),
        "completed_steps": 0,
        "finite": False,
    }

    try:
        action_model.calc(data, x0, u_zero)
        x_one = np.asarray(data.xnext, dtype=float).copy()
        one_step_difference = float(
            np.linalg.norm(state.diff(x0, x_one))
        )
        result["one_step_state_difference_norm"] = finite_scalar(
            one_step_difference
        )

        xs = [x0.copy()]
        x = x0.copy()
        termination = "completed"

        for step in range(ROLLOUT_STEPS):
            action_model.calc(data, x, u_zero)
            x_next = np.asarray(data.xnext, dtype=float).copy()

            if not np.all(np.isfinite(x_next)):
                termination = "non_finite_state"
                break

            difference = float(np.linalg.norm(state.diff(x0, x_next)))
            if not np.isfinite(difference) or difference > 100.0:
                termination = "diverged_state"
                break

            xs.append(x_next)
            x = x_next

        result["completed_steps"] = int(len(xs) - 1)
        result["termination"] = termination
        result["finite"] = bool(termination == "completed")
        result["final_state_difference_norm"] = finite_scalar(
            np.linalg.norm(state.diff(x0, xs[-1]))
        )
        result.update(support_errors(
            model,
            xs,
            left_frame_id,
            right_frame_id,
            refs,
        ))
    except Exception as exc:
        result["termination"] = "exception"
        result["exception"] = str(exc)

    return result


def concise_report(result: dict):
    if result.get("termination") == "exception":
        return (
            f"exception: {result.get('exception', 'unknown')}"
        )
    return (
        f"steps={result.get('completed_steps', 0):02d}/{ROLLOUT_STEPS}, "
        f"finite={result.get('finite', False)}, "
        f"one-step={result.get('one_step_state_difference_norm')}, "
        f"final={result.get('final_state_difference_norm')}, "
        f"term={result.get('termination')}"
    )


def main():
    print("===== STAGE 7C DIAGNOSTIC-5: CONTACT KKT DAMPING AUDIT =====")
    print("This script does not run FDDP and does not overwrite trajectories.")
    print(f"URDF: {URDF_PATH}")
    print(f"Damping sweep: {DAMPING_CASES}")

    if not URDF_PATH.exists():
        raise FileNotFoundError(f"URDF not found: {URDF_PATH}")

    model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
    if model.nq != 13 or model.nv != 12:
        raise RuntimeError(
            f"Unexpected model dimensions nq={model.nq}, nv={model.nv}; "
            "expected nq=13, nv=12."
        )

    x0 = load_x0(X0_PATH, model.nq + model.nv)
    left_frame_id = model.getFrameId(LEFT_FOOT_FRAME)
    right_frame_id = model.getFrameId(RIGHT_FOOT_FRAME)
    refs = get_support_references(
        model, x0, left_frame_id, right_frame_id
    )

    # Directly measure the same rank deficiency that matters to KKT dynamics.
    data = model.createData()
    pin.forwardKinematics(model, data, x0[:model.nq])
    pin.updateFramePlacements(model, data)
    left_jacobian = np.asarray(
        pin.computeFrameJacobian(
            model,
            data,
            x0[:model.nq],
            left_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        ),
        dtype=float,
    )
    right_jacobian = np.asarray(
        pin.computeFrameJacobian(
            model,
            data,
            x0[:model.nq],
            right_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        ),
        dtype=float,
    )
    rank_6d, tolerance_6d, singular_6d = numeric_rank(
        np.vstack([left_jacobian, right_jacobian])
    )
    rank_3d, tolerance_3d, singular_3d = numeric_rank(
        np.vstack([left_jacobian[:3, :], right_jacobian[:3, :]])
    )

    print(
        f"Two-foot 6D contact Jacobian rank: {rank_6d}/12 "
        f"(nullity {model.nv - rank_6d})"
    )
    print(
        f"Two-foot 3D contact Jacobian rank: {rank_3d}/6 "
        f"(nullity {model.nv - rank_3d})"
    )
    print()

    results = {"6D": [], "3D": []}
    for kind in ("6D", "3D"):
        print(f"----- {kind} CONTACTS -----")
        for damping in DAMPING_CASES:
            result = run_case(
                kind,
                damping,
                model,
                x0,
                left_frame_id,
                right_frame_id,
                refs,
            )
            results[kind].append(result)
            print(
                f"  damping={damping:.0e}: {concise_report(result)}"
            )
        print()

    # Prefer the smallest stable 6D damping. "Stable" here means finite for the
    # 0.40 s hold and remaining close to the known standing state.
    candidates = []
    for result in results["6D"]:
        if not result.get("finite", False):
            continue
        final_difference = result.get("final_state_difference_norm")
        left_error = result.get("max_left_translation_error_m")
        right_error = result.get("max_right_translation_error_m")
        if (
            final_difference is not None
            and final_difference < 1e-3
            and left_error is not None
            and right_error is not None
            and left_error < 1e-4
            and right_error < 1e-4
        ):
            candidates.append(result)

    recommended_damping = (
        min(candidates, key=lambda item: item["damping"])["damping"]
        if candidates
        else None
    )

    interpretation = (
        (
            "A stable 6D flat-foot damping candidate was found. Future "
            "double-support FDDP models must pass this non-zero value as "
            "JMinvJt_damping."
        )
        if recommended_damping is not None
        else
        (
            "No tested 6D damping value produced a stable static hold. "
            "Do not run another FDDP solve; inspect the model/contact frames "
            "and constrained forward dynamics first."
        )
    )

    output = {
        "stage": "7C_contact_kkt_damping_audit",
        "purpose": (
            "Find a non-zero operational-space-inertia damping value for "
            "rank-deficient bilateral contact dynamics before resuming FDDP."
        ),
        "model": {
            "urdf": str(URDF_PATH),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": 6,
        },
        "contact_jacobian_rank": {
            "two_foot_6d": {
                "rank": int(rank_6d),
                "rows": 12,
                "nullity": int(model.nv - rank_6d),
                "svd_tolerance": float(tolerance_6d),
                "singular_values": [float(v) for v in singular_6d],
            },
            "two_foot_3d": {
                "rank": int(rank_3d),
                "rows": 6,
                "nullity": int(model.nv - rank_3d),
                "svd_tolerance": float(tolerance_3d),
                "singular_values": [float(v) for v in singular_3d],
            },
        },
        "damping_cases": [float(v) for v in DAMPING_CASES],
        "results": results,
        "recommended_6d_damping": (
            float(recommended_damping)
            if recommended_damping is not None
            else None
        ),
        "interpretation": interpretation,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        yaml.safe_dump(output, sort_keys=False),
        encoding="utf-8",
    )

    print("Recommendation:")
    print(f"  {interpretation}")
    print(f"  recommended_6d_damping = {recommended_damping}")
    print()
    print(f"Saved:\n  {OUTPUT_PATH}")
    print("CONTACT KKT DAMPING AUDIT COMPLETED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nCONTACT KKT DAMPING AUDIT FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
