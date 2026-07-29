#!/usr/bin/env python3
"""
C1-B0 — Fixed-phase FDDP-MPC runtime feasibility benchmark.

Purpose
-------
Benchmark nominal-trajectory-warm-started FDDP replanning under several fixed
iteration budgets. This is the necessary bridge between:

  C1-A: "the nonlinear fixed-phase replan is dynamically feasible"
and
  C1-B: "the controller can be run repeatedly at a chosen update rate".

It does NOT command MuJoCo or claim real-time capability. It measures it.

For each direction and iteration budget, the script reports:
  - wall-clock solve time;
  - final shooting gap;
  - 0.30-s terminal pitch recovery;
  - terminal pitch-rate error;
  - first q_cmd correction;
  - joint-limit margin;
  - whether the candidate satisfies the physical/constraint gates.

Important:
  the C0 nominal state/control trajectory is used as a warm start. This is
  representative of the intended FDDP-to-MPC architecture, where the offline
  FDDP reference is available at every control update.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import sys
import time

import crocoddyl
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import yaml


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "model" / "biped_v1.urdf"
TRAJ_DIR = ROOT / "transfer" / "trajectories"
LOG_DIR = ROOT / "results" / "logs"
FIG_DIR = ROOT / "results" / "figures"

LEFT_FOOT = "left_foot_contact_fixed"
RIGHT_FOOT = "right_foot_contact_fixed"

DT = 0.010
HORIZON = 30
PHASE_INDEX = 40
DAMPING = 1.0e-8
CONTACT_GAINS = np.array([0.0, 0.0])

INJECTED_PITCH_DEG = 2.0
INJECTED_PITCH_RATE_RAD_S = 0.25

# Same C1-A v2 task weights.
RUN_W = np.array([
    5, 5, 5,
    2500, 20, 20,
    10, 10, 10, 10, 10, 10,
    2, 2, 2,
    1200, 10, 10,
    1, 1, 1, 1, 1, 1,
], dtype=float)
TERM_W = np.array([
    30, 30, 30,
    30000, 100, 100,
    100, 100, 100, 100, 100, 100,
    20, 20, 20,
    8000, 50, 50,
    5, 5, 5, 5, 5, 5,
], dtype=float)
R_WEIGHT = 2.0e-3
COM_W = np.array([2e2, 5e4, 2e2], dtype=float)

# Physical/constraint gates identical in spirit to C1-A v2.
MAX_GAP = 1e-7
MAX_TERMINAL_PITCH_ERROR_DEG = 0.50
MIN_PITCH_REDUCTION = 0.70
MAX_TERMINAL_RATE_ERROR = 0.10
MAX_NEXT_Q_CORRECTION_DEG = 5.0
MIN_LIMIT_MARGIN_RAD = 1e-2


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def root_pitch(model: pin.Model, x: np.ndarray) -> float:
    q = x[:model.nq]
    quat = pin.Quaternion(q[6], q[3], q[4], q[5])
    rotation = quat.matrix()
    return float(np.arctan2(rotation[2, 1], rotation[2, 2]))


def root_pitch_rate(model: pin.Model, x: np.ndarray) -> float:
    return float(x[model.nq + 3])


def compute_com_series(model: pin.Model, xs: np.ndarray) -> np.ndarray:
    data = model.createData()
    com = np.zeros((len(xs), 3), dtype=float)
    for index, x in enumerate(xs):
        com[index] = np.asarray(
            pin.centerOfMass(model, data, x[:model.nq]),
            dtype=float,
        ).reshape(3)
    return com


def support_references(
    model: pin.Model, x: np.ndarray, left_frame_id: int, right_frame_id: int
):
    data = model.createData()
    pin.forwardKinematics(model, data, x[:model.nq])
    pin.updateFramePlacements(model, data)
    return data.oMf[left_frame_id].copy(), data.oMf[right_frame_id].copy()


def build_action(
    state,
    actuation,
    x_ref: np.ndarray,
    u_ref: np.ndarray,
    com_ref: np.ndarray,
    state_weights: np.ndarray,
    control_weight: float,
    left_frame_id: int,
    right_frame_id: int,
    left_pose,
    right_pose,
    dt: float,
):
    nu = actuation.nu

    contacts = crocoddyl.ContactModelMultiple(state, nu)
    contacts.addContact(
        LEFT_FOOT,
        crocoddyl.ContactModel6D(
            state,
            left_frame_id,
            left_pose,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            nu,
            CONTACT_GAINS,
        ),
    )
    contacts.addContact(
        RIGHT_FOOT,
        crocoddyl.ContactModel6D(
            state,
            right_frame_id,
            right_pose,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            nu,
            CONTACT_GAINS,
        ),
    )

    costs = crocoddyl.CostModelSum(state, nu)

    state_residual = crocoddyl.ResidualModelState(state, x_ref, nu)
    state_activation = crocoddyl.ActivationModelWeightedQuad(state_weights)
    costs.addCost(
        "weighted_state_tracking",
        crocoddyl.CostModelResidual(state, state_activation, state_residual),
        1.0,
    )

    if control_weight > 0.0:
        control_residual = crocoddyl.ResidualModelControl(state, u_ref)
        costs.addCost(
            "nominal_control_tracking",
            crocoddyl.CostModelResidual(state, control_residual),
            control_weight,
        )

    com_residual = crocoddyl.ResidualModelCoMPosition(state, com_ref, nu)
    com_activation = crocoddyl.ActivationModelWeightedQuad(COM_W)
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
        DAMPING,
        False,
    )
    return crocoddyl.IntegratedActionModelEuler(differential, dt)


def make_problem(direction: str):
    ref_path = TRAJ_DIR / f"com_shift_{direction}_ref.npz"
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing C0 reference: {ref_path}")

    model = pin.buildModelFromUrdf(str(URDF), pin.JointModelFreeFlyer())
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)

    if (model.nq, model.nv, state.ndx, actuation.nu) != (13, 12, 24, 6):
        raise RuntimeError(
            f"Unexpected dimensions: nq={model.nq}, nv={model.nv}, "
            f"ndx={state.ndx}, nu={actuation.nu}"
        )

    with np.load(ref_path) as loaded:
        xs_all = np.asarray(loaded["state_ref"], dtype=float)
        us_all = np.asarray(loaded["tau_ref"], dtype=float)
        t_all = np.asarray(loaded["t_ref"], dtype=float)

    if PHASE_INDEX + HORIZON >= len(xs_all):
        raise RuntimeError("C1 prediction window exceeds C0 reference.")

    nominal_xs = xs_all[PHASE_INDEX:PHASE_INDEX + HORIZON + 1].copy()
    nominal_us = us_all[PHASE_INDEX:PHASE_INDEX + HORIZON].copy()
    nominal_com = compute_com_series(model, nominal_xs)

    left_frame_id = model.getFrameId(LEFT_FOOT)
    right_frame_id = model.getFrameId(RIGHT_FOOT)
    left_pose, right_pose = support_references(
        model, nominal_xs[0], left_frame_id, right_frame_id
    )

    running = [
        build_action(
            state,
            actuation,
            nominal_xs[k],
            nominal_us[k],
            nominal_com[k],
            RUN_W,
            R_WEIGHT,
            left_frame_id,
            right_frame_id,
            left_pose,
            right_pose,
            DT,
        )
        for k in range(HORIZON)
    ]
    terminal = build_action(
        state,
        actuation,
        nominal_xs[-1],
        np.zeros(actuation.nu, dtype=float),
        nominal_com[-1],
        TERM_W,
        0.0,
        left_frame_id,
        right_frame_id,
        left_pose,
        right_pose,
        0.0,
    )

    dx = np.zeros(state.ndx, dtype=float)
    dx[3] = np.deg2rad(INJECTED_PITCH_DEG)
    dx[15] = INJECTED_PITCH_RATE_RAD_S
    measured_x0 = np.asarray(state.integrate(nominal_xs[0], dx), dtype=float)

    problem = crocoddyl.ShootingProblem(measured_x0, running, terminal)
    return {
        "model": model,
        "state": state,
        "problem": problem,
        "nominal_xs": nominal_xs,
        "nominal_us": nominal_us,
        "t_phase_s": float(t_all[PHASE_INDEX]),
        "reference_path": ref_path,
    }


def gap_max(problem, state, xs, us) -> float:
    problem.calc(xs, us)
    gaps = []
    for k, data in enumerate(problem.runningDatas):
        vector = np.asarray(state.diff(xs[k + 1], data.xnext), dtype=float)
        gaps.append(float(np.linalg.norm(vector)))
    return float(np.max(gaps))


def joint_margin(model: pin.Model, xs: np.ndarray) -> float:
    q = xs[:, 7:model.nq]
    low = np.asarray(model.lowerPositionLimit[7:model.nq], dtype=float)
    high = np.asarray(model.upperPositionLimit[7:model.nq], dtype=float)
    return float(min(np.min(q - low), np.min(high - q)))


def one_run(context: dict, iteration_budget: int) -> dict:
    model = context["model"]
    state = context["state"]
    problem = context["problem"]
    nominal_xs = context["nominal_xs"]
    nominal_us = context["nominal_us"]

    xs_seed = [x.copy() for x in nominal_xs]
    xs_seed[0] = problem.x0.copy()
    us_seed = [u.copy() for u in nominal_us]

    solver = crocoddyl.SolverFDDP(problem)
    solver.th_stop = 1e-7

    start = time.perf_counter_ns()
    solver_return = bool(solver.solve(xs_seed, us_seed, iteration_budget, False))
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6

    xs = np.asarray(solver.xs, dtype=float)
    us = np.asarray(solver.us, dtype=float)
    terminal_pitch_error_deg = float(
        np.rad2deg(root_pitch(model, xs[-1]) - root_pitch(model, nominal_xs[-1]))
    )
    initial_pitch_error_deg = float(
        np.rad2deg(root_pitch(model, problem.x0) - root_pitch(model, nominal_xs[0]))
    )
    pitch_reduction = float(
        1.0
        - abs(terminal_pitch_error_deg)
        / max(abs(initial_pitch_error_deg), 1e-12)
    )
    terminal_rate_error = float(
        root_pitch_rate(model, xs[-1]) - root_pitch_rate(model, nominal_xs[-1])
    )
    correction_deg = float(
        np.max(np.abs(np.rad2deg(xs[1, 7:model.nq] - nominal_xs[1, 7:model.nq])))
    )
    max_gap = gap_max(
        problem,
        state,
        [x.copy() for x in xs],
        [u.copy() for u in us],
    )
    margin = joint_margin(model, xs)
    finite = bool(np.all(np.isfinite(xs)) and np.all(np.isfinite(us)))

    physics_pass = bool(
        finite
        and max_gap <= MAX_GAP
        and abs(terminal_pitch_error_deg) <= MAX_TERMINAL_PITCH_ERROR_DEG
        and pitch_reduction >= MIN_PITCH_REDUCTION
        and abs(terminal_rate_error) <= MAX_TERMINAL_RATE_ERROR
        and correction_deg <= MAX_NEXT_Q_CORRECTION_DEG
        and margin >= MIN_LIMIT_MARGIN_RAD
    )

    return {
        "iteration_budget": int(iteration_budget),
        "wall_clock_solve_time_ms": float(elapsed_ms),
        "solver_return_success": bool(solver_return),
        "solver_stop_criterion": float(solver.stop),
        "final_max_shooting_gap": float(max_gap),
        "terminal_pitch_error_deg": float(terminal_pitch_error_deg),
        "pitch_error_reduction_ratio": float(pitch_reduction),
        "terminal_pitch_rate_error_rad_s": float(terminal_rate_error),
        "max_first_q_command_correction_deg": float(correction_deg),
        "minimum_joint_limit_margin_rad": float(margin),
        "physics_constraint_pass": physics_pass,
    }


def aggregate(runs: list[dict]) -> dict:
    scalar_keys = [
        "wall_clock_solve_time_ms",
        "solver_stop_criterion",
        "final_max_shooting_gap",
        "terminal_pitch_error_deg",
        "pitch_error_reduction_ratio",
        "terminal_pitch_rate_error_rad_s",
        "max_first_q_command_correction_deg",
        "minimum_joint_limit_margin_rad",
    ]
    output = {
        "iteration_budget": int(runs[0]["iteration_budget"]),
        "repeat_count": len(runs),
        "solver_return_success_all": bool(
            all(item["solver_return_success"] for item in runs)
        ),
        "physics_constraint_pass_all": bool(
            all(item["physics_constraint_pass"] for item in runs)
        ),
    }
    for key in scalar_keys:
        values = np.asarray([item[key] for item in runs], dtype=float)
        output[f"{key}_median"] = float(np.median(values))
        output[f"{key}_min"] = float(np.min(values))
        output[f"{key}_max"] = float(np.max(values))
    output["supports_10ms_update"] = bool(
        output["wall_clock_solve_time_ms_median"] <= 10.0
    )
    output["supports_20ms_update"] = bool(
        output["wall_clock_solve_time_ms_median"] <= 20.0
    )
    output["supports_50ms_update"] = bool(
        output["wall_clock_solve_time_ms_median"] <= 50.0
    )
    output["supports_100ms_update"] = bool(
        output["wall_clock_solve_time_ms_median"] <= 100.0
    )
    return output


def make_plot(direction: str, results: list[dict]) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    budgets = [item["iteration_budget"] for item in results]
    solve_ms = [item["wall_clock_solve_time_ms_median"] for item in results]
    terminal_error = [
        abs(item["terminal_pitch_error_deg_median"]) for item in results
    ]

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(budgets, solve_ms, marker="o", label="median solve time")
    axis.axhline(10.0, linestyle="--", label="10 ms / 100 Hz")
    axis.axhline(20.0, linestyle="--", label="20 ms / 50 Hz")
    axis.axhline(50.0, linestyle="--", label="50 ms / 20 Hz")
    axis.set_xlabel("FDDP iteration budget")
    axis.set_ylabel("Wall-clock solve time (ms)")
    axis.set_title(f"C1-B0 fixed-phase FDDP-MPC timing — {direction}")
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    path = FIG_DIR / f"c1_runtime_benchmark_{direction}_timing.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        budgets,
        terminal_error,
        marker="o",
        label="absolute terminal pitch error",
    )
    axis.axhline(0.50, linestyle="--", label="acceptance limit")
    axis.set_xlabel("FDDP iteration budget")
    axis.set_ylabel("Terminal root-pitch error (deg)")
    axis.set_title(f"C1-B0 fixed-phase FDDP-MPC recovery — {direction}")
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    path = FIG_DIR / f"c1_runtime_benchmark_{direction}_recovery.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)

    return path


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark C1 fixed-phase FDDP-MPC iteration budgets."
    )
    parser.add_argument(
        "--directions",
        choices=("forward", "backward", "both"),
        default="both",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Repeated solves per iteration budget (default: 2).",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 20, 50],
        help="FDDP iteration budgets to benchmark.",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")
    if any(value < 1 for value in args.budgets):
        raise ValueError("All --budgets values must be positive.")

    directions = (
        ("forward", "backward")
        if args.directions == "both"
        else (args.directions,)
    )

    print("===== C1-B0: FDDP-MPC RUNTIME FEASIBILITY BENCHMARK =====")
    print("This measures compute time; it does not claim real-time execution.")
    print(f"Directions: {', '.join(directions)}")
    print(f"Iteration budgets: {args.budgets}")
    print(f"Repeats per budget: {args.repeats}")
    print(
        "Synthetic measured state: "
        f"+{INJECTED_PITCH_DEG:.1f} deg pitch, "
        f"+{INJECTED_PITCH_RATE_RAD_S:.2f} rad/s pitch rate"
    )

    report = {
        "stage": "C1B0_fixed_phase_fddp_mpc_runtime_feasibility_benchmark",
        "purpose": (
            "Measure nominal-trajectory-warm-started FDDP-MPC solve time as "
            "a function of fixed iteration budget before online closed-loop implementation."
        ),
        "model": {
            "urdf": str(URDF),
            "prediction_horizon_nodes": HORIZON,
            "prediction_horizon_s": HORIZON * DT,
            "phase_index": PHASE_INDEX,
            "JMinvJt_damping": DAMPING,
        },
        "disturbance": {
            "root_pitch_deg": INJECTED_PITCH_DEG,
            "root_pitch_rate_rad_s": INJECTED_PITCH_RATE_RAD_S,
        },
        "benchmark_settings": {
            "iteration_budgets": [int(value) for value in args.budgets],
            "repeats": int(args.repeats),
            "nominal_trajectory_used_as_warm_start": True,
        },
        "directions": {},
    }

    all_rows = []
    for direction in directions:
        context = make_problem(direction)
        direction_results = []

        print(f"\n----- {direction.upper()} -----")
        for budget in args.budgets:
            individual = [one_run(context, budget) for _ in range(args.repeats)]
            result = aggregate(individual)
            direction_results.append(result)

            print(
                f"iter={budget:>3d} | "
                f"median={result['wall_clock_solve_time_ms_median']:.2f} ms | "
                f"gap={result['final_max_shooting_gap_median']:.2e} | "
                f"terminal pitch err={result['terminal_pitch_error_deg_median']:+.3f} deg | "
                f"recovery={100.0 * result['pitch_error_reduction_ratio_median']:.1f}% | "
                f"physical-pass={result['physics_constraint_pass_all']}"
            )

            row = {"direction": direction}
            row.update(result)
            all_rows.append(row)

        make_plot(direction, direction_results)
        report["directions"][direction] = {
            "c0_reference": str(context["reference_path"]),
            "phase_time_s": context["t_phase_s"],
            "results": direction_results,
            "timing_plot": str(
                FIG_DIR / f"c1_runtime_benchmark_{direction}_timing.png"
            ),
            "recovery_plot": str(
                FIG_DIR / f"c1_runtime_benchmark_{direction}_recovery.png"
            ),
        }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    yaml_path = LOG_DIR / "c1_runtime_feasibility_benchmark.yaml"
    csv_path = LOG_DIR / "c1_runtime_feasibility_benchmark.csv"
    write_yaml(yaml_path, report)

    if all_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    print("\nBENCHMARK COMPLETE")
    print(f"Saved YAML:\n  {yaml_path}")
    print(f"Saved CSV:\n  {csv_path}")
    print(
        "Interpretation: only choose a repeated online FDDP update period "
        "that exceeds the measured median solve time with margin."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nC1-B0 RUNTIME BENCHMARK FAILED: {error}", file=sys.stderr)
        sys.exit(1)
