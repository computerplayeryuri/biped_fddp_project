#!/usr/bin/env python3
"""One-factor C1 FIR-QP parameter ablation on the two-step FDDP reference."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
TABLE_CSV = ROOT / "results" / "logs" / "fddp_two_step_c1_parameter_ablation.csv"
TRACE_CSV = ROOT / "results" / "logs" / "fddp_two_step_c1_parameter_ablation_traces.csv"
SUMMARY_YAML = ROOT / "results" / "logs" / "fddp_two_step_c1_parameter_ablation.yaml"
FIGURE = ROOT / "results" / "figures" / "fddp_two_step_c1_parameter_ablation.png"

MIN_CONTACTS = 2
MIN_HEIGHT_RATIO = 0.65
MIN_IAE_IMROVEMENT_RATIO = 0.02
MAX_PEAK_INCREASE_RATIO = 0.05
MAX_ALPHA_DEG = 5.0


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_ablation_candidates() -> list[dict]:
    """Return balanced plus one-factor q/r/slew-penalty variants."""
    return [
        {"name": "balanced", "q": 1500.0, "r": 20.0, "slew": 25.0},
        {"name": "q_low", "q": 800.0, "r": 20.0, "slew": 25.0},
        {"name": "q_high", "q": 2500.0, "r": 20.0, "slew": 25.0},
        {"name": "r_low", "q": 1500.0, "r": 10.0, "slew": 25.0},
        {"name": "r_high", "q": 1500.0, "r": 50.0, "slew": 25.0},
        {"name": "slew_penalty_low", "q": 1500.0, "r": 20.0, "slew": 12.0},
        {"name": "slew_penalty_high", "q": 1500.0, "r": 20.0, "slew": 80.0},
    ]


def evaluate_candidate(metrics: dict, c0_metrics: dict) -> dict:
    iae_improvement = (
        c0_metrics["iae_deg_s"] - metrics["iae_deg_s"]
    ) / max(c0_metrics["iae_deg_s"], 1e-12)
    peak_increase = (
        metrics["peak_abs_error_deg"] - c0_metrics["peak_abs_error_deg"]
    ) / max(c0_metrics["peak_abs_error_deg"], 1e-12)
    contact_pass = metrics["min_contacts"] >= MIN_CONTACTS
    height_pass = (
        metrics["min_torso_height_m"]
        >= MIN_HEIGHT_RATIO * metrics["initial_torso_height_m"]
    )
    command_pass = metrics["max_abs_alpha_deg"] <= MAX_ALPHA_DEG
    performance_pass = (
        iae_improvement >= MIN_IAE_IMROVEMENT_RATIO
        and peak_increase <= MAX_PEAK_INCREASE_RATIO
    )
    return {
        "iae_improvement_ratio": float(iae_improvement),
        "peak_increase_ratio": float(peak_increase),
        "contact_pass": bool(contact_pass),
        "height_pass": bool(height_pass),
        "command_pass": bool(command_pass),
        "performance_pass": bool(performance_pass),
        "acceptance_passed": bool(
            contact_pass and height_pass and command_pass and performance_pass
        ),
    }


def build_result_row(candidate: dict, metrics: dict, evaluation: dict) -> dict:
    """Map the in-memory candidate name to the persisted candidate column."""
    return {
        "candidate": candidate["name"],
        "q": candidate["q"],
        "r": candidate["r"],
        "slew": candidate["slew"],
        **metrics,
        **evaluation,
    }


def _c0_metrics(c1, baseline: pd.DataFrame) -> dict:
    disturbed = pd.read_csv(c1.DISTURBANCE_CSV)
    time_s = disturbed["time_s"].to_numpy()
    measured = disturbed["measured_pitch_rad"].to_numpy()
    reference = np.interp(time_s, baseline["time_s"], baseline["root_pitch_rad"])
    error_deg = np.rad2deg(measured - reference)
    mask = time_s >= c1.DISTURBANCE_TIME_S
    return {
        "iae_deg_s": float(np.trapezoid(np.abs(error_deg[mask]), time_s[mask])),
        "peak_abs_error_deg": float(np.max(np.abs(error_deg[mask]))),
    }


def simulate_candidate(candidate: dict) -> tuple[pd.DataFrame, dict]:
    c1 = _load(ROOT / "mpc" / "16_validate_fddp_two_step_c1_fir_qp.py", "ablation_c1")
    common = _load(
        ROOT / "mpc" / "07_run_reference_aligned_fir_qp_mpc_sweep.py",
        "ablation_firqp",
    )
    proxy = c1._load_proxy()
    fir = yaml.safe_load(c1.FIR_YAML.read_text(encoding="utf-8"))
    ar = yaml.safe_load(c1.AR_YAML.read_text(encoding="utf-8"))
    controller = common.FIRQP(
        np.asarray(fir["fir_coefficients_rad_per_rad"], dtype=float),
        np.asarray(ar["coefficients"], dtype=float),
        candidate,
    )
    t_ref, q_ref = c1.load_two_step_reference()
    baseline = pd.read_csv(c1.BASELINE_CSV)
    model = mujoco.MjModel.from_xml_path(str(ROOT / "model" / "biped_planar.xml"))
    data = mujoco.MjData(model)
    mapping = proxy.resolve_joint_and_actuator_mapping(model)
    torso = proxy.resolve_torso_body(model)
    pitch_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "root_pitch_x"
    )
    if pitch_joint < 0:
        raise RuntimeError("root_pitch_x not found")
    pitch_qadr = int(model.jnt_qposadr[pitch_joint])
    pitch_dadr = int(model.jnt_dofadr[pitch_joint])
    for address, value in zip(mapping["qpos_addresses"], q_ref[0]):
        data.qpos[address] = value
    mujoco.mj_forward(model, data)
    initial_height = float(data.xpos[torso][2])
    actuator = proxy.PositionActuatorProxy(6, float(model.opt.timestep))
    actuator.reset(q_ref[0])
    update_steps = max(1, int(round(common.TS / model.opt.timestep)))
    duration = float(t_ref[-1])
    applied = False
    alpha = 0.0
    step = 0
    rows = []
    while data.time <= duration:
        now = min(float(data.time), duration)
        if not applied and now >= c1.DISTURBANCE_TIME_S:
            data.qpos[pitch_qadr] += c1.DISTURBANCE_PITCH_RAD
            data.qvel[pitch_dadr] += c1.DISTURBANCE_RATE_RAD_S
            mujoco.mj_forward(model, data)
            applied = True
        pitch_ref = float(
            np.interp(now, baseline["time_s"], baseline["root_pitch_rad"])
        )
        measured = float(proxy.torso_pitch_about_x(data, torso))
        if step % update_steps == 0:
            alpha = controller.update(measured - pitch_ref)
        q_cmd = proxy.interpolate_reference(now, t_ref, q_ref) + c1.G * alpha
        proxied = actuator.update(q_cmd)
        data.ctrl[:] = 0.0
        for actuator_id, value in zip(mapping["actuator_ids"], proxied):
            data.ctrl[actuator_id] = value
        mujoco.mj_step(model, data)
        step += 1
        measured_after = float(proxy.torso_pitch_about_x(data, torso))
        rows.append(
            {
                "candidate": candidate["name"],
                "time_s": float(data.time),
                "pitch_error_deg": float(np.rad2deg(measured_after - pitch_ref)),
                "alpha_deg": float(np.rad2deg(alpha)),
                "ncon": int(data.ncon),
                "torso_height_m": float(data.xpos[torso][2]),
            }
        )
    frame = pd.DataFrame(rows)
    time_s = frame["time_s"].to_numpy()
    error_deg = frame["pitch_error_deg"].to_numpy()
    mask = time_s >= c1.DISTURBANCE_TIME_S
    metrics = {
        "iae_deg_s": float(np.trapezoid(np.abs(error_deg[mask]), time_s[mask])),
        "peak_abs_error_deg": float(np.max(np.abs(error_deg[mask]))),
        "max_abs_alpha_deg": float(np.max(np.abs(frame["alpha_deg"]))),
        "min_contacts": int(frame["ncon"].min()),
        "initial_torso_height_m": initial_height,
        "min_torso_height_m": float(frame["torso_height_m"].min()),
    }
    return frame, metrics


def _plot(table: pd.DataFrame) -> None:
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    labels = table["candidate"].tolist()
    x = np.arange(len(labels))
    colors = ["#0072B2" if value else "#D55E00" for value in table["acceptance_passed"]]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].bar(x, 100.0 * table["iae_improvement_ratio"], color=colors)
    axes[0].axhline(100.0 * MIN_IAE_IMROVEMENT_RATIO, color="k", linestyle="--")
    axes[0].set_ylabel("IAE improvement (%)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(x, 100.0 * table["peak_increase_ratio"], color=colors)
    axes[1].axhline(100.0 * MAX_PEAK_INCREASE_RATIO, color="k", linestyle="--")
    axes[1].set_ylabel("Peak increase (%)")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("Two-step FDDP C1 one-factor FIR-QP parameter ablation")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)


def main() -> None:
    c1 = _load(ROOT / "mpc" / "16_validate_fddp_two_step_c1_fir_qp.py", "ablation_c1_inputs")
    required = (c1.BASELINE_CSV, c1.FIR_YAML, c1.AR_YAML, c1.DISTURBANCE_CSV)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"run mpc/16 first; missing: {missing}")
    baseline = pd.read_csv(c1.BASELINE_CSV)
    c0_metrics = _c0_metrics(c1, baseline)
    rows = []
    traces = []
    for candidate in build_ablation_candidates():
        frame, metrics = simulate_candidate(candidate)
        evaluation = evaluate_candidate(metrics, c0_metrics)
        rows.append(build_result_row(candidate, metrics, evaluation))
        traces.append(frame)
        print(
            f"{candidate['name']}: IAE={metrics['iae_deg_s']:.4f} deg s, "
            f"peak={metrics['peak_abs_error_deg']:.4f} deg, "
            f"pass={evaluation['acceptance_passed']}"
        )
    table = pd.DataFrame(rows)
    TABLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_CSV, index=False)
    pd.concat(traces, ignore_index=True).to_csv(TRACE_CSV, index=False)
    _plot(table)
    best = table.sort_values(
        ["acceptance_passed", "iae_improvement_ratio"], ascending=[False, False]
    ).iloc[0]
    report = {
        "stage": "two_step_fddp_c1_one_factor_parameter_ablation",
        "reference_npz": str(c1.NPZ_PATH),
        "tau_ref_usage": "analysis_only",
        "disturbance": {
            "time_s": c1.DISTURBANCE_TIME_S,
            "root_pitch_deg": float(np.rad2deg(c1.DISTURBANCE_PITCH_RAD)),
            "root_pitch_rate_rad_s": c1.DISTURBANCE_RATE_RAD_S,
        },
        "method": "balanced plus one-factor q/r/slew-penalty low/high variants",
        "acceptance": {
            "min_contacts": MIN_CONTACTS,
            "min_height_ratio": MIN_HEIGHT_RATIO,
            "min_iae_improvement_ratio": MIN_IAE_IMROVEMENT_RATIO,
            "max_peak_increase_ratio": MAX_PEAK_INCREASE_RATIO,
            "max_abs_alpha_deg": MAX_ALPHA_DEG,
        },
        "c0_metrics": c0_metrics,
        "candidate_results": rows,
        "best_by_acceptance_then_iae": str(best["candidate"]),
        "passing_candidates": table.loc[
            table["acceptance_passed"], "candidate"
        ].tolist(),
        "outputs": {
            "table_csv": str(TABLE_CSV),
            "trace_csv": str(TRACE_CSV),
            "figure": str(FIGURE),
        },
    }
    SUMMARY_YAML.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    print(SUMMARY_YAML)


if __name__ == "__main__":
    main()
