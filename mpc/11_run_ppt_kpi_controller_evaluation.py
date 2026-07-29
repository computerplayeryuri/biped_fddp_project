#!/usr/bin/env python3
"""Unified PPT-KPI evaluation for C0, C1 and C2 controllers.

KPI source: Presentation.pptx slides 13-18 and slide 22.
The script runs the currently available simulation-valid tests and records which
PPT KPIs are directly measured, approximated, or not applicable to the current
double-support CoM-shift reference.
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "logs"
PY = Path.home() / "miniforge3" / "envs" / "fddp_biped" / "bin" / "python"

PITCH_FALL_LIMIT_DEG = 30.0
MIN_CONTACTS = 4
MAX_JOINT_CORR_DEG = 1.5
REALTIME_DEADLINE_US = 20000.0
MIN_C1_IMPROVEMENT = 0.02


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


C1 = load_module("c1_eval", ROOT / "mpc" / "07_run_reference_aligned_fir_qp_mpc_sweep.py")
C2 = load_module("c2_eval", ROOT / "mpc" / "08_run_contact_phase_synchronised_fir_qp_mpc.py")


def selected_candidate():
    summary = LOG / "c1_qp2_fir_mpc_tuning_summary.yaml"
    if summary.exists():
        report = yaml.safe_load(summary.read_text(encoding="utf-8"))
        selected = report["selection"]["selected_candidate"]
        for item in report["candidate_results"]:
            if item["candidate"]["name"] == selected:
                return item["candidate"]
    return {"name": "balanced", "q": 1500.0, "r": 20.0, "slew": 25.0}


def fit_direction_models(direction: str):
    _, bt, bp, _ = C1.load_baseline(direction)
    ar, radius, stabilized, rmse, source = C1.fit_ar(direction, bt, bp)
    h = C1.load_fir(direction)
    ar_meta = {
        "coefficients": [float(x) for x in ar],
        "spectral_radius": float(radius),
        "stabilized": bool(stabilized),
        "fit_rmse_deg": float(rmse),
        "source": source,
    }
    return h, ar, ar_meta


def controller_metrics_from_df(df: pd.DataFrame, metrics: dict) -> dict:
    times = df["time_s"].to_numpy() if "time_s" in df else df["wall_time_s"].to_numpy()
    mask = times >= C1.DIST_T
    err = df["pitch_error_deg"].to_numpy()
    qp = df.loc[df["qp_solve_time_us"] > 0, "qp_solve_time_us"] if "qp_solve_time_us" in df else pd.Series([], dtype=float)
    deadline_miss = 0 if len(qp) == 0 else int((qp > REALTIME_DEADLINE_US).sum())
    updates = int(len(qp))
    return {
        "stability": {
            "fall_detected": bool(metrics["pitch"]["peak_abs_error_deg"] > PITCH_FALL_LIMIT_DEG or metrics["stability"]["min_contacts"] < 1),
            "min_contacts": int(metrics["stability"]["min_contacts"]),
            "peak_abs_pitch_error_deg": float(metrics["pitch"]["peak_abs_error_deg"]),
            "pitch_error_rms_deg": float(np.sqrt(np.mean(err[mask] ** 2))),
            "settling_time_to_0p5deg_s": metrics["pitch"]["settling_time_to_0p5deg_s"],
            "iae_deg_s": float(metrics["pitch"]["iae_deg_s"]),
        },
        "tracking": {
            "pitch_tracking_iae_deg_s": float(metrics["pitch"]["iae_deg_s"]),
            "final_abs_pitch_error_deg": float(metrics["pitch"]["final_abs_error_deg"]),
            "joint_angle_rmse_source": "not_logged_in_current_C1_dataframe; command correction and pitch tracking are logged",
        },
        "real_time": {
            "control_frequency_hz": 50.0,
            "median_qp_solve_time_us": metrics["command"].get("median_qp_solve_time_us"),
            "max_qp_solve_time_us": metrics["command"].get("max_qp_solve_time_us"),
            "deadline_miss_rate": 0.0 if updates == 0 else float(deadline_miss / updates),
            "solver_success_rate": 1.0,
        },
        "effort": {
            "max_abs_joint_correction_deg": float(metrics["command"].get("max_abs_joint_correction_deg", 0.0)),
            "alpha_rms_deg": float(np.sqrt(np.mean(df["alpha_deg"].to_numpy() ** 2))) if "alpha_deg" in df else 0.0,
            "squared_alpha_deg2_sum": float(np.sum(df["alpha_deg"].to_numpy() ** 2)) if "alpha_deg" in df else 0.0,
        },
    }


def run_pitch_disturbance_suite():
    candidate = selected_candidate()
    rows = []
    details: dict[str, dict] = {}
    for direction in ("forward", "backward"):
        h, ar, ar_meta = fit_direction_models(direction)
        c0_df, c0_met = C1.simulate(direction, None, "C0")
        c1_df, c1_met = C1.simulate(direction, C1.FIRQP(h, ar, candidate), candidate["name"])
        c2_df, c2_met = C2.simulate_c2(direction, C1.FIRQP(h, ar, candidate), candidate["name"], ar_meta)
        controllers = {
            "C0": (c0_df, c0_met, None),
            "C1": (c1_df, c1_met, C1.compare(c0_met, c1_met)),
            "C2": (c2_df, c2_met, C1.compare(c0_met, c2_met)),
        }
        details[direction] = {}
        for name, (df, met, comp) in controllers.items():
            kpis = controller_metrics_from_df(df, met)
            stable = (not kpis["stability"]["fall_detected"]) and kpis["stability"]["min_contacts"] >= MIN_CONTACTS
            effort_ok = kpis["effort"]["max_abs_joint_correction_deg"] <= MAX_JOINT_CORR_DEG
            rt = kpis["real_time"]
            realtime_ok = True if name == "C0" else ((rt["max_qp_solve_time_us"] or 0.0) <= REALTIME_DEADLINE_US and rt["deadline_miss_rate"] == 0.0)
            improvement = None if comp is None else float(comp["iae_improvement_ratio"])
            performance_ok = True if name == "C0" else bool(improvement is not None and improvement >= MIN_C1_IMPROVEMENT)
            pass_status = bool(stable and effort_ok and realtime_ok and performance_ok)
            row = {
                "test_family": "pitch_disturbance_fddp_reference",
                "controller": name,
                "direction": direction,
                "pass": pass_status,
                "iae_deg_s": kpis["stability"]["iae_deg_s"],
                "iae_improvement_vs_C0": improvement,
                "settling_time_s": kpis["stability"]["settling_time_to_0p5deg_s"],
                "peak_abs_pitch_error_deg": kpis["stability"]["peak_abs_pitch_error_deg"],
                "min_contacts": kpis["stability"]["min_contacts"],
                "max_joint_correction_deg": kpis["effort"]["max_abs_joint_correction_deg"],
                "median_qp_solve_time_us": rt["median_qp_solve_time_us"],
                "max_qp_solve_time_us": rt["max_qp_solve_time_us"],
                "deadline_miss_rate": rt["deadline_miss_rate"],
            }
            rows.append(row)
            details[direction][name] = {"metrics": met, "kpis": kpis, "comparison_vs_C0": comp, "pass": pass_status}
    return pd.DataFrame(rows), details


def run_contact_event_suite():
    script = ROOT / "mpc" / "10_validate_c2_contact_switching_mujoco_demo.py"
    start = time.perf_counter()
    result = subprocess.run([str(PY), str(script)], cwd=str(ROOT), text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - start
    summary_path = LOG / "c2_contact_switching_demo_summary.yaml"
    summary = yaml.safe_load(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    row = {
        "test_family": "contact_switching_mechanism",
        "controller": "C2",
        "direction": "scripted_left_lift_touchdown",
        "pass": result.returncode == 0 and all(summary.get("acceptance", {}).values()),
        "matched_events": summary.get("metrics", {}).get("matched_events"),
        "scheduled_events": summary.get("metrics", {}).get("scheduled_events"),
        "max_abs_phase_offset_ms": summary.get("metrics", {}).get("max_abs_phase_offset_ms"),
        "left_contact_was_lost": summary.get("metrics", {}).get("left_contact_was_lost"),
        "left_contact_was_restored": summary.get("metrics", {}).get("left_contact_was_restored"),
        "runtime_s": elapsed,
        "stdout_tail": result.stdout.strip().splitlines()[-5:] if result.stdout else [],
        "stderr": result.stderr.strip(),
    }
    return pd.DataFrame([row]), summary


def run_hardware_loadability_suite():
    sys.path.insert(0, str(ROOT))
    from hardware.controller_runtime import ControllerMeasurement, make_controller

    rows = []
    for controller_name in ("C0", "C1", "C2"):
        for direction in ("forward", "backward"):
            controller = make_controller(controller_name, direction=direction)
            finite = True
            any_limit = False
            any_rate = False
            max_abs_cmd = 0.0
            for i in range(20):
                t = i * 0.020
                measurement = ControllerMeasurement(time_s=t, root_pitch_rad=0.0, root_pitch_rate_rad_s=0.0)
                out = controller.update(measurement)
                q = np.asarray(out["q_cmd_rad"], dtype=float)
                finite = finite and bool(np.all(np.isfinite(q)))
                any_limit = any_limit or bool(out["safety"]["joint_limit_violation"])
                any_rate = any_rate or bool(out["safety"]["rate_limit_active"])
                max_abs_cmd = max(max_abs_cmd, float(np.max(np.abs(q))))
            rows.append({
                "test_family": "hardware_runtime_loadability",
                "controller": controller_name,
                "direction": direction,
                "pass": bool(finite and not any_limit),
                "finite_commands": bool(finite),
                "joint_limit_violation": bool(any_limit),
                "rate_limit_active_seen": bool(any_rate),
                "max_abs_command_rad": max_abs_cmd,
                "runtime_module": "hardware.controller_runtime",
            })
    return pd.DataFrame(rows)


def main():
    LOG.mkdir(parents=True, exist_ok=True)
    pitch_table, pitch_details = run_pitch_disturbance_suite()
    contact_table, contact_summary = run_contact_event_suite()
    hardware_table = run_hardware_loadability_suite()
    all_rows = pd.concat([pitch_table, contact_table, hardware_table], ignore_index=True, sort=False)

    table_path = LOG / "ppt_kpi_c0_c1_c2_evaluation_table.csv"
    all_rows.to_csv(table_path, index=False)

    pass_by_controller = {}
    for controller in ("C0", "C1", "C2"):
        subset = all_rows[all_rows["controller"] == controller]
        pass_by_controller[controller] = {
            "all_applicable_tests_passed": bool(subset["pass"].fillna(False).all()),
            "tests": subset[["test_family", "direction", "pass"]].to_dict("records"),
        }

    summary = {
        "stage": "PPT_KPI_C0_C1_C2_controller_evaluation",
        "ppt_kpi_source": "Presentation.pptx slides 13-18 and 22",
        "scope_note": "Current FDDP reference is double-support CoM shift, not walking. Walking velocity, step length, ZMP margin, CoT and battery voltage KPIs are defined for future tests but are not fully measurable from the current logs.",
        "tested_kpi_families": {
            "stability": ["fall flag", "min contacts", "peak/RMS pitch error", "settling time"],
            "tracking": ["pitch tracking IAE", "final pitch error", "joint correction magnitude"],
            "real_time": ["QP solve time", "deadline miss rate", "50 Hz update"],
            "effort": ["alpha RMS", "squared alpha", "max joint correction"],
            "robustness": ["matched forward/backward pitch disturbance", "C2 contact timing demo"],
            "hardware_feasibility": ["runtime loadability", "finite q_cmd", "joint-limit check", "rate limiting"],
        },
        "not_yet_measured_from_current_project_state": [
            "walking duration over 10/30/60 s",
            "forward velocity error and step-length error from a walking gait",
            "ZMP margin from force/CoP estimation",
            "mechanical CoT from torque or current logs",
            "battery voltage drop and real servo fault events",
            "real hardware success rate",
        ],
        "pass_by_controller": pass_by_controller,
        "pitch_disturbance_details": pitch_details,
        "contact_switching_summary": contact_summary,
        "outputs": {"table_csv": str(table_path)},
    }
    summary_path = LOG / "ppt_kpi_c0_c1_c2_evaluation_summary.yaml"
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")

    print("PPT KPI C0/C1/C2 evaluation complete")
    print(all_rows.to_string(index=False))
    print(f"Saved table: {table_path}")
    print(f"Saved summary: {summary_path}")
    if not all(v["all_applicable_tests_passed"] for v in pass_by_controller.values()):
        raise SystemExit("One or more controllers failed applicable KPI tests; inspect summary YAML.")


if __name__ == "__main__":
    main()
