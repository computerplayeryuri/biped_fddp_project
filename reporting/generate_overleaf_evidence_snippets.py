#!/usr/bin/env python3
"""Generate Overleaf-ready thesis snippets from current simulation evidence."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml


SUMMARY_NAME = "ppt_kpi_c0_c1_c2_evaluation_summary.yaml"
TABLE_NAME = "ppt_kpi_c0_c1_c2_evaluation_table.csv"

KEY_FIGURES = [
    "results/figures/c2_phase_sync_forward_pitch_comparison.png",
    "results/figures/c2_phase_sync_backward_pitch_comparison.png",
    "results/figures/c2_phase_sync_forward_contacts_phase.png",
    "results/figures/c2_phase_sync_backward_contacts_phase.png",
    "results/figures/c2_contact_switching_demo_contacts.png",
    "results/figures/c2_contact_switching_demo_phase_offset.png",
    "results/figures/c2_contact_switching_demo_joint_retiming.png",
    "results/figures/c1_runtime_benchmark_forward_timing.png",
    "results/figures/fddp_com_shift_mujoco_replay_forward_body.png",
    "results/figures/fddp_com_shift_mujoco_replay_backward_body.png",
    "results/figures/stage8_scripted_alternating_step_contacts.png",
    "results/figures/stage8_scripted_alternating_step_phase_offset.png",
    "results/figures/stage8_scripted_alternating_step_body.png",
]


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected mapping in {path}")
    return data


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _as_int(value: str | None) -> int | None:
    number = _as_float(value)
    return None if number is None else int(round(number))


def build_context(project_root: Path) -> dict[str, Any]:
    """Collect the current thesis evidence from generated project outputs."""

    log_dir = project_root / "results" / "logs"
    summary = _read_yaml(log_dir / SUMMARY_NAME)
    rows = _read_csv(log_dir / TABLE_NAME)

    pitch_rows = [
        row
        for row in rows
        if row.get("test_family") == "pitch_disturbance_fddp_reference" and row.get("pass") == "True"
    ]
    contact_row = next(
        row
        for row in rows
        if row.get("test_family") == "contact_switching_mechanism" and row.get("pass") == "True"
    )
    hardware_rows = [
        row
        for row in rows
        if row.get("test_family") == "hardware_runtime_loadability"
    ]
    stage8_path = log_dir / "stage8_scripted_alternating_step_locomotion_summary.yaml"
    stage8_summary = _read_yaml(stage8_path) if stage8_path.exists() else {}
    stage8_metrics = stage8_summary.get("metrics", {})

    existing_figures = [name for name in KEY_FIGURES if (project_root / name).exists()]

    return {
        "summary_path": str(log_dir / SUMMARY_NAME),
        "table_path": str(log_dir / TABLE_NAME),
        "scope_note": summary.get("scope_note", ""),
        "missing_locomotion_metrics": list(summary.get("not_yet_measured_from_current_project_state", [])),
        "pitch_rows": pitch_rows,
        "c2_contact_demo": {
            "matched_events": _as_int(contact_row.get("matched_events")),
            "scheduled_events": _as_int(contact_row.get("scheduled_events")),
            "max_abs_phase_offset_ms": _as_float(contact_row.get("max_abs_phase_offset_ms")),
            "left_contact_was_lost": contact_row.get("left_contact_was_lost") == "True",
            "left_contact_was_restored": contact_row.get("left_contact_was_restored") == "True",
        },
        "hardware_runtime": {
            "rows": hardware_rows,
            "all_loadability_tests_passed": bool(hardware_rows) and all(
                row.get("pass") == "True"
                and row.get("finite_commands") == "True"
                and row.get("joint_limit_violation") == "False"
                for row in hardware_rows
            ),
            "tested_controllers": sorted({row.get("controller", "") for row in hardware_rows if row.get("controller")}),
            "tested_directions": sorted({row.get("direction", "") for row in hardware_rows if row.get("direction")}),
        },
        "stage8_scripted_locomotion": {
            "summary_path": str(stage8_path),
            "available": bool(stage8_summary),
            "scope_note": stage8_summary.get("scope_note", ""),
            "duration_s": float(stage8_metrics.get("duration_s", 0.0)),
            "scripted_step_count": int(stage8_metrics.get("scripted_step_count", 0)),
            "matched_contact_events": int(stage8_metrics.get("matched_contact_events", 0)),
            "scheduled_contact_events": int(stage8_metrics.get("scheduled_contact_events", 0)),
            "pitch_rms_deg": float(stage8_metrics.get("pitch_rms_deg", 0.0)),
            "peak_abs_pitch_deg": float(stage8_metrics.get("peak_abs_pitch_deg", 0.0)),
            "forward_displacement_m": float(stage8_metrics.get("forward_displacement_m", 0.0)),
            "max_abs_phase_offset_ms": float(stage8_metrics.get("max_abs_phase_offset_ms", 0.0)),
            "fall_detected": bool(stage8_metrics.get("fall_detected", False)),
            "acceptance": stage8_summary.get("acceptance", {}),
        },
        "figures": existing_figures,
        "videos": [
            "results/videos/c2_contact_switching_demo.gif",
            "results/videos/fddp_forward_com_shift_replay.gif",
            "results/videos/scripted_alternating_step_preview.gif",
        ],
    }


def _fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def _pitch_table_rows(rows: list[dict[str, str]]) -> str:
    output = []
    for row in rows:
        controller = row["controller"]
        direction = row["direction"]
        iae = _fmt_float(_as_float(row.get("iae_deg_s")), 4)
        improvement = _as_float(row.get("iae_improvement_vs_C0"))
        improvement_text = "--" if improvement is None else f"{100.0 * improvement:.1f}\\%"
        settling = _fmt_float(_as_float(row.get("settling_time_s")), 3)
        max_corr = _fmt_float(_as_float(row.get("max_joint_correction_deg")), 3)
        solve = _fmt_float(_as_float(row.get("median_qp_solve_time_us")), 1)
        output.append(
            f"{controller} & {direction} & {iae} & {improvement_text} & {settling} & {max_corr} & {solve} \\\\"
        )
    return "\n".join(output)


def _figure_items(figures: list[str]) -> str:
    return "\n".join(f"    \\item \\path{{{figure}}}" for figure in figures)


def render_latex(context: dict[str, Any]) -> str:
    """Render a conservative LaTeX update block for the Overleaf thesis."""

    c2 = context["c2_contact_demo"]
    hardware = context["hardware_runtime"]
    stage8 = context["stage8_scripted_locomotion"]
    missing_items = "\n".join(f"    \\item {item}" for item in context["missing_locomotion_metrics"])
    figure_items = _figure_items(context["figures"])
    controllers = ", ".join(hardware["tested_controllers"])
    directions = ", ".join(hardware["tested_directions"])

    return f"""% Generated from reporting/generate_overleaf_evidence_snippets.py.
% Source summary: \\path{{{context['summary_path']}}}
% Source table: \\path{{{context['table_path']}}}

\\section{{C2 Contact-Phase Synchronisation}}
The current C2 layer is no longer only future work. It has been implemented as
a C2 contact-phase mechanism and contact-event phase synchronisation wrapper around the C1 FIR-QP-MPC
correction layer. In the current double-support CoM-shift reference, C2 gives
the same pitch-recovery behaviour as C1 because the reference contains no
scheduled contact transitions. A separate scripted MuJoCo contact-switching
mechanism test validates the event-matching path: C2 matched {c2['matched_events']}/{c2['scheduled_events']} scheduled contact events, with a
maximum absolute phase offset of approximately
\\SI{{{_fmt_float(c2['max_abs_phase_offset_ms'], 1)}}}{{\\milli\\second}}.
This should be described as contact-phase mechanism validation, not as FDDP
walking-gait validation.

\\section{{Hardware Runtime Export}}
The project now includes a MuJoCo-free hardware-side runtime at
\\path{{hardware/controller_runtime/controller_runtime.py}} and an exported
controller package at \\path{{hardware/deploy/controller_package.json}}. The
runtime exposes C0, C1, and C2 as position-command controllers for the six
ST3215 joints. The current loadability check covers controllers {controllers}
in the {directions} directions. All generated commands are finite and no joint
limit violation is reported in the current runtime smoke test. This remains a
software loadability result; it is not yet real hardware validation.

\\section{{Stage 8 Scripted Alternating-Step Locomotion Preview}}
The project now includes a conservative Stage 8 scripted alternating-step
locomotion preview. It is designed to exercise the left/right contact-transition
pipeline before a full FDDP walking reference is available. The reference
contains two scripted swing phases and scheduled left/right liftoff-touchdown
events. In the current MuJoCo run, the preview matched {stage8['matched_contact_events']}/{stage8['scheduled_contact_events']} scheduled contact events,
reported a pitch RMS of \\SI{{{_fmt_float(stage8['pitch_rms_deg'], 3)}}}{{\\degree}},
and produced approximately \\SI{{{1000.0 * stage8['forward_displacement_m']:.1f}}}{{\\milli\\metre}}
of forward displacement over \\SI{{{_fmt_float(stage8['duration_s'], 2)}}}{{\\second}}.
The maximum phase offset was \\SI{{{_fmt_float(stage8['max_abs_phase_offset_ms'], 1)}}}{{\\milli\\second}}.
This result should be reported as scripted alternating-contact locomotion
plumbing, not as final FDDP walking validation.

\\begin{{table}}[htbp]
\\centering
\\caption{{Current C0/C1/C2 pitch-disturbance and runtime evidence.}}
\\label{{tab:c0_c1_c2_current_evidence}}
\\begin{{tabular}}{{@{{}}lllllll@{{}}}}
\\toprule
Controller & Direction & IAE (deg s) & Improvement & Settling (s) & Max corr. (deg) & Median solve (us) \\\\
\\midrule
{_pitch_table_rows(context['pitch_rows'])}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\section{{Current Figure Assets to Import}}
The following generated figures can replace the placeholder boxes currently in
the Overleaf draft:
\\begin{{itemize}}
{figure_items}
\\end{{itemize}}

\\section{{Remaining Metrics Required by the Unchanged Title}}
The title can remain unchanged, but the following metrics are still missing
from the current project state and should either be produced by the walking and
hardware stages or explicitly bounded in the thesis:
\\begin{{itemize}}
{missing_items}
\\end{{itemize}}
"""


def render_manifest(context: dict[str, Any]) -> str:
    figure_lines = "\n".join(context["figures"])
    video_lines = "\n".join(context["videos"])
    return f"""Overleaf evidence update manifest
=================================

Generated from:
{context['summary_path']}
{context['table_path']}

Figures:
{figure_lines}

Videos:
{video_lines}
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "generated")
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    context = build_context(project_root)
    (output_dir / "overleaf_current_evidence_update.tex").write_text(render_latex(context), encoding="utf-8")
    (output_dir / "overleaf_current_evidence_manifest.txt").write_text(render_manifest(context), encoding="utf-8")

    print(f"Wrote {output_dir / 'overleaf_current_evidence_update.tex'}")
    print(f"Wrote {output_dir / 'overleaf_current_evidence_manifest.txt'}")


if __name__ == "__main__":
    main()
