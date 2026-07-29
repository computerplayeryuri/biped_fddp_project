#!/usr/bin/env python3
"""Slide-ready robustness figures built only from recorded project evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from reporting.generate_robustness_ppt_figures import (
    build_scenario_profile,
    load_push_evidence,
    load_scenario_evidence,
    summarise_push_trials,
)
from reporting.regenerate_publication_figures import OKABE_ITO, apply_publication_style


PUSH_STEM = "robustness_bidirectional_push_profile"
SCENARIO_STEM = "robustness_32_scenario_profile"

SCENARIO_LABELS = {
    "nominal": "Nominal",
    "push_minus_2N": "Push -2 N",
    "push_plus_2N": "Push +2 N",
    "push_minus_4N": "Push -4 N",
    "push_plus_4N": "Push +4 N",
    "friction_mu_0p40": "Friction μ = 0.40",
    "friction_mu_0p50": "Friction μ = 0.50",
    "friction_mu_0p80": "Friction μ = 0.80",
    "slope_minus_5deg": "Slope -5°",
    "slope_minus_3deg": "Slope -3°",
    "slope_plus_3deg": "Slope +3°",
    "slope_plus_5deg": "Slope +5°",
    "step_2mm": "Step 2 mm",
    "step_4mm": "Step 4 mm",
    "step_6mm": "Step 6 mm",
    "command_delay_4ms": "Command delay 4 ms",
    "command_delay_12ms": "Command delay 12 ms",
    "command_delay_20ms": "Command delay 20 ms",
    "sensor_delay_0ms": "Sensor delay 0 ms",
    "sensor_delay_20ms": "Sensor delay 20 ms",
    "sensor_delay_40ms": "Sensor delay 40 ms",
    "actuator_gain_minus_20pct": "Actuator gain -20%",
    "actuator_gain_plus_20pct": "Actuator gain +20%",
    "actuator_tau_20ms": "Actuator τ = 20 ms",
    "actuator_tau_30ms": "Actuator τ = 30 ms",
    "actuator_tau_40ms": "Actuator τ = 40 ms",
    "actuator_tau_50ms": "Actuator τ = 50 ms",
    "joint_noise_0p002rad": "Joint noise σ = 0.002 rad",
    "pitch_noise_0p5deg": "Pitch noise σ = 0.5°",
    "combined_A": "Combined A",
    "combined_B": "Combined B",
    "combined_C": "Combined C",
}

CATEGORY_LABELS = {
    "nominal": "NOMINAL",
    "push": "EXTERNAL PUSH",
    "friction": "FRICTION",
    "slope": "SLOPE",
    "step": "STEP HEIGHT",
    "command_delay": "COMMAND DELAY",
    "sensor_delay": "SENSOR DELAY",
    "actuator_gain": "ACTUATOR GAIN",
    "actuator_time_constant": "ACTUATOR DYNAMICS",
    "noise": "SENSOR / STATE NOISE",
    "combined": "COMBINED",
}


def _is_true(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def scenario_display_label(scenario: str) -> str:
    """Return the exact, unit-bearing English label used in the slide figure."""

    try:
        return SCENARIO_LABELS[scenario]
    except KeyError as error:
        raise ValueError(f"No presentation label defined for scenario {scenario!r}") from error


def _save_png_pdf(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def plot_bidirectional_push_profile(
    trials: pd.DataFrame,
    selection: dict,
    output_dir: Path,
) -> tuple[list[Path], dict[str, str]]:
    """Plot every recorded ±4 N trial plus the pitch gate and worst-case margin."""

    pitch_gate_deg = float(selection["limits"]["maximum_pitch_error_deg"])
    summary = summarise_push_trials(trials, pitch_gate_deg=pitch_gate_deg)
    push = selection["push"]
    directions = ["forward", "backward"]
    labels = ["Forward push (+4 N)", "Backward push (-4 N)"]
    colors = [OKABE_ITO["blue"], OKABE_ITO["vermillion"]]
    markers = ["o", "D"]

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(12.0, 6.75), facecolor="white")
    ax.set_facecolor("white")
    for x_position, direction, color, marker in zip(
        range(2), directions, colors, markers, strict=True
    ):
        values = (
            trials.loc[trials["direction"] == direction, "peak_abs_error_deg"]
            .astype(float)
            .sort_values()
            .to_numpy()
        )
        jitter = np.linspace(-0.17, 0.17, len(values))
        ax.scatter(
            x_position + jitter,
            values,
            s=34,
            marker=marker,
            facecolor=color if direction == "backward" else "white",
            edgecolor=color,
            linewidth=1.1,
            alpha=0.92,
            zorder=3,
            label="Individual seeded trials" if direction == "forward" else None,
        )
        row = summary.loc[direction]
        mean = float(row["mean_peak_deg"])
        minimum = float(row["min_peak_deg"])
        maximum = float(row["max_peak_deg"])
        ax.errorbar(
            x_position,
            mean,
            yerr=[[mean - minimum], [maximum - mean]],
            fmt="_",
            markersize=18,
            markeredgewidth=2.4,
            color="black",
            ecolor="black",
            elinewidth=1.4,
            capsize=6,
            capthick=1.4,
            zorder=5,
            label="Mean and observed range" if direction == "forward" else None,
        )
        annotation_y = 11.0 if direction == "forward" else 12.1
        ax.annotate(
            (
                f"{int(row['passed'])}/{int(row['trial_count'])} passed\n"
                f"Mean = {mean:.2f}°\n"
                f"Worst-case margin = {float(row['margin_deg']):.2f}°"
            ),
            xy=(x_position, mean),
            xytext=(x_position + (-0.38 if direction == "forward" else 0.38), annotation_y),
            ha="center",
            va="center",
            fontsize=10,
            color="#222222",
            bbox={"boxstyle": "round,pad=0.45", "fc": "white", "ec": color, "lw": 1.2},
            arrowprops={"arrowstyle": "-", "color": color, "lw": 1.1},
            zorder=6,
        )

    ax.axhline(
        pitch_gate_deg,
        color="#B2182B",
        linewidth=1.6,
        linestyle=(0, (6, 3)),
        label=f"Acceptance gate = {pitch_gate_deg:.0f}°",
        zorder=2,
    )
    ax.text(1.48, pitch_gate_deg + 0.12, "Pitch limit", color="#8F1423", fontsize=9, ha="right")
    ax.set_xlim(-0.65, 1.65)
    ax.set_ylim(0.0, pitch_gate_deg + 1.0)
    ax.set_xticks([0, 1], labels)
    ax.set_ylabel("Peak absolute pitch error (deg)", fontsize=11)
    ax.set_title(
        "Bidirectional Push-Recovery Robustness",
        loc="left",
        fontsize=18,
        weight="bold",
        pad=24,
    )
    fig.text(
        0.125,
        0.91,
        (
            f"{abs(float(push['forces_n'][1])):.0f} N pulse at t = "
            f"{float(push['start_s']):.1f} s for {float(push['duration_s']):.2f} s; "
            "30 seeds per direction"
        ),
        fontsize=10,
        color="#4D4D4D",
    )
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=9)
    fig.text(
        0.125,
        0.025,
        "Scripted MuJoCo/FDDP validation; not a hardware result.",
        fontsize=8.5,
        color="#5A5A5A",
    )
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.15, top=0.82)

    metadata = {
        "forward_passed": f"{int(summary.loc['forward', 'passed'])}/{int(summary.loc['forward', 'trial_count'])}",
        "backward_passed": f"{int(summary.loc['backward', 'passed'])}/{int(summary.loc['backward', 'trial_count'])}",
        "forward_mean_peak_deg": f"{summary.loc['forward', 'mean_peak_deg']:.2f}",
        "backward_mean_peak_deg": f"{summary.loc['backward', 'mean_peak_deg']:.2f}",
        "forward_worst_margin_deg": f"{summary.loc['forward', 'margin_deg']:.2f}",
        "backward_worst_margin_deg": f"{summary.loc['backward', 'margin_deg']:.2f}",
    }
    return _save_png_pdf(fig, Path(output_dir), PUSH_STEM), metadata


def _add_category_bands(axes: tuple[plt.Axes, plt.Axes], block: pd.DataFrame) -> None:
    categories = block["category"].astype(str).tolist()
    y_positions = np.arange(len(block))[::-1]
    start = 0
    band_index = 0
    while start < len(block):
        category = categories[start]
        end = start
        while end + 1 < len(block) and categories[end + 1] == category:
            end += 1
        low = y_positions[end] - 0.48
        high = y_positions[start] + 0.48
        fill = "#F4F6F8" if band_index % 2 == 0 else "#FFFFFF"
        for axis in axes:
            axis.axhspan(low, high, color=fill, zorder=0)
            axis.axhline(low, color="#D9DDE1", linewidth=0.45, zorder=1)
        axes[0].text(
            0.01,
            high - 0.08,
            CATEGORY_LABELS.get(category, category.upper()),
            transform=axes[0].get_yaxis_transform(),
            ha="left",
            va="top",
            fontsize=5.8,
            weight="bold",
            color="#6A6A6A",
            zorder=5,
        )
        start = end + 1
        band_index += 1


def _plot_scenario_block(
    ax_pitch: plt.Axes,
    ax_success: plt.Axes,
    block: pd.DataFrame,
    block_title: str,
) -> None:
    y_positions = np.arange(len(block))[::-1]
    _add_category_bands((ax_pitch, ax_success), block)
    for y, (_, row) in zip(y_positions, block.iterrows(), strict=True):
        critical = _is_true(row["critical"])
        color = OKABE_ITO["vermillion"] if critical else OKABE_ITO["blue"]
        marker = "o" if critical else "s"
        facecolor = color if critical else "white"
        mean_peak = float(row["mean_peak_deg"])
        min_peak = float(row["min_peak_deg"])
        max_peak = float(row["max_peak_deg"])
        ax_pitch.errorbar(
            mean_peak,
            y,
            xerr=[[mean_peak - min_peak], [max_peak - mean_peak]],
            fmt=marker,
            color=color,
            markerfacecolor=facecolor,
            markeredgecolor=color,
            markersize=4.8,
            elinewidth=1.2,
            capsize=2.6,
            zorder=3,
        )

        success = float(row["success_rate"])
        wilson_low = float(row["wilson_low"])
        wilson_high = float(row["wilson_high"])
        ax_success.errorbar(
            success,
            y,
            xerr=[[success - wilson_low], [wilson_high - success]],
            fmt=marker,
            color=color,
            markerfacecolor=facecolor,
            markeredgecolor=color,
            markersize=4.8,
            elinewidth=1.2,
            capsize=2.6,
            zorder=3,
        )

    ax_pitch.axvline(15.0, color="#B2182B", linestyle=(0, (5, 3)), linewidth=1.2, zorder=2)
    ax_pitch.set_xlim(0.0, 15.8)
    ax_pitch.set_xticks([0, 5, 10, 15])
    ax_pitch.set_yticks(y_positions, [scenario_display_label(x) for x in block["scenario"]])
    ax_pitch.tick_params(axis="y", labelsize=7.0, length=0, pad=4)
    ax_pitch.set_xlabel("Peak |pitch error| (deg)", fontsize=8)
    ax_pitch.set_title(block_title, loc="left", fontsize=10.5, weight="bold", pad=9)

    ax_success.axvline(0.90, color="#B2182B", linestyle=(0, (5, 3)), linewidth=1.2, zorder=2)
    ax_success.axvline(0.95, color="#555555", linestyle=(0, (1, 2)), linewidth=1.0, zorder=2)
    ax_success.set_xlim(0.84, 1.012)
    ax_success.set_xticks([0.90, 0.95, 1.00])
    ax_success.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax_success.set_yticks(y_positions)
    ax_success.tick_params(axis="y", labelleft=False, length=0)
    ax_success.set_xlabel("Observed pass rate + Wilson 95% CI", fontsize=8)
    ax_success.set_title("Success evidence", fontsize=9, pad=9)

    for axis in (ax_pitch, ax_success):
        axis.set_ylim(-0.65, len(block) - 0.35)
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.55)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="x", labelsize=7)


def plot_robustness_scenario_profile(
    profile: pd.DataFrame,
    output_dir: Path,
) -> tuple[list[Path], dict[str, str]]:
    """Plot all 32 scenarios as pitch-range and success-uncertainty profiles."""

    if len(profile) != 32:
        raise ValueError(f"Expected 32 scenarios, received {len(profile)}")
    labels = [scenario_display_label(name) for name in profile["scenario"]]
    if len(labels) != len(set(labels)):
        raise ValueError("Presentation scenario labels must be unique")

    apply_publication_style()
    fig = plt.figure(figsize=(16.0, 9.0), facecolor="white")
    outer = fig.add_gridspec(
        1,
        2,
        left=0.055,
        right=0.985,
        bottom=0.13,
        top=0.80,
        wspace=0.22,
    )
    axes: list[plt.Axes] = []
    for block_number, block_slice in enumerate((slice(0, 16), slice(16, 32)), start=1):
        inner = outer[block_number - 1].subgridspec(1, 2, width_ratios=[1.75, 1.0], wspace=0.09)
        ax_pitch = fig.add_subplot(inner[0, 0])
        ax_success = fig.add_subplot(inner[0, 1], sharey=ax_pitch)
        block = profile.iloc[block_slice].reset_index(drop=True)
        _plot_scenario_block(
            ax_pitch,
            ax_success,
            block,
            f"Scenarios {block_slice.start + 1}–{block_slice.stop}",
        )
        axes.extend([ax_pitch, ax_success])

    total_trials = int(profile["trial_count"].sum())
    total_passed = int(profile["passed"].sum())
    overall_rate = total_passed / total_trials
    wilson_low = float(profile["wilson_low"].min())
    most_demanding = profile.loc[profile["max_peak_deg"].astype(float).idxmax()]

    fig.text(
        0.055,
        0.925,
        "Robustness Evidence Across the Full 32-Scenario Campaign",
        fontsize=18,
        weight="bold",
        ha="left",
    )
    fig.text(
        0.055,
        0.875,
        f"32 scenarios × 30 seeds = {total_trials} trials  |  {total_passed}/{total_trials} passed  |  scripted simulation validation",
        fontsize=10.5,
        color="#444444",
        ha="left",
    )
    legend_handles = [
        Line2D([0], [0], marker="o", color=OKABE_ITO["vermillion"], markerfacecolor=OKABE_ITO["vermillion"], linewidth=0, label="Critical scenario"),
        Line2D([0], [0], marker="s", color=OKABE_ITO["blue"], markerfacecolor="white", linewidth=0, label="Intermediate scenario"),
        Line2D([0], [0], color="#B2182B", linestyle=(0, (5, 3)), label="Acceptance gate: 15° / 90% observed"),
        Line2D([0], [0], color="#555555", linestyle=(0, (1, 2)), label="95% campaign target reference"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.925),
        ncol=2,
        frameon=False,
        fontsize=8.2,
    )
    fig.text(
        0.055,
        0.055,
        "Per-scenario acceptance uses observed success ≥ 90%; Wilson intervals show sampling uncertainty (30/30 gives a lower bound of 88.65%).",
        fontsize=8.2,
        color="#4F4F4F",
        ha="left",
    )
    fig.text(
        0.055,
        0.025,
        "All values are from the recorded C2_final MuJoCo campaign; no hardware claim is made.",
        fontsize=8.2,
        color="#4F4F4F",
        ha="left",
    )

    metadata = {
        "scenario_count": str(len(profile)),
        "trial_count": str(total_trials),
        "passed": f"{total_passed}/{total_trials}",
        "success_rate": f"{overall_rate:.1%}",
        "per_scenario_wilson_low": f"{wilson_low:.2%}",
        "most_demanding_scenario": str(most_demanding["scenario"]),
        "most_demanding_label": scenario_display_label(str(most_demanding["scenario"])),
        "most_demanding_max_peak_deg": f"{float(most_demanding['max_peak_deg']):.2f}",
    }
    return _save_png_pdf(fig, Path(output_dir), SCENARIO_STEM), metadata


def generate(
    project_root: Path,
    figure: str = "all",
) -> tuple[list[Path], dict[str, dict[str, str]]]:
    """Generate one or both figures from the project's recorded logs."""

    log_dir = project_root / "results" / "logs"
    output_dir = project_root / "results" / "figures" / "presentation"
    outputs: list[Path] = []
    metadata: dict[str, dict[str, str]] = {}
    if figure in {"push", "all"}:
        push_trials, selection = load_push_evidence(log_dir)
        paths, values = plot_bidirectional_push_profile(push_trials, selection, output_dir)
        outputs.extend(paths)
        metadata["push"] = values
    if figure in {"scenarios", "all"}:
        trials, aggregate, campaign = load_scenario_evidence(log_dir)
        profile = build_scenario_profile(trials, aggregate, campaign)
        paths, values = plot_robustness_scenario_profile(profile, output_dir)
        outputs.extend(paths)
        metadata["scenarios"] = values
    return outputs, metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--figure", choices=("push", "scenarios", "all"), default="all")
    args = parser.parse_args(argv)
    outputs, metadata = generate(args.project_root.resolve(), args.figure)
    print("Generated robustness presentation figures:")
    for path in outputs:
        print(f"  {path}")
    print("Verified metadata:")
    for section, values in metadata.items():
        print(f"  {section}")
        for key, value in values.items():
            print(f"    {key}: {value}")


__all__ = [
    "generate",
    "main",
    "plot_bidirectional_push_profile",
    "plot_robustness_scenario_profile",
    "scenario_display_label",
]
