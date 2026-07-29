#!/usr/bin/env python3
"""Regenerate thesis figures with publication-quality styling.

The script reads existing result CSV files and redraws the nine figures selected
for the thesis update. It does not run simulation or modify controller logic.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
}

JOINTS = [
    "left_hip_pitch_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
]

CONTACT_FORCES = [
    ("left_toe_force_N", "Left toe", OKABE_ITO["blue"], "-"),
    ("left_heel_force_N", "Left heel", OKABE_ITO["sky_blue"], "--"),
    ("right_toe_force_N", "Right toe", OKABE_ITO["orange"], "-"),
    ("right_heel_force_N", "Right heel", OKABE_ITO["vermillion"], "--"),
]

DISTURBANCE_T_S = 0.400
SETTLING_BAND_DEG = 0.50
TOUCH_THRESHOLD_N = 0.25
FALL_HEIGHT_M = 0.12
FALL_PITCH_DEG = 30.0

C2_DEMO_EVENTS = [
    (0.36, "left liftoff"),
    (1.00, "left touchdown"),
]

STAGE8_EVENTS = [
    (0.36, "left liftoff"),
    (1.00, "left touchdown"),
    (1.30, "right liftoff"),
    (1.95, "right touchdown"),
]

FIGURE_STEMS = [
    "c2_phase_sync_forward_pitch_comparison",
    "c2_phase_sync_backward_pitch_comparison",
    "c2_contact_switching_demo_phase_offset",
    "c2_contact_switching_demo_contacts",
    "stage8_scripted_alternating_step_contacts",
    "stage8_scripted_alternating_step_phase_offset",
    "stage8_scripted_alternating_step_body",
    "stage9_scripted_robustness_matrix_pass_fail",
    "stage9_scripted_robustness_matrix_pitch",
]


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.3,
            "lines.markersize": 3.2,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        color=[
            OKABE_ITO["blue"],
            OKABE_ITO["vermillion"],
            OKABE_ITO["bluish_green"],
            OKABE_ITO["orange"],
            OKABE_ITO["reddish_purple"],
            OKABE_ITO["sky_blue"],
            OKABE_ITO["black"],
        ]
    )


def read_csv(log_dir: Path, name: str) -> pd.DataFrame:
    path = log_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def finish_axes(ax: plt.Axes) -> None:
    ax.tick_params(direction="out")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)


def annotate_events(ax: plt.Axes, events: list[tuple[float, str]], *, y: float = 0.97) -> None:
    for event_t, label in events:
        ax.axvline(event_t, color="0.45", linestyle=(0, (3, 2)), linewidth=0.7, zorder=0)
        ax.text(
            event_t,
            y,
            label,
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="right",
            color="0.30",
            fontsize=6,
        )


def save_figure(
    fig: plt.Figure,
    stem: str,
    figure_dir: Path,
    publication_dir: Path,
    mirror_dir: Path | None,
) -> list[Path]:
    outputs: list[Path] = []
    publication_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    original_png = figure_dir / f"{stem}.png"
    publication_png = publication_dir / f"{stem}.png"
    publication_pdf = publication_dir / f"{stem}.pdf"

    fig.savefig(original_png)
    fig.savefig(publication_png)
    fig.savefig(publication_pdf)
    outputs.extend([original_png, publication_png, publication_pdf])

    if mirror_dir is not None:
        mirror_dir.mkdir(parents=True, exist_ok=True)
        mirror_png = mirror_dir / f"{stem}.png"
        mirror_pdf = mirror_dir / f"{stem}.pdf"
        shutil.copy2(publication_png, mirror_png)
        shutil.copy2(publication_pdf, mirror_pdf)
        outputs.extend([mirror_png, mirror_pdf])

    plt.close(fig)
    return outputs


def plot_pitch_comparison(
    log_dir: Path,
    figure_dir: Path,
    publication_dir: Path,
    mirror_dir: Path | None,
    direction: str,
) -> list[Path]:
    c0 = read_csv(log_dir, f"c1_qp2_c0_{direction}.csv")
    c1 = read_csv(log_dir, f"c1_qp2_balanced_{direction}.csv")
    c2 = read_csv(log_dir, f"c2_phase_sync_{direction}.csv")

    fig, ax = plt.subplots(figsize=(6.8, 2.65), constrained_layout=True)
    ax.plot(c0["time_s"], c0["pitch_error_deg"], color="0.25", linestyle="-", label="C0 reference")
    ax.plot(
        c1["time_s"],
        c1["pitch_error_deg"],
        color=OKABE_ITO["blue"],
        linestyle="--",
        label="C1 FIR-QP-MPC",
    )
    ax.plot(
        c2["wall_time_s"],
        c2["pitch_error_deg"],
        color=OKABE_ITO["vermillion"],
        linestyle="-",
        label="C2 contact-phase sync",
    )
    ax.axvline(DISTURBANCE_T_S, color="0.30", linestyle=(0, (3, 2)), linewidth=0.8, label="disturbance")
    ax.axhspan(-SETTLING_BAND_DEG, SETTLING_BAND_DEG, color="0.92", zorder=-2, label="+/-0.5 deg band")
    ax.axhline(0.0, color="0.20", linewidth=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch error (deg)")
    ax.set_xlim(0.0, max(c0["time_s"].max(), c2["wall_time_s"].max()))
    finish_axes(ax)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.20), handlelength=2.2, columnspacing=1.0)
    return save_figure(
        fig,
        f"c2_phase_sync_{direction}_pitch_comparison",
        figure_dir,
        publication_dir,
        mirror_dir,
    )


def plot_contact_forces(
    df: pd.DataFrame,
    stem: str,
    events: list[tuple[float, str]],
    figure_dir: Path,
    publication_dir: Path,
    mirror_dir: Path | None,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.8, 2.75), constrained_layout=True)
    for column, label, color, linestyle in CONTACT_FORCES:
        ax.plot(df["time_s"], df[column], color=color, linestyle=linestyle, label=label)
    ax.axhline(
        TOUCH_THRESHOLD_N,
        color="0.20",
        linestyle=(0, (1, 2)),
        linewidth=0.8,
        label="touch threshold",
    )
    annotate_events(ax, events)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Touch force (N)")
    ax.set_xlim(0.0, df["time_s"].max())
    ax.set_ylim(bottom=0.0)
    finish_axes(ax)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.20), columnspacing=0.8, handlelength=1.8)
    return save_figure(fig, stem, figure_dir, publication_dir, mirror_dir)


def plot_phase_offset(
    df: pd.DataFrame,
    stem: str,
    events: list[tuple[float, str]],
    figure_dir: Path,
    publication_dir: Path,
    mirror_dir: Path | None,
) -> list[Path]:
    phase_ms = 1000.0 * df["phase_offset_s"]
    ymax = max(55.0, float(np.ceil(np.max(np.abs(phase_ms)) / 10.0) * 10.0 + 10.0))
    fig, ax = plt.subplots(figsize=(6.8, 2.45), constrained_layout=True)
    ax.plot(df["time_s"], phase_ms, color=OKABE_ITO["reddish_purple"], label="C2 phase offset")
    ax.axhline(0.0, color="0.20", linewidth=0.7)
    annotate_events(ax, events)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Phase offset (ms)")
    ax.set_xlim(0.0, df["time_s"].max())
    ax.set_ylim(-ymax, ymax)
    finish_axes(ax)
    ax.legend(loc="upper right")
    return save_figure(fig, stem, figure_dir, publication_dir, mirror_dir)


def plot_stage8_body(
    log_dir: Path,
    figure_dir: Path,
    publication_dir: Path,
    mirror_dir: Path | None,
) -> list[Path]:
    df = read_csv(log_dir, "stage8_scripted_alternating_step_locomotion.csv")
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 4.6), sharex=True, constrained_layout=True)

    axes[0].plot(df["time_s"], np.rad2deg(df["torso_pitch_rad"]), color=OKABE_ITO["blue"])
    axes[0].axhline(0.0, color="0.20", linewidth=0.6)
    axes[0].set_ylabel("Pitch (deg)")

    axes[1].plot(df["time_s"], df["torso_height_m"], color=OKABE_ITO["bluish_green"])
    axes[1].axhline(FALL_HEIGHT_M, color="0.20", linestyle=(0, (1, 2)), linewidth=0.8, label="fall height")
    axes[1].set_ylabel("Height (m)")
    axes[1].legend(loc="lower right")

    axes[2].plot(
        df["time_s"],
        1000.0 * df["torso_forward_displacement_m"],
        color=OKABE_ITO["orange"],
    )
    axes[2].set_ylabel("Forward disp. (mm)")
    axes[2].set_xlabel("Time (s)")

    for ax in axes:
        annotate_events(ax, STAGE8_EVENTS, y=0.98)
        ax.set_xlim(0.0, df["time_s"].max())
        finish_axes(ax)

    return save_figure(
        fig,
        "stage8_scripted_alternating_step_body",
        figure_dir,
        publication_dir,
        mirror_dir,
    )


def pretty_stage9_label(row: pd.Series) -> str:
    category = row["category"]
    if category == "duration":
        return f"Duration {int(round(row['duration_s']))} s"
    if category == "friction":
        return f"Friction mu={row['friction_mu']:.2f}"
    if category == "slope_equivalent":
        return f"Slope {row['slope_deg']:+.0f} deg"
    if category == "small_step":
        return f"Step {1000.0 * row['step_height_m']:.0f} mm"
    if category == "push":
        return f"Push {row['push_force_n']:+.0f} N"
    return str(row["label"])


def stage9_family_label(category: str) -> str:
    labels = {
        "duration": "Duration",
        "friction": "Friction",
        "slope_equivalent": "Slope load",
        "small_step": "Toe step",
        "push": "Push",
    }
    return labels.get(category, category)


def add_stage9_family_bands(ax: plt.Axes, table: pd.DataFrame) -> None:
    categories = table["category"].tolist()
    start = 0
    group_index = 0
    for index in range(1, len(categories) + 1):
        if index == len(categories) or categories[index] != categories[start]:
            end = index - 1
            if group_index % 2 == 0:
                ax.axhspan(start - 0.5, end + 0.5, color="0.96", zorder=-3)
            start = index
            group_index += 1


def plot_stage9_pass_fail(
    log_dir: Path,
    figure_dir: Path,
    publication_dir: Path,
    mirror_dir: Path | None,
) -> list[Path]:
    table = read_csv(log_dir, "stage9_scripted_robustness_matrix_table.csv")
    labels = [pretty_stage9_label(row) for _, row in table.iterrows()]
    passed = table["robustness_pass"].astype(bool).to_numpy()
    colors = np.where(passed, OKABE_ITO["bluish_green"], OKABE_ITO["vermillion"])
    y = np.arange(len(table))

    fig, ax = plt.subplots(figsize=(6.8, 5.1), constrained_layout=True)
    ax.barh(y, np.ones_like(y), color=colors, edgecolor="white", linewidth=0.7)
    for yi, is_pass in zip(y, passed, strict=True):
        ax.text(
            0.50,
            yi,
            "PASS" if is_pass else "FAIL",
            ha="center",
            va="center",
            color="white",
            fontsize=7,
            fontweight="bold",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_xlabel("Scenario outcome")
    ax.invert_yaxis()
    add_stage9_family_bands(ax, table)
    ax.spines["bottom"].set_visible(False)
    return save_figure(
        fig,
        "stage9_scripted_robustness_matrix_pass_fail",
        figure_dir,
        publication_dir,
        mirror_dir,
    )


def plot_stage9_pitch(
    log_dir: Path,
    figure_dir: Path,
    publication_dir: Path,
    mirror_dir: Path | None,
) -> list[Path]:
    table = read_csv(log_dir, "stage9_scripted_robustness_matrix_table.csv")
    labels = [pretty_stage9_label(row) for _, row in table.iterrows()]
    passed = table["robustness_pass"].astype(bool).to_numpy()
    colors = np.where(passed, OKABE_ITO["blue"], OKABE_ITO["vermillion"])
    y = np.arange(len(table))

    fig, ax = plt.subplots(figsize=(6.8, 5.1), constrained_layout=True)
    ax.barh(y, table["peak_abs_pitch_deg"], color=colors, edgecolor="white", linewidth=0.7)
    ax.axvline(
        FALL_PITCH_DEG,
        color="0.20",
        linestyle=(0, (2, 2)),
        linewidth=0.9,
        label="fall threshold",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Peak absolute torso pitch (deg)")
    ax.set_xlim(0.0, 32.0)
    ax.invert_yaxis()
    add_stage9_family_bands(ax, table)
    finish_axes(ax)
    ax.legend(loc="lower right")
    return save_figure(
        fig,
        "stage9_scripted_robustness_matrix_pitch",
        figure_dir,
        publication_dir,
        mirror_dir,
    )


def regenerate(project_root: Path, mirror_dir: Path | None) -> list[Path]:
    apply_publication_style()
    log_dir = project_root / "results" / "logs"
    figure_dir = project_root / "results" / "figures"
    publication_dir = figure_dir / "publication"
    outputs: list[Path] = []

    outputs.extend(plot_pitch_comparison(log_dir, figure_dir, publication_dir, mirror_dir, "forward"))
    outputs.extend(plot_pitch_comparison(log_dir, figure_dir, publication_dir, mirror_dir, "backward"))

    c2_demo = read_csv(log_dir, "c2_contact_switching_demo_phase_sync.csv")
    outputs.extend(
        plot_phase_offset(
            c2_demo,
            "c2_contact_switching_demo_phase_offset",
            C2_DEMO_EVENTS,
            figure_dir,
            publication_dir,
            mirror_dir,
        )
    )
    outputs.extend(
        plot_contact_forces(
            c2_demo,
            "c2_contact_switching_demo_contacts",
            C2_DEMO_EVENTS,
            figure_dir,
            publication_dir,
            mirror_dir,
        )
    )

    stage8 = read_csv(log_dir, "stage8_scripted_alternating_step_locomotion.csv")
    outputs.extend(
        plot_contact_forces(
            stage8,
            "stage8_scripted_alternating_step_contacts",
            STAGE8_EVENTS,
            figure_dir,
            publication_dir,
            mirror_dir,
        )
    )
    outputs.extend(
        plot_phase_offset(
            stage8,
            "stage8_scripted_alternating_step_phase_offset",
            STAGE8_EVENTS,
            figure_dir,
            publication_dir,
            mirror_dir,
        )
    )
    outputs.extend(plot_stage8_body(log_dir, figure_dir, publication_dir, mirror_dir))
    outputs.extend(plot_stage9_pass_fail(log_dir, figure_dir, publication_dir, mirror_dir))
    outputs.extend(plot_stage9_pitch(log_dir, figure_dir, publication_dir, mirror_dir))

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/home/zhijian/biped_fddp_project"),
        help="Path to the biped_fddp_project root.",
    )
    parser.add_argument(
        "--mirror-dir",
        type=Path,
        default=None,
        help="Optional directory that receives PNG and PDF copies.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = regenerate(args.project_root.resolve(), args.mirror_dir.resolve() if args.mirror_dir else None)
    print("Generated publication figures:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
