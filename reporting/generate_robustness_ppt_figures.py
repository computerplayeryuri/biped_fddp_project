#!/usr/bin/env python3
"""Generate slide-ready robustness figures from recorded project evidence.

This module reads existing CSV/YAML artefacts only.  It does not run FDDP,
MuJoCo, or any controller code.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


PUSH_TRIALS_FILE = "fddp_multistep_recovery_sweep.csv"
PUSH_SELECTION_FILE = "fddp_multistep_recovery_selection.yaml"
SCENARIO_TRIALS_FILE = "fddp_multistep_robustness_trials.csv"
SCENARIO_AGGREGATE_FILE = "fddp_multistep_robustness_aggregate.csv"
SCENARIO_SUMMARY_FILE = "fddp_multistep_robustness_summary.yaml"


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} missing required columns: {', '.join(missing)}")


def load_push_evidence(log_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load the selected bidirectional bounded-push campaign."""

    log_dir = Path(log_dir)
    selection = _read_yaml(log_dir / PUSH_SELECTION_FILE)
    trials = _read_csv(log_dir / PUSH_TRIALS_FILE)
    _require_columns(
        trials,
        {"candidate", "direction", "seed", "passed", "peak_abs_error_deg"},
        PUSH_TRIALS_FILE,
    )
    selected = selection.get("selected_candidate")
    if not selected:
        raise ValueError(f"{PUSH_SELECTION_FILE} has no selected_candidate")
    trials = trials.loc[trials["candidate"] == selected].copy()
    if trials.empty:
        raise ValueError(f"No push trials found for selected candidate {selected!r}")
    return trials, selection


def summarise_push_trials(
    trials: pd.DataFrame,
    pitch_gate_deg: float,
) -> pd.DataFrame:
    """Summarise the recorded forward/backward peak-pitch distributions."""

    _require_columns(
        trials,
        {"direction", "passed", "peak_abs_error_deg"},
        "push trials",
    )
    summary = trials.groupby("direction", sort=False).agg(
        trial_count=("passed", "size"),
        passed=("passed", "sum"),
        min_peak_deg=("peak_abs_error_deg", "min"),
        mean_peak_deg=("peak_abs_error_deg", "mean"),
        max_peak_deg=("peak_abs_error_deg", "max"),
    )
    summary["margin_deg"] = float(pitch_gate_deg) - summary["max_peak_deg"]
    preferred_order = [name for name in ("forward", "backward") if name in summary.index]
    remaining = [name for name in summary.index if name not in preferred_order]
    return summary.loc[preferred_order + remaining]


def load_scenario_evidence(
    log_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load raw trials, aggregate statistics, and the ordered campaign registry."""

    log_dir = Path(log_dir)
    trials = _read_csv(log_dir / SCENARIO_TRIALS_FILE)
    aggregate = _read_csv(log_dir / SCENARIO_AGGREGATE_FILE)
    campaign = _read_yaml(log_dir / SCENARIO_SUMMARY_FILE)
    _require_columns(
        trials,
        {"scenario", "category", "critical", "passed", "peak_abs_error_deg"},
        SCENARIO_TRIALS_FILE,
    )
    _require_columns(
        aggregate,
        {
            "scenario",
            "critical",
            "trial_count",
            "success_count",
            "success_rate",
            "wilson_95_low",
            "wilson_95_high",
        },
        SCENARIO_AGGREGATE_FILE,
    )
    if not isinstance(campaign.get("scenarios"), list):
        raise ValueError(f"{SCENARIO_SUMMARY_FILE} has no scenario registry")
    return trials, aggregate, campaign


def build_scenario_profile(
    trials: pd.DataFrame,
    aggregate: pd.DataFrame,
    campaign: dict,
) -> pd.DataFrame:
    """Merge raw pitch ranges with recorded success statistics in registry order."""

    _require_columns(
        trials,
        {"scenario", "category", "critical", "passed", "peak_abs_error_deg"},
        "robustness trials",
    )
    _require_columns(
        aggregate,
        {
            "scenario",
            "critical",
            "trial_count",
            "success_count",
            "success_rate",
            "wilson_95_low",
            "wilson_95_high",
        },
        "robustness aggregate",
    )

    registry = pd.DataFrame(campaign["scenarios"])[["label", "category", "critical"]]
    registry = registry.rename(columns={"label": "scenario", "critical": "registry_critical"})
    registry["scenario_order"] = range(len(registry))

    raw = trials.groupby("scenario", sort=False).agg(
        raw_trial_count=("passed", "size"),
        raw_success_count=("passed", "sum"),
        min_peak_deg=("peak_abs_error_deg", "min"),
        mean_peak_deg=("peak_abs_error_deg", "mean"),
        max_peak_deg=("peak_abs_error_deg", "max"),
    )
    recorded = aggregate.rename(
        columns={
            "success_count": "passed",
            "wilson_95_low": "wilson_low",
            "wilson_95_high": "wilson_high",
        }
    )
    profile = registry.merge(recorded, on="scenario", how="left", validate="one_to_one")
    profile = profile.merge(raw, on="scenario", how="left", validate="one_to_one")

    if profile[["trial_count", "raw_trial_count"]].isna().any().any():
        missing = profile.loc[profile["trial_count"].isna(), "scenario"].tolist()
        raise ValueError(f"Missing aggregate or trial evidence for scenarios: {missing}")
    if not (profile["trial_count"].astype(int) == profile["raw_trial_count"].astype(int)).all():
        raise ValueError("Recorded aggregate trial counts do not match raw trials")
    if not (profile["passed"].astype(int) == profile["raw_success_count"].astype(int)).all():
        raise ValueError("Recorded aggregate success counts do not match raw trials")
    if not (profile["critical"].astype(bool) == profile["registry_critical"].astype(bool)).all():
        raise ValueError("Critical-scenario flags disagree between registry and aggregate")

    return profile.sort_values("scenario_order").reset_index(drop=True)
