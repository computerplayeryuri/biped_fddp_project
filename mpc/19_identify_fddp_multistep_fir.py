#!/usr/bin/env python3
"""Retarget the existing two-step FIR/AR identification to the final ten-step reference."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REUSE_SOURCE_PATH = ROOT / "mpc" / "16_validate_fddp_two_step_c1_fir_qp.py"
REPLAY_SOURCE_PATH = ROOT / "simulation" / "03u_replay_fddp_multistep_mujoco.py"
FINAL_REFERENCE_PATH = (
    ROOT / "transfer" / "trajectories" / "fddp_multistep_10step_ref.npz"
)
BASELINE_CSV_PATH = (
    ROOT / "results" / "logs" / "fddp_multistep_c1_identification_baseline.csv"
)
FIR_YAML_PATH = (
    ROOT / "results" / "logs" / "fddp_multistep_c1_fir_identification.yaml"
)
DISTURBANCE_CSV_PATH = (
    ROOT / "results" / "logs" / "fddp_multistep_c1_c0_pitch_disturbance.csv"
)
AR_YAML_PATH = (
    ROOT / "results" / "logs" / "fddp_multistep_c1_ar_fit.yaml"
)
ACTUATOR_CONFIG_PATH = ROOT / "config" / "actuator_model.yaml"
CONTROL_PERIOD_S = float(
    yaml.safe_load(ACTUATOR_CONFIG_PATH.read_text())["control_timestep_s"]
)


def _load_existing(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load existing module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


replay = _load_existing(REPLAY_SOURCE_PATH, "multistep_identification_replay")
COMMAND_REFERENCE_ADVANCE_S = float(replay.COMMAND_REFERENCE_ADVANCE_S)


def configure_reused_identification():
    """Point the existing mpc/16 implementation at ten-step inputs/outputs."""

    module = _load_existing(
        REUSE_SOURCE_PATH, "multistep_reused_identification"
    )
    module.NPZ_PATH = FINAL_REFERENCE_PATH
    module.BASELINE_CSV = BASELINE_CSV_PATH
    module.FIR_YAML = FIR_YAML_PATH
    module.DISTURBANCE_CSV = DISTURBANCE_CSV_PATH
    module.AR_YAML = AR_YAML_PATH
    module.COMMAND_REFERENCE_ADVANCE_S = COMMAND_REFERENCE_ADVANCE_S
    module.FIR_SAMPLE_S = CONTROL_PERIOD_S
    module.PULSE_WIDTH_S = CONTROL_PERIOD_S
    return module


def _retag(path: Path, stage: str) -> None:
    report = yaml.safe_load(path.read_text(encoding="utf-8"))
    report["stage"] = stage
    report["reused_implementation"] = str(REUSE_SOURCE_PATH)
    report["command_reference_advance_s"] = COMMAND_REFERENCE_ADVANCE_S
    path.write_text(
        yaml.safe_dump(report, sort_keys=False), encoding="utf-8"
    )


def main() -> None:
    module = configure_reused_identification()
    module.run_baseline()
    module.run_pulse_identification()
    module.run_disturbed_c0()
    module.fit_ar_model()
    _retag(FIR_YAML_PATH, "multistep_fddp_c1_pulse_fir_identification")
    _retag(AR_YAML_PATH, "multistep_fddp_c1_ar_fit")
    print(yaml.safe_dump({
        "stage": "multistep_fddp_c1_identification",
        "reused_implementation": str(REUSE_SOURCE_PATH),
        "reference_npz": str(FINAL_REFERENCE_PATH),
        "command_reference_advance_s": COMMAND_REFERENCE_ADVANCE_S,
        "outputs": {
            "baseline_csv": str(BASELINE_CSV_PATH),
            "fir_yaml": str(FIR_YAML_PATH),
            "disturbance_csv": str(DISTURBANCE_CSV_PATH),
            "ar_yaml": str(AR_YAML_PATH),
        },
    }, sort_keys=False))


if __name__ == "__main__":
    main()
