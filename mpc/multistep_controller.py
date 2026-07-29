#!/usr/bin/env python3
"""Thin shared controller adapter for the accepted ten-step FDDP reference.

The module composes the existing hardware runtime FIR-QP, safety limiter, and
contact-phase synchroniser. It remains MuJoCo-free and outputs joint positions;
tau_ref is validated by the strict replay loader but is never exposed here.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from hardware.controller_runtime import controller_runtime as runtime


FINAL_REFERENCE_PATH = (
    ROOT / "transfer" / "trajectories" / "fddp_multistep_10step_ref.npz"
)
EXPERIMENT_CONFIG_PATH = ROOT / "config" / "fddp_multistep_experiment.yaml"
REPLAY_MODULE_PATH = ROOT / "simulation" / "03u_replay_fddp_multistep_mujoco.py"
PHASE_MODULE_PATH = ROOT / "mpc" / "08_run_contact_phase_synchronised_fir_qp_mpc.py"
BASELINE_CSV_PATH = ROOT / "results" / "logs" / "fddp_multistep_nominal_replay.csv"
BASELINE_YAML_PATH = ROOT / "results" / "logs" / "fddp_multistep_nominal_replay.yaml"
FIR_YAML_PATH = ROOT / "results" / "logs" / "fddp_multistep_c1_fir_identification.yaml"
AR_YAML_PATH = ROOT / "results" / "logs" / "fddp_multistep_c1_ar_fit.yaml"

SCHEMA_VERSION = "fddp-controller-v1"
MODES = ("C0", "C1", "phase_only", "C2_final")

def project_alpha_to_joint_limits(
    alpha_rad: float,
    q_ref: np.ndarray,
    joint_low: np.ndarray,
    joint_high: np.ndarray,
) -> float:
    """Project scalar FIR correction into the current joint-feasible interval."""
    q_ref = np.asarray(q_ref, dtype=float).reshape(6)
    joint_low = np.asarray(joint_low, dtype=float).reshape(6)
    joint_high = np.asarray(joint_high, dtype=float).reshape(6)
    values = np.concatenate([
        q_ref, joint_low, joint_high, [float(alpha_rad)]
    ])
    if not np.all(np.isfinite(values)):
        raise ValueError("alpha projection inputs must be finite")
    if np.any(joint_low > joint_high):
        raise ValueError("invalid joint limits")
    lower = -np.inf
    upper = np.inf
    for q_value, gain, low, high in zip(
        q_ref, runtime.G, joint_low, joint_high
    ):
        if abs(float(gain)) <= 1.0e-15:
            if q_value < low or q_value > high:
                raise ValueError("reference violates an unactuated joint limit")
            continue
        first = (low - q_value) / gain
        second = (high - q_value) / gain
        lower = max(lower, min(first, second))
        upper = min(upper, max(first, second))
    if lower > upper:
        raise ValueError("no feasible scalar correction interval")
    return float(np.clip(float(alpha_rad), lower, upper))



def _load_existing(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load existing module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


replay = _load_existing(REPLAY_MODULE_PATH, "multistep_controller_replay")
phase_runtime = _load_existing(PHASE_MODULE_PATH, "multistep_controller_phase")
ReferenceIntegrityError = replay.ReferenceIntegrityError


@dataclass(frozen=True)
class ControllerInput:
    wall_time_s: float
    root_pitch_rad: float
    root_pitch_rate_rad_s: float = 0.0
    left_toe_contact: bool = True
    left_heel_contact: bool = True
    right_toe_contact: bool = True
    right_heel_contact: bool = True
    deadline_missed: bool = False


@dataclass(frozen=True)
class ControllerOutput:
    q_cmd: np.ndarray
    q_ref: np.ndarray
    delta_q: np.ndarray
    wall_time_s: float
    reference_time_s: float
    command_reference_time_s: float
    pitch_error_rad: float
    alpha_rad: float
    phase_offset_s: float
    matched_events: int
    skipped_events: int
    limiter_active: bool
    saturated: bool
    deadline_missed: bool
    fallback_active: bool
    controller_state: str
    mode: str
    schema_version: str = SCHEMA_VERSION


class MultiStepReferenceBundle:
    """Strict ten-step reference plus its accepted MuJoCo pitch baseline."""

    def __init__(
        self,
        reference_path: Path = FINAL_REFERENCE_PATH,
        baseline_csv_path: Path = BASELINE_CSV_PATH,
        baseline_yaml_path: Path = BASELINE_YAML_PATH,
    ):
        self.reference = replay.load_multistep_reference(Path(reference_path))
        baseline_summary = yaml.safe_load(
            Path(baseline_yaml_path).read_text(encoding="utf-8")
        )
        if not bool(baseline_summary.get("integrity_passed", False)):
            raise ReferenceIntegrityError(
                f"multi-step baseline is not accepted: {baseline_yaml_path}"
            )
        baseline = np.genfromtxt(
            baseline_csv_path, delimiter=",", names=True, dtype=None, encoding=None
        )
        self.baseline_t = np.asarray(baseline["reference_time_s"], dtype=float)
        self.baseline_pitch = np.asarray(baseline["torso_pitch_rad"], dtype=float)
        if (
            self.baseline_t.ndim != 1
            or self.baseline_t.size < 2
            or self.baseline_pitch.shape != self.baseline_t.shape
            or not np.all(np.isfinite(self.baseline_t))
            or not np.all(np.isfinite(self.baseline_pitch))
        ):
            raise ReferenceIntegrityError("invalid accepted multi-step pitch baseline")

    @property
    def duration_s(self) -> float:
        return self.reference.duration_s

    def q_at(self, time_s: float) -> np.ndarray:
        time_s = float(np.clip(time_s, 0.0, self.duration_s))
        return np.asarray(
            [
                np.interp(time_s, self.reference.t_ref, self.reference.q_ref[:, i])
                for i in range(6)
            ],
            dtype=float,
        )

    def pitch_ref_at(self, time_s: float) -> float:
        time_s = float(np.clip(time_s, self.baseline_t[0], self.baseline_t[-1]))
        return float(np.interp(time_s, self.baseline_t, self.baseline_pitch))

    def contact_events(self) -> list[dict[str, Any]]:
        return replay.build_reference_events(self.reference)


class MultiStepController:
    """One stateful interface for C0, C1, phase-only, and integrated C2-final."""

    def __init__(
        self,
        reference_path: Path = FINAL_REFERENCE_PATH,
        fir_yaml_path: Path = FIR_YAML_PATH,
        ar_yaml_path: Path = AR_YAML_PATH,
        fir_qp_candidate: dict[str, float] | None = None,
        recovery_fir_qp_candidate: dict[str, float] | None = None,
        phase_sync: dict[str, float] | None = None,
        recovery_activation_pitch_deg: float | None = None,
    ):
        self.bundle = MultiStepReferenceBundle(reference_path=reference_path)
        self.reference = self.bundle.reference
        fir = yaml.safe_load(Path(fir_yaml_path).read_text(encoding="utf-8"))
        ar = yaml.safe_load(Path(ar_yaml_path).read_text(encoding="utf-8"))
        h = np.asarray(fir["fir_coefficients_rad_per_rad"], dtype=float)
        a = np.asarray(ar["coefficients"], dtype=float)
        if h.ndim != 1 or a.ndim != 1 or h.size < 2 or a.size < 1:
            raise ValueError("FIR/AR artifacts must contain one-dimensional models")
        if not np.all(np.isfinite(h)) or not np.all(np.isfinite(a)):
            raise ValueError("FIR/AR artifacts contain non-finite coefficients")

        controller_config = yaml.safe_load(
            EXPERIMENT_CONFIG_PATH.read_text(encoding="utf-8")
        )["controller"]
        candidate = dict(controller_config["fir_qp_candidate"])
        if fir_qp_candidate is not None:
            candidate.update(dict(fir_qp_candidate))
        for key in ("q", "r", "slew"):
            value = float(candidate[key])
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"invalid FIR-QP recovery parameter {key}={value}")
            candidate[key] = value

        recovery_candidate = dict(
            controller_config.get("recovery_fir_qp_candidate", {})
        )
        if recovery_fir_qp_candidate is not None:
            recovery_candidate.update(dict(recovery_fir_qp_candidate))
        for key in ("q", "r", "slew"):
            if not recovery_candidate:
                break
            value = float(recovery_candidate[key])
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"invalid scheduled recovery FIR-QP parameter {key}={value}"
                )
            recovery_candidate[key] = value
        if recovery_candidate:
            activation_deg = (
                controller_config.get("recovery_activation_pitch_deg")
                if recovery_activation_pitch_deg is None
                else recovery_activation_pitch_deg
            )
            if activation_deg is None:
                raise ValueError(
                    "recovery_activation_pitch_deg is required when a recovery "
                    "FIR-QP candidate is configured"
                )
            self.recovery_activation_pitch_deg = float(activation_deg)
            if (
                not np.isfinite(self.recovery_activation_pitch_deg)
                or self.recovery_activation_pitch_deg <= 0.0
            ):
                raise ValueError(
                    "recovery_activation_pitch_deg must be finite and positive"
                )
        else:
            self.recovery_activation_pitch_deg = float("inf")

        sync_kwargs = {
            "max_offset_s": controller_config["phase_sync_max_offset_s"],
            "match_window_s": controller_config["phase_sync_match_window_s"],
            "max_step_s": controller_config["phase_sync_max_step_s"],
            "correction_gain": controller_config["phase_sync_correction_gain"],
        }
        if phase_sync is not None:
            unknown = set(phase_sync) - set(sync_kwargs)
            if unknown:
                raise ValueError(f"unknown phase-sync recovery parameters: {sorted(unknown)}")
            sync_kwargs.update(
                {key: float(value) for key, value in phase_sync.items()}
            )
        if (
            sync_kwargs["max_offset_s"] < 0.0
            or sync_kwargs["match_window_s"] <= 0.0
            or sync_kwargs["max_step_s"] < 0.0
            or not 0.0 <= sync_kwargs["correction_gain"] <= 1.0
        ):
            raise ValueError("invalid phase-sync recovery parameters")
        self.controller_configuration = {
            "fir_qp_candidate": dict(candidate),
            "recovery_fir_qp_candidate": dict(recovery_candidate),
            "recovery_activation_pitch_deg": self.recovery_activation_pitch_deg,
            "phase_sync": dict(sync_kwargs),
        }
        self.command_reference_advance_s = float(
            replay.COMMAND_REFERENCE_ADVANCE_S
        )
        self._qps = {
            "C1": runtime.FIRQP(h, a, candidate),
            "C2_final": runtime.FIRQP(h, a, candidate),
        }
        self._recovery_qps = (
            {
                "C1": runtime.FIRQP(h, a, recovery_candidate),
                "C2_final": runtime.FIRQP(h, a, recovery_candidate),
            }
            if recovery_candidate
            else {}
        )
        events = self.bundle.contact_events()
        self._synchronizers = {
            "phase_only": phase_runtime.ContactPhaseSynchronizer(
                events, predictive=True, **sync_kwargs
            ),
            "C2_final": phase_runtime.ContactPhaseSynchronizer(
                events, predictive=True, **sync_kwargs
            ),
        }
        initial_q = self.bundle.q_at(self.command_reference_advance_s)
        self._limiters = {
            mode: runtime.SafetyLimiter(initial_q.copy()) for mode in MODES
        }
        self._last_q_ref = {
            mode: initial_q.copy() for mode in MODES
        }
        self._last_contacts = {
            mode: {"left": True, "right": True}
            for mode in ("phase_only", "C2_final")
        }

    @staticmethod
    def _leg_contacts(sample: ControllerInput) -> dict[str, bool]:
        return {
            "left": bool(sample.left_toe_contact or sample.left_heel_contact),
            "right": bool(sample.right_toe_contact or sample.right_heel_contact),
        }

    def _observed_events(
        self, sample: ControllerInput, mode: str
    ) -> list[dict[str, Any]]:
        current = self._leg_contacts(sample)
        previous = self._last_contacts[mode]
        events = []
        for side in ("left", "right"):
            if current[side] != previous[side]:
                events.append(
                    {
                        "time_s": float(sample.wall_time_s),
                        "kind": (
                            f"{side}_{'touchdown' if current[side] else 'liftoff'}"
                        ),
                    }
                )
        self._last_contacts[mode] = current
        return events

    def update(self, sample: ControllerInput, mode: str) -> ControllerOutput:
        if mode not in MODES:
            raise ValueError(f"invalid controller mode {mode!r}; expected one of {MODES}")
        values = (
            sample.wall_time_s,
            sample.root_pitch_rad,
            sample.root_pitch_rate_rad_s,
        )
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError("controller input contains non-finite values")

        wall_time_s = float(np.clip(sample.wall_time_s, 0.0, self.bundle.duration_s))
        phase_offset_s = 0.0
        matched_events = 0
        skipped_events = 0
        reference_time_s = wall_time_s
        motion_reference_time_s = wall_time_s
        if mode in self._synchronizers:
            synchronizer = self._synchronizers[mode]
            synchronizer.update(
                wall_time_s,
                self._observed_events(sample, mode),
            )
            motion_reference_time_s = synchronizer.reference_time(
                wall_time_s, self.bundle.duration_s
            )
            phase_offset_s = float(synchronizer.offset_s)
            matched_events = int(len(synchronizer.matched))
            skipped_events = int(len(synchronizer.skipped))

        base_command_reference_time_s = float(
            np.clip(
                wall_time_s + self.command_reference_advance_s,
                0.0,
                self.bundle.duration_s,
            )
        )
        command_reference_time_s = float(
            np.clip(
                motion_reference_time_s + self.command_reference_advance_s,
                0.0,
                self.bundle.duration_s,
            )
        )
        q_ref = self.bundle.q_at(base_command_reference_time_s)
        if mode in self._synchronizers:
            synchronizer = self._synchronizers[mode]
            if synchronizer.next_index < len(synchronizer.reference_events):
                event_kind = synchronizer.reference_events[
                    synchronizer.next_index
                ]["kind"]
                phase_q_ref = self.bundle.q_at(command_reference_time_s)
                if event_kind.startswith("left_"):
                    q_ref[:3] = phase_q_ref[:3]
                elif event_kind.startswith("right_"):
                    q_ref[3:] = phase_q_ref[3:]
        pitch_error_rad = float(
            sample.root_pitch_rad - self.bundle.pitch_ref_at(reference_time_s)
        )
        alpha_rad = 0.0
        controller_state = "normal"
        if mode in self._qps:
            alpha_rad = float(self._qps[mode].update(pitch_error_rad))
            if mode in self._recovery_qps:
                recovery_alpha_rad = float(
                    self._recovery_qps[mode].update(pitch_error_rad)
                )
                if (
                    abs(np.rad2deg(pitch_error_rad))
                    >= self.recovery_activation_pitch_deg
                ):
                    alpha_rad = recovery_alpha_rad
                    controller_state = "recovery"
        limiter = self._limiters[mode]
        alpha_rad = project_alpha_to_joint_limits(
            alpha_rad,
            q_ref,
            limiter.low,
            limiter.high,
        )
        raw_command = q_ref + runtime.G * alpha_rad
        limiter.prev = (
            limiter.prev + q_ref - self._last_q_ref[mode]
        )
        q_cmd, safety = limiter.apply(raw_command)
        self._last_q_ref[mode] = q_ref.copy()
        delta_q = np.asarray(q_cmd - q_ref, dtype=float)

        return ControllerOutput(
            q_cmd=np.asarray(q_cmd, dtype=float),
            q_ref=np.asarray(q_ref, dtype=float),
            delta_q=delta_q,
            wall_time_s=wall_time_s,
            reference_time_s=float(reference_time_s),
            command_reference_time_s=command_reference_time_s,
            pitch_error_rad=pitch_error_rad,
            alpha_rad=alpha_rad,
            phase_offset_s=phase_offset_s,
            matched_events=matched_events,
            skipped_events=skipped_events,
            limiter_active=bool(
                safety["rate_limit_active"] or safety["joint_limit_violation"]
            ),
            saturated=bool(safety["joint_limit_violation"]),
            deadline_missed=bool(sample.deadline_missed),
            fallback_active=False,
            controller_state=controller_state,
            mode=mode,
        )
