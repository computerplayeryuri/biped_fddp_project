#!/usr/bin/env python3
"""Deterministic safety wrapper for the multi-step position controller.

This module reuses the project's hardware SafetyLimiter. It contains no MuJoCo
dependency and never creates torque commands.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware.controller_runtime.controller_runtime import SafetyLimiter


class RuntimeState(Enum):
    NORMAL = auto()
    DEGRADED = auto()
    FALLBACK = auto()
    SAFE_HOLD = auto()
    FAILED = auto()


class Fault(Enum):
    STALE_MEASUREMENT = auto()
    NAN_MEASUREMENT = auto()
    INF_MEASUREMENT = auto()
    STUCK_CONTACT = auto()
    MISSING_CONTACT_EVENT = auto()
    OUT_OF_ORDER_CONTACT_EVENT = auto()
    INVALID_QP = auto()
    INVALID_COMMAND = auto()
    DEADLINE_MISS = auto()
    COMMAND_SATURATION = auto()
    RATE_SATURATION = auto()
    ACTUATOR_DELAY = auto()
    MISSING_REFERENCE = auto()
    TRUNCATED_REFERENCE = auto()
    INVALID_REFERENCE_SCHEMA = auto()


STARTUP_FAULTS = {
    Fault.MISSING_REFERENCE,
    Fault.TRUNCATED_REFERENCE,
    Fault.INVALID_REFERENCE_SCHEMA,
}
SAFE_HOLD_FAULTS = {
    Fault.NAN_MEASUREMENT,
    Fault.INF_MEASUREMENT,
    Fault.INVALID_COMMAND,
}
FALLBACK_FAULTS = {Fault.INVALID_QP}


@dataclass(frozen=True)
class SafeControllerResult:
    q_cmd: np.ndarray
    state: RuntimeState
    faults: tuple[Fault, ...]
    fallback_active: bool
    controller_output: Any | None


class RuntimeSafetyMachine:
    """Pure fault classification plus deterministic transition logging."""

    def __init__(self, q_hold: np.ndarray):
        q_hold = np.asarray(q_hold, dtype=float).reshape(6)
        if not np.all(np.isfinite(q_hold)):
            raise ValueError("q_hold must contain six finite joint positions")
        self.state = RuntimeState.NORMAL
        self.transitions: list[dict[str, Any]] = []
        self._last_wall_time_s: float | None = None
        self._limiter = SafetyLimiter(q_hold.copy())
        self.max_step_rad = float(self._limiter.max_step)

    def validate_input(self, sample: Any) -> list[Fault]:
        faults: list[Fault] = []
        values = {
            "wall_time_s": float(getattr(sample, "wall_time_s", np.nan)),
            "root_pitch_rad": float(getattr(sample, "root_pitch_rad", np.nan)),
            "root_pitch_rate_rad_s": float(
                getattr(sample, "root_pitch_rate_rad_s", np.nan)
            ),
        }
        array = np.asarray(list(values.values()), dtype=float)
        if np.any(np.isnan(array)):
            faults.append(Fault.NAN_MEASUREMENT)
        if np.any(np.isinf(array)):
            faults.append(Fault.INF_MEASUREMENT)
        wall_time_s = values["wall_time_s"]
        if np.isfinite(wall_time_s):
            if (
                self._last_wall_time_s is not None
                and wall_time_s <= self._last_wall_time_s
            ):
                faults.append(Fault.STALE_MEASUREMENT)
            else:
                self._last_wall_time_s = wall_time_s
        return faults

    def validate_output(self, output: Any) -> list[Fault]:
        faults: list[Fault] = []
        try:
            q_cmd = np.asarray(output.q_cmd, dtype=float).reshape(6)
        except (AttributeError, TypeError, ValueError):
            q_cmd = np.full(6, np.nan)
        if not np.all(np.isfinite(q_cmd)):
            faults.append(Fault.INVALID_COMMAND)
        if bool(getattr(output, "saturated", False)):
            faults.append(Fault.COMMAND_SATURATION)
        if bool(getattr(output, "limiter_active", False)):
            faults.append(Fault.RATE_SATURATION)
        if bool(getattr(output, "deadline_missed", False)):
            faults.append(Fault.DEADLINE_MISS)
        return faults

    def handle(
        self,
        fault: Fault,
        now_s: float,
        recovery_outcome: str | None = None,
    ) -> RuntimeState:
        if not isinstance(fault, Fault):
            raise TypeError("fault must be a Fault enum value")
        previous = self.state
        if previous is RuntimeState.FAILED:
            next_state = RuntimeState.FAILED
        elif fault in STARTUP_FAULTS:
            next_state = RuntimeState.FAILED
        elif fault in SAFE_HOLD_FAULTS:
            next_state = RuntimeState.SAFE_HOLD
        elif fault in FALLBACK_FAULTS:
            next_state = RuntimeState.FALLBACK
        elif previous is RuntimeState.NORMAL:
            next_state = RuntimeState.DEGRADED
        elif previous is RuntimeState.DEGRADED:
            next_state = RuntimeState.FALLBACK
        elif previous is RuntimeState.FALLBACK:
            next_state = RuntimeState.SAFE_HOLD
        else:
            next_state = previous
        self.state = next_state
        if recovery_outcome is None:
            if next_state is RuntimeState.FAILED:
                recovery_outcome = "startup_rejected"
            elif next_state is RuntimeState.SAFE_HOLD:
                recovery_outcome = "safe_hold_command_required"
            elif next_state is RuntimeState.FALLBACK:
                recovery_outcome = "c0_fallback_required"
            else:
                recovery_outcome = "fault_contained"
        self.transitions.append(
            {
                "time_s": float(now_s),
                "reason": fault.name,
                "previous_state": previous.name,
                "next_state": next_state.name,
                "recovery_outcome": str(recovery_outcome),
            }
        )
        return self.state

    def safe_command(
        self,
        q_last_valid: np.ndarray,
        q_hold: np.ndarray,
        dt_s: float,
    ) -> np.ndarray:
        q_last_valid = np.asarray(q_last_valid, dtype=float).reshape(6)
        q_hold = np.asarray(q_hold, dtype=float).reshape(6)
        if not np.all(np.isfinite(q_last_valid)) or not np.all(np.isfinite(q_hold)):
            raise ValueError("safe command inputs must be finite")
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        self._limiter.prev = q_last_valid.copy()
        command, _ = self._limiter.apply(q_hold)
        return np.asarray(command, dtype=float)


class SafeControllerAdapter:
    """Catch controller faults and select bounded C0 fallback or safe hold."""

    def __init__(self, controller: Any, q_hold: np.ndarray):
        self.controller = controller
        self.q_hold = np.asarray(q_hold, dtype=float).reshape(6)
        self.machine = RuntimeSafetyMachine(self.q_hold)
        self._command_limiter = SafetyLimiter(self.q_hold.copy())
        self.last_valid_q = self.q_hold.copy()

    def _bounded(self, q_cmd: np.ndarray) -> np.ndarray:
        self._command_limiter.prev = self.last_valid_q.copy()
        command, _ = self._command_limiter.apply(q_cmd)
        command = np.asarray(command, dtype=float)
        if not np.all(np.isfinite(command)):
            raise ValueError("bounded command is non-finite")
        return command

    def _safe_result(
        self,
        sample: Any,
        faults: list[Fault],
        controller_output: Any | None = None,
    ) -> SafeControllerResult:
        dt_s = 0.020
        command = self.machine.safe_command(
            self.last_valid_q, self.q_hold, dt_s=dt_s
        )
        self.last_valid_q = command.copy()
        return SafeControllerResult(
            q_cmd=command,
            state=self.machine.state,
            faults=tuple(faults),
            fallback_active=True,
            controller_output=controller_output,
        )

    def update(self, sample: Any, mode: str) -> SafeControllerResult:
        now_s = float(getattr(sample, "wall_time_s", 0.0))
        faults = self.machine.validate_input(sample)
        for fault in faults:
            self.machine.handle(fault, now_s)
        if self.machine.state in {RuntimeState.SAFE_HOLD, RuntimeState.FAILED}:
            return self._safe_result(sample, faults)

        fallback_active = False
        try:
            output = self.controller.update(sample, mode)
        except Exception:
            fault = Fault.INVALID_QP
            faults.append(fault)
            self.machine.handle(fault, now_s)
            fallback_active = True
            try:
                output = self.controller.update(sample, "C0")
            except Exception:
                invalid = Fault.INVALID_COMMAND
                faults.append(invalid)
                self.machine.handle(invalid, now_s)
                return self._safe_result(sample, faults)

        output_faults = self.machine.validate_output(output)
        faults.extend(output_faults)
        for fault in output_faults:
            self.machine.handle(fault, now_s)
        if Fault.INVALID_COMMAND in output_faults:
            return self._safe_result(sample, faults, controller_output=output)

        command = self._bounded(np.asarray(output.q_cmd, dtype=float))
        self.last_valid_q = command.copy()
        return SafeControllerResult(
            q_cmd=command,
            state=self.machine.state,
            faults=tuple(faults),
            fallback_active=fallback_active,
            controller_output=output,
        )
