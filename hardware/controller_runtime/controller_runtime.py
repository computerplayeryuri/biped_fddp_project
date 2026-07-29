#!/usr/bin/env python3
"""Hardware-loadable controller runtime for C0/C1/C2 joint-position commands.

The runtime is intentionally MuJoCo-free. It loads exported FDDP references and
C1 FIR identification results, then returns six joint-position commands in rad.
It is suitable for ROS 2 / PC-side hardware tests that send ST3215 position
commands. It does not send torque commands.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[2]
TRAJ = PROJECT / "transfer" / "trajectories"
LOG = PROJECT / "results" / "logs"
CONFIG = PROJECT / "config"

JOINT_ORDER = [
    "left_hip_pitch_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
]

G = np.array([0.27, 0.062, -0.228, 0.27, 0.062, -0.228], dtype=float)
TS = 0.020
N = 12
AR_ORDER = 4
QP_ITERS = 120
ALPHA_LIMIT_DEG = 5.0
ALPHA_SLEW_DEG = 1.0
DEFAULT_BALANCED = {"name": "balanced", "q": 1500.0, "r": 20.0, "slew": 25.0}


@dataclass
class ControllerMeasurement:
    time_s: float
    root_pitch_rad: float = 0.0
    root_pitch_rate_rad_s: float = 0.0
    left_toe_contact: bool = True
    left_heel_contact: bool = True
    right_toe_contact: bool = True
    right_heel_contact: bool = True


class SafetyLimiter:
    def __init__(self, initial_q: np.ndarray):
        limits = yaml.safe_load((CONFIG / "robot_limits.yaml").read_text())
        actuator = yaml.safe_load((CONFIG / "actuator_model.yaml").read_text())
        self.low = np.array([limits["joint_limits_rad"][name][0] for name in JOINT_ORDER], dtype=float)
        self.high = np.array([limits["joint_limits_rad"][name][1] for name in JOINT_ORDER], dtype=float)
        self.safety_margin = float(limits.get("safety_margin_rad", 0.0))
        self.max_step = float(actuator["max_reference_change_per_cycle_rad"])
        self.safe_speed = float(actuator["safe_speed_rad_s"])
        self.prev = np.asarray(initial_q, dtype=float).reshape(6)

    def apply(self, q_cmd: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        q_cmd = np.asarray(q_cmd, dtype=float).reshape(6)
        clipped_limit = np.clip(q_cmd, self.low, self.high)
        rate_limited = self.prev + np.clip(clipped_limit - self.prev, -self.max_step, self.max_step)
        saturated = np.abs(clipped_limit - q_cmd) > 1e-12
        rate_hit = np.abs(rate_limited - clipped_limit) > 1e-12
        self.prev = rate_limited.copy()
        return rate_limited, {
            "joint_limit_violation": bool(np.any(saturated)),
            "rate_limit_active": bool(np.any(rate_hit)),
            "actuator_saturation_ratio": float(np.mean(saturated)),
            "rate_saturation_ratio": float(np.mean(rate_hit)),
            "min_margin_to_joint_limit_rad": float(np.min(np.minimum(rate_limited - self.low, self.high - rate_limited))),
        }


class ReferenceBundle:
    def __init__(self, direction: str):
        self.direction = direction
        path = TRAJ / f"com_shift_{direction}_ref.npz"
        with np.load(path) as z:
            self.t = np.asarray(z["t_ref"], dtype=float)
            self.q = np.asarray(z["q_ref"], dtype=float)
            self.phase = np.asarray(z["phase"]).astype(str) if "phase" in z.files else np.asarray(["unknown"] * len(self.t))
            self.left_contact = np.asarray(z["left_contact_active"], dtype=bool) if "left_contact_active" in z.files else np.ones(len(self.t), dtype=bool)
            self.right_contact = np.asarray(z["right_contact_active"], dtype=bool) if "right_contact_active" in z.files else np.ones(len(self.t), dtype=bool)
        base = np.genfromtxt(LOG / f"c1_qp0_baseline_{direction}.csv", delimiter=",", names=True)
        self.baseline_t = np.asarray(base["time_s"], dtype=float)
        self.baseline_pitch = np.asarray(base["root_pitch_rad"], dtype=float)
        self.baseline_pitch_rate = np.asarray(base["root_pitch_rate_rad_s"], dtype=float)

    @property
    def duration_s(self) -> float:
        return float(self.t[-1])

    def q_at(self, t: float) -> np.ndarray:
        tt = float(np.clip(t, self.t[0], self.t[-1]))
        return np.array([np.interp(tt, self.t, self.q[:, i]) for i in range(6)], dtype=float)

    def pitch_ref_at(self, t: float) -> float:
        return float(np.interp(float(np.clip(t, self.baseline_t[0], self.baseline_t[-1])), self.baseline_t, self.baseline_pitch))

    def contact_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for leg, active in (("left", self.left_contact), ("right", self.right_contact)):
            for i in range(1, len(active)):
                if bool(active[i]) != bool(active[i - 1]):
                    events.append({"time_s": float(self.t[i]), "kind": f"{leg}_{'touchdown' if active[i] else 'liftoff'}"})
        return sorted(events, key=lambda x: x["time_s"])


class FIRQP:
    def __init__(self, h: np.ndarray, ar: np.ndarray, candidate: dict[str, float] | None = None):
        cand = dict(DEFAULT_BALANCED if candidate is None else candidate)
        self.h = np.asarray(h, dtype=float)
        self.ar = np.asarray(ar, dtype=float)
        self.M = len(self.h)
        self.p = len(self.ar)
        self.q = float(cand["q"])
        self.r = float(cand["r"])
        self.slew = float(cand["slew"])
        self.prev = 0.0
        self.u_hist = deque([0.0] * (self.M - 1), maxlen=self.M - 1)
        self.d_hist = deque([0.0] * self.p, maxlen=self.p)
        self.S = self._make_s()
        self.D = np.eye(N)
        for i in range(1, N):
            self.D[i, i - 1] = -1.0
        self.H = self.q * (self.S.T @ self.S) + self.r * np.eye(N) + self.slew * (self.D.T @ self.D) + 1e-10 * np.eye(N)
        eigmax = float(np.max(np.linalg.eigvalsh(self.H)))
        self.qp_step = 1.0 / max(eigmax, 1e-10)

    def _make_s(self) -> np.ndarray:
        S = np.zeros((N, N))
        for row in range(N):
            future = row + 1
            for col in range(N):
                lag = future - col
                if 0 <= lag < self.M:
                    S[row, col] = self.h[lag]
        return S

    def _natural_future(self) -> np.ndarray:
        state = list(self.d_hist)
        out = []
        for _ in range(N):
            value = float(np.dot(self.ar, np.asarray(state[-self.p:][::-1])))
            state.append(value)
            out.append(value)
        return np.asarray(out)

    def update(self, error_rad: float) -> float:
        known = 0.0
        for index, uold in enumerate(self.u_hist, start=1):
            if index < self.M:
                known += self.h[index] * uold
        self.d_hist.append(float(error_rad - known))
        old = list(self.u_hist)
        past = []
        for row in range(N):
            future = row + 1
            val = 0.0
            for index, uold in enumerate(old, start=1):
                lag = future + index
                if lag < self.M:
                    val += self.h[lag] * uold
            past.append(val)
        free = self._natural_future() + np.asarray(past)
        dprev = np.zeros(N)
        dprev[0] = self.prev
        f = self.q * (self.S.T @ free) - self.slew * (self.D.T @ dprev)
        u = np.zeros(N)
        amax = math.radians(ALPHA_LIMIT_DEG)
        for _ in range(QP_ITERS):
            u = np.clip(u - self.qp_step * (self.H @ u + f), -amax, amax)
        slew = math.radians(ALPHA_SLEW_DEG)
        alpha = float(np.clip(u[0], self.prev - slew, self.prev + slew))
        alpha = float(np.clip(alpha, -amax, amax))
        self.prev = alpha
        self.u_hist.appendleft(alpha)
        return alpha


class ContactPhaseSynchronizer:
    def __init__(self, reference_events: list[dict[str, Any]], max_offset_s: float = 0.060):
        self.reference_events = list(reference_events)
        self.max_offset_s = float(max_offset_s)
        self.next_index = 0
        self.offset_s = 0.0
        self.matched = 0

    def reference_time(self, wall_time_s: float, duration_s: float) -> float:
        return float(np.clip(wall_time_s + self.offset_s, 0.0, duration_s))

    def update(self, observed_events: list[dict[str, Any]]) -> None:
        for observed in observed_events:
            if self.next_index >= len(self.reference_events):
                return
            expected = self.reference_events[self.next_index]
            expected_wall = float(expected["time_s"] - self.offset_s)
            timing_error = float(observed["time_s"] - expected_wall)
            if observed["kind"] == expected["kind"] and abs(timing_error) <= 0.120:
                target = float(np.clip(expected["time_s"] - observed["time_s"], -self.max_offset_s, self.max_offset_s))
                step = float(np.clip(0.75 * (target - self.offset_s), -0.020, 0.020))
                self.offset_s = float(np.clip(self.offset_s + step, -self.max_offset_s, self.max_offset_s))
                self.next_index += 1
                self.matched += 1


class C0Controller:
    controller_id = "C0_reference_playback"

    def __init__(self, direction: str = "forward"):
        self.ref = ReferenceBundle(direction)
        self.limiter = SafetyLimiter(self.ref.q_at(0.0))

    def update(self, measurement: ControllerMeasurement) -> dict[str, Any]:
        q_raw = self.ref.q_at(measurement.time_s)
        q_cmd, safety = self.limiter.apply(q_raw)
        return {"q_cmd_rad": q_cmd, "delta_q_rad": np.zeros(6), "alpha_rad": 0.0, "reference_time_s": measurement.time_s, "safety": safety}


class C1FIRQPMPCController(C0Controller):
    controller_id = "C1_fixed_phase_FIR_QP_MPC"

    def __init__(self, direction: str = "forward", ar: np.ndarray | None = None):
        super().__init__(direction)
        report = yaml.safe_load((LOG / "c1_qp1_fir_identification.yaml").read_text())
        h = np.asarray(report["directions"][direction]["fir_coefficients_rad_per_rad"], dtype=float)
        if ar is None:
            ar = np.array([0.80, 0.10, 0.05, 0.02], dtype=float)
        self.qp = FIRQP(h, ar)

    def update(self, measurement: ControllerMeasurement) -> dict[str, Any]:
        reference_time = float(np.clip(measurement.time_s, 0.0, self.ref.duration_s))
        pitch_error = float(measurement.root_pitch_rad - self.ref.pitch_ref_at(reference_time))
        alpha = self.qp.update(pitch_error)
        delta = G * alpha
        q_raw = self.ref.q_at(reference_time) + delta
        q_cmd, safety = self.limiter.apply(q_raw)
        return {"q_cmd_rad": q_cmd, "delta_q_rad": delta, "alpha_rad": alpha, "reference_time_s": reference_time, "safety": safety}


class C2ContactPhaseFIRQPMPCController(C1FIRQPMPCController):
    controller_id = "C2_contact_phase_FIR_QP_MPC"

    def __init__(self, direction: str = "forward", ar: np.ndarray | None = None):
        super().__init__(direction, ar=ar)
        self.phase_sync = ContactPhaseSynchronizer(self.ref.contact_events())
        self._last_left = True
        self._last_right = True

    def _observed_events(self, measurement: ControllerMeasurement) -> list[dict[str, Any]]:
        left = bool(measurement.left_toe_contact or measurement.left_heel_contact)
        right = bool(measurement.right_toe_contact or measurement.right_heel_contact)
        events = []
        if left != self._last_left:
            events.append({"time_s": measurement.time_s, "kind": f"left_{'touchdown' if left else 'liftoff'}"})
            self._last_left = left
        if right != self._last_right:
            events.append({"time_s": measurement.time_s, "kind": f"right_{'touchdown' if right else 'liftoff'}"})
            self._last_right = right
        return events

    def update(self, measurement: ControllerMeasurement) -> dict[str, Any]:
        self.phase_sync.update(self._observed_events(measurement))
        phased_time = self.phase_sync.reference_time(measurement.time_s, self.ref.duration_s)
        phased = ControllerMeasurement(
            time_s=phased_time,
            root_pitch_rad=measurement.root_pitch_rad,
            root_pitch_rate_rad_s=measurement.root_pitch_rate_rad_s,
            left_toe_contact=measurement.left_toe_contact,
            left_heel_contact=measurement.left_heel_contact,
            right_toe_contact=measurement.right_toe_contact,
            right_heel_contact=measurement.right_heel_contact,
        )
        out = super().update(phased)
        out["wall_time_s"] = measurement.time_s
        out["phase_offset_s"] = self.phase_sync.offset_s
        out["matched_contact_events"] = self.phase_sync.matched
        return out


CONTROLLERS = {
    "C0": C0Controller,
    "C1": C1FIRQPMPCController,
    "C2": C2ContactPhaseFIRQPMPCController,
}


def make_controller(name: str, direction: str = "forward"):
    return CONTROLLERS[name.upper()](direction=direction)
