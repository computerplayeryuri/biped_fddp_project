from __future__ import annotations

from collections import deque
import numpy as np


class ActuatorReferenceFilter:
    """Simple ST3215 reference-side proxy: delay + first-order response + rate limit.

    This is intentionally a command filter, not a torque model. It represents the fact that
    a position-controlled serial servo cannot follow an arbitrary joint reference instantly.
    """

    def __init__(
        self,
        n_joints: int,
        control_dt_s: float,
        delay_s: float,
        time_constant_s: float,
        safe_speed_rad_s: float,
        max_change_per_cycle_rad: float,
        initial_reference: np.ndarray | None = None,
    ) -> None:
        if control_dt_s <= 0 or time_constant_s <= 0:
            raise ValueError("control_dt_s and time_constant_s must be positive")

        self.n_joints = int(n_joints)
        self.dt = float(control_dt_s)
        self.tau = float(time_constant_s)
        self.safe_speed = float(safe_speed_rad_s)
        self.max_change = float(max_change_per_cycle_rad)
        self.delay_steps = max(0, int(round(float(delay_s) / self.dt)))

        if initial_reference is None:
            initial_reference = np.zeros(self.n_joints, dtype=float)
        initial_reference = np.asarray(initial_reference, dtype=float).copy()

        self._buffer = deque(
            [initial_reference.copy() for _ in range(self.delay_steps + 1)],
            maxlen=self.delay_steps + 1,
        )
        self.command = initial_reference.copy()

    def update(self, reference: np.ndarray) -> np.ndarray:
        reference = np.asarray(reference, dtype=float)
        if reference.shape != (self.n_joints,):
            raise ValueError(
                f"Expected reference shape ({self.n_joints},), got {reference.shape}"
            )

        self._buffer.append(reference.copy())
        delayed_reference = self._buffer[0]

        desired_increment = (delayed_reference - self.command) * self.dt / self.tau
        speed_increment_limit = self.safe_speed * self.dt
        increment_limit = min(speed_increment_limit, self.max_change)
        increment = np.clip(desired_increment, -increment_limit, increment_limit)
        self.command = self.command + increment

        return self.command.copy()
