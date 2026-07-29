from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "mpc" / "runtime_safety.py"


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RuntimeSafetyMachineTests(unittest.TestCase):
    def test_state_set_and_transition_log_are_explicit(self):
        module = load_module("test_runtime_safety_states")
        self.assertEqual(
            set(module.RuntimeState.__members__),
            {"NORMAL", "DEGRADED", "FALLBACK", "SAFE_HOLD", "FAILED"},
        )
        machine = module.RuntimeSafetyMachine(q_hold=np.zeros(6))

        state = machine.handle(module.Fault.DEADLINE_MISS, now_s=1.2)

        self.assertEqual(state, module.RuntimeState.DEGRADED)
        self.assertEqual(
            set(machine.transitions[-1]),
            {"time_s", "reason", "previous_state", "next_state", "recovery_outcome"},
        )

    def test_nonfinite_input_safe_holds_and_corrupt_reference_fails_startup(self):
        module = load_module("test_runtime_safety_input")
        machine = module.RuntimeSafetyMachine(q_hold=np.zeros(6))
        sample = SimpleNamespace(
            wall_time_s=0.0,
            root_pitch_rad=np.nan,
            root_pitch_rate_rad_s=np.inf,
        )

        faults = machine.validate_input(sample)
        self.assertIn(module.Fault.NAN_MEASUREMENT, faults)
        self.assertIn(module.Fault.INF_MEASUREMENT, faults)
        machine.handle(faults[0], now_s=0.0)
        self.assertEqual(machine.state, module.RuntimeState.SAFE_HOLD)

        startup = module.RuntimeSafetyMachine(q_hold=np.zeros(6))
        startup.handle(module.Fault.INVALID_REFERENCE_SCHEMA, now_s=0.0)
        self.assertEqual(startup.state, module.RuntimeState.FAILED)

    def test_safe_command_is_finite_and_rate_bounded(self):
        module = load_module("test_runtime_safety_command")
        machine = module.RuntimeSafetyMachine(q_hold=np.ones(6))
        q_last = np.zeros(6)

        command = machine.safe_command(q_last, np.ones(6), dt_s=0.020)

        self.assertTrue(np.all(np.isfinite(command)))
        self.assertTrue(np.all(np.abs(command - q_last) <= machine.max_step_rad + 1e-12))

    def test_invalid_output_is_detected(self):
        module = load_module("test_runtime_safety_output")
        machine = module.RuntimeSafetyMachine(q_hold=np.zeros(6))
        output = SimpleNamespace(
            q_cmd=np.array([0.0, 0.0, np.nan, 0.0, 0.0, 0.0]),
            saturated=True,
            limiter_active=True,
            deadline_missed=True,
        )

        faults = machine.validate_output(output)

        self.assertIn(module.Fault.INVALID_COMMAND, faults)
        self.assertIn(module.Fault.COMMAND_SATURATION, faults)
        self.assertIn(module.Fault.RATE_SATURATION, faults)
        self.assertIn(module.Fault.DEADLINE_MISS, faults)


class SafeControllerAdapterTests(unittest.TestCase):
    def test_primary_exception_falls_back_to_existing_c0_path(self):
        module = load_module("test_runtime_safety_adapter")

        class FakeController:
            def update(self, sample, mode):
                if mode != "C0":
                    raise RuntimeError("invalid QP")
                return SimpleNamespace(
                    q_cmd=np.full(6, 0.1),
                    saturated=False,
                    limiter_active=False,
                    deadline_missed=False,
                )

        adapter = module.SafeControllerAdapter(
            FakeController(), q_hold=np.zeros(6)
        )
        sample = SimpleNamespace(
            wall_time_s=0.1,
            root_pitch_rad=0.0,
            root_pitch_rate_rad_s=0.0,
        )

        result = adapter.update(sample, mode="C2_final")

        self.assertTrue(result.fallback_active)
        self.assertEqual(result.state, module.RuntimeState.FALLBACK)
        self.assertTrue(np.all(np.isfinite(result.q_cmd)))
        self.assertTrue(np.all(result.q_cmd > 0.0))
        self.assertTrue(np.all(
            result.q_cmd <= adapter.machine.max_step_rad + 1e-12
        ))


if __name__ == "__main__":
    unittest.main()
