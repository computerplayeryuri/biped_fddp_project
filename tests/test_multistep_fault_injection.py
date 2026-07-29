from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "mpc" / "20_run_multistep_fault_injection.py"


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MultiStepFaultInjectionTests(unittest.TestCase):
    def test_registry_covers_required_fault_classes(self):
        module = load_module("test_multistep_fault_registry")
        labels = {scenario.label for scenario in module.build_fault_scenarios()}
        self.assertTrue({
            "stale_pitch",
            "nan_pitch",
            "inf_pitch",
            "contact_stuck_on",
            "contact_stuck_off",
            "missing_contact_event",
            "out_of_order_contact_event",
            "invalid_qp",
            "deadline_miss",
            "command_saturation",
            "rate_saturation",
            "actuator_delay",
            "missing_reference",
            "truncated_reference",
            "invalid_reference_schema",
        }.issubset(labels))

    def test_campaign_contains_every_fault_without_unsafe_command(self):
        module = load_module("test_multistep_fault_campaign")
        report = module.run_fault_campaign(write_outputs=False)

        self.assertTrue(report["accepted"])
        self.assertEqual(
            len(report["rows"]), len(module.build_fault_scenarios())
        )
        self.assertTrue(all(
            not row["uncaught_exception"]
            and row["command_finite"]
            and row["command_within_limits"]
            and row["command_rate_bounded"]
            for row in report["rows"]
        ))
        states = {
            row["scenario"]: row["next_state"] for row in report["rows"]
        }
        self.assertEqual(states["invalid_qp"], "FALLBACK")
        self.assertEqual(states["nan_pitch"], "SAFE_HOLD")
        self.assertEqual(states["invalid_reference_schema"], "FAILED")


if __name__ == "__main__":
    unittest.main()
