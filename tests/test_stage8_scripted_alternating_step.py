import importlib.util
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "mpc" / "12_validate_scripted_alternating_step_locomotion.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage8_scripted_step", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ScriptedAlternatingStepTests(unittest.TestCase):
    def test_expected_events_are_ordered_left_then_right(self):
        module = load_module()

        kinds = [event["kind"] for event in module.EXPECTED_EVENTS]
        times = [event["time_s"] for event in module.EXPECTED_EVENTS]

        self.assertEqual(kinds, ["left_liftoff", "left_touchdown", "right_liftoff", "right_touchdown"])
        self.assertEqual(times, sorted(times))

    def test_q_reference_lifts_each_leg_during_its_swing_window(self):
        module = load_module()

        left_mid = module.q_reference(0.55)
        right_mid = module.q_reference(1.50)
        quiet = module.q_reference(0.05)

        self.assertGreater(abs(left_mid[1]), math.radians(35.0))
        self.assertLess(abs(left_mid[4]), math.radians(5.0))
        self.assertGreater(abs(right_mid[4]), math.radians(35.0))
        self.assertLess(abs(right_mid[1]), math.radians(5.0))
        self.assertTrue((abs(quiet) < 1e-12).all())

    def test_build_summary_marks_four_matched_events_as_acceptance_pass(self):
        module = load_module()
        observed = [{"time_s": event["time_s"], "kind": event["kind"]} for event in module.EXPECTED_EVENTS]
        rows = [
            {
                "time_s": 0.0,
                "phase_offset_s": 0.0,
                "torso_pitch_rad": 0.0,
                "torso_height_m": 0.288,
                "left_contact_active": True,
                "right_contact_active": True,
                "ncon": 4,
            },
            {
                "time_s": module.DURATION_S,
                "phase_offset_s": 0.01,
                "torso_pitch_rad": 0.01,
                "torso_height_m": 0.280,
                "left_contact_active": True,
                "right_contact_active": True,
                "ncon": 4,
            },
        ]

        summary = module.build_summary(rows, observed, matched_count=4, skipped_count=0)

        self.assertEqual(summary["metrics"]["scheduled_contact_events"], 4)
        self.assertEqual(summary["metrics"]["matched_contact_events"], 4)
        self.assertEqual(summary["metrics"]["scripted_step_count"], 2)
        self.assertTrue(all(summary["acceptance"].values()))


if __name__ == "__main__":
    unittest.main()
