import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "mpc" / "13_run_scripted_robustness_matrix.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage9_robustness", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RobustnessMatrixTests(unittest.TestCase):
    def test_build_scenarios_includes_required_software_stress_categories(self):
        module = load_module()

        scenarios = module.build_scenarios()
        categories = {scenario.category for scenario in scenarios}
        labels = {scenario.label for scenario in scenarios}

        self.assertIn("duration", categories)
        self.assertIn("friction", categories)
        self.assertIn("slope_equivalent", categories)
        self.assertIn("small_step", categories)
        self.assertIn("push", categories)
        self.assertIn("duration_60s", labels)
        self.assertGreaterEqual(len(scenarios), 15)

    def test_expected_events_repeat_left_then_right_until_duration(self):
        module = load_module()

        events = module.expected_events_for_duration(6.1)
        kinds = [event["kind"] for event in events]
        times = [event["time_s"] for event in events]

        self.assertEqual(kinds[:4], ["left_liftoff", "left_touchdown", "right_liftoff", "right_touchdown"])
        self.assertEqual(times, sorted(times))
        self.assertEqual(len(events), 12)
        self.assertAlmostEqual(events[4]["time_s"] - events[0]["time_s"], module.CYCLE_S)

    def test_build_step_xml_inserts_left_and_right_toe_obstacles(self):
        module = load_module()

        xml_text = module.build_step_xml(0.004)

        self.assertIn('name="stage9_left_toe_step"', xml_text)
        self.assertIn('name="stage9_right_toe_step"', xml_text)
        self.assertIn('type="box"', xml_text)
        self.assertIn("0.002", xml_text)

    def test_scenario_passed_reports_individual_acceptance_flags(self):
        module = load_module()

        passing = module.scenario_passed(
            {
                "fall_detected": False,
                "min_contacts": 1,
                "peak_abs_pitch_deg": 12.0,
                "min_torso_height_m": 0.24,
                "matched_contact_events": 18,
                "scheduled_contact_events": 20,
                "max_abs_phase_offset_ms": 70.0,
            }
        )
        failing = module.scenario_passed(
            {
                "fall_detected": True,
                "min_contacts": 0,
                "peak_abs_pitch_deg": 35.0,
                "min_torso_height_m": 0.10,
                "matched_contact_events": 5,
                "scheduled_contact_events": 20,
                "max_abs_phase_offset_ms": 120.0,
            }
        )

        self.assertTrue(passing["robustness_pass"])
        self.assertEqual(passing["matched_event_ratio"], 0.9)
        self.assertFalse(failing["robustness_pass"])
        self.assertFalse(failing["no_fall_detected"])
        self.assertFalse(failing["event_match_ratio_ok"])


if __name__ == "__main__":
    unittest.main()
