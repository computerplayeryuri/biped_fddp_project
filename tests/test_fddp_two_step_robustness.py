import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "mpc" / "15_run_fddp_two_step_robustness_matrix.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fddp_two_step_robustness", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FDDPTwoStepRobustnessTests(unittest.TestCase):
    def test_scenarios_cover_required_physics_and_latency_categories(self):
        module = load_module()
        scenarios = module.build_scenarios()
        categories = {scenario.category for scenario in scenarios}
        self.assertTrue(
            {"nominal", "push", "friction", "slope", "step", "command_delay", "sensor_delay"}
            <= categories
        )
        self.assertTrue(all(scenario.repeat_count >= 5 for scenario in scenarios))

    def test_wilson_interval_bounds_success_rate(self):
        module = load_module()
        low, high = module.wilson_interval(successes=4, trials=5)
        self.assertLessEqual(0.0, low)
        self.assertLessEqual(low, 0.8)
        self.assertGreaterEqual(high, 0.8)
        self.assertLessEqual(high, 1.0)


    def test_aggregate_reports_success_ci_and_event_match_rate(self):
        module = load_module()
        aggregate = module.summarize_trials(
            [
                {"passed": True, "matched_contact_events": 3, "scheduled_contact_events": 4},
                {"passed": True, "matched_contact_events": 4, "scheduled_contact_events": 4},
                {"passed": False, "matched_contact_events": 1, "scheduled_contact_events": 4},
                {"passed": True, "matched_contact_events": 3, "scheduled_contact_events": 4},
                {"passed": True, "matched_contact_events": 2, "scheduled_contact_events": 4},
            ]
        )
        self.assertEqual(aggregate["trial_count"], 5)
        self.assertEqual(aggregate["success_count"], 4)
        self.assertAlmostEqual(aggregate["success_rate"], 0.8)
        self.assertAlmostEqual(aggregate["event_match_rate"], 13.0 / 20.0)
        self.assertIn("success_rate_wilson_95", aggregate)




    def test_trial_acceptance_requires_stability_and_event_evidence(self):
        module = load_module()
        accepted = module.trial_passed(
            {
                "min_torso_height_m": 0.28,
                "initial_torso_height_m": 0.288,
                "min_global_contact_count": 1,
                "max_abs_torso_pitch_deg": 12.0,
                "scheduled_contact_events": 4,
                "matched_contact_events": 3,
            }
        )
        self.assertTrue(accepted["passed"])
        rejected = module.trial_passed(
            {**accepted["metrics"], "min_global_contact_count": 0}
        )
        self.assertFalse(rejected["passed"])
        self.assertIn("lost ground contact", rejected["failures"])


    def test_c0_trial_does_not_fake_contact_match_evidence(self):
        module = load_module()
        result = module.trial_passed(
            {
                "mode": "c0",
                "min_torso_height_m": 0.28,
                "initial_torso_height_m": 0.288,
                "min_global_contact_count": 1,
                "max_abs_torso_pitch_deg": 12.0,
                "scheduled_contact_events": 4,
                "matched_contact_events": 0,
            }
        )
        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
