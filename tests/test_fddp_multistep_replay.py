import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "simulation" / "03u_replay_fddp_multistep_mujoco.py"
FINAL_REFERENCE = ROOT / "transfer" / "trajectories" / "fddp_multistep_10step_ref.npz"
OLD_REFERENCE = ROOT / "transfer" / "trajectories" / "fddp_two_step_alternating_ref.npz"


def load_module():
    spec = importlib.util.spec_from_file_location("fddp_multistep_replay", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MultiStepReferenceTests(unittest.TestCase):
    def test_final_reference_has_strict_schema_and_twenty_ordered_events(self):
        module = load_module()
        reference = module.load_multistep_reference(FINAL_REFERENCE)

        self.assertEqual(reference.schema_version, "fddp-multistep-v1")
        self.assertEqual(reference.acceptance_state, "planning_and_contact_accepted")
        self.assertEqual(reference.step_count, 10)
        self.assertEqual(reference.q_ref.shape, (1001, 6))
        self.assertEqual(reference.dq_ref.shape, (1001, 6))
        self.assertEqual(reference.com_ref_world.shape, (1001, 3))
        self.assertEqual(reference.duration_s, 10.0)
        self.assertNotIn("tau_ref", reference.__dataclass_fields__)

        events = module.build_reference_events(reference)
        self.assertEqual(len(events), 20)
        self.assertEqual(
            [event["kind"] for event in events[:4]],
            ["left_liftoff", "left_touchdown", "right_liftoff", "right_touchdown"],
        )
        self.assertEqual([event["index"] for event in events[:4]], [20, 90, 110, 180])
        self.assertEqual([event["index"] for event in events[-4:]], [820, 890, 910, 980])

    def test_loader_rejects_non_multistep_or_nonfinal_reference(self):
        module = load_module()
        with self.assertRaises(module.ReferenceIntegrityError):
            module.load_multistep_reference(OLD_REFERENCE)


class EventMatchingTests(unittest.TestCase):
    def test_ordered_matching_is_inclusive_at_tolerance(self):
        module = load_module()
        scheduled = [
            {"time_s": 0.20, "kind": "left_liftoff"},
            {"time_s": 0.90, "kind": "left_touchdown"},
            {"time_s": 1.10, "kind": "right_liftoff"},
            {"time_s": 1.80, "kind": "right_touchdown"},
        ]
        observed = [
            {"time_s": 0.00, "kind": "left_touchdown"},
            {"time_s": 0.28, "kind": "left_liftoff"},
            {"time_s": 0.82, "kind": "left_touchdown"},
            {"time_s": 1.10, "kind": "right_liftoff"},
            {"time_s": 1.80, "kind": "right_touchdown"},
        ]

        result = module.match_contact_events(scheduled, observed, tolerance_s=0.08)

        self.assertEqual(result["matched_count"], 4)
        self.assertEqual(result["skipped_count"], 0)
        self.assertEqual(result["extra_observed_count"], 1)
        self.assertEqual(result["match_ratio"], 1.0)
        self.assertTrue(result["event_order_valid"])
        self.assertTrue(result["all_event_kinds_present"])

    def test_wrong_order_cannot_be_counted_as_a_full_match(self):
        module = load_module()
        scheduled = [
            {"time_s": 0.20, "kind": "left_liftoff"},
            {"time_s": 0.90, "kind": "left_touchdown"},
        ]
        observed = [
            {"time_s": 0.20, "kind": "left_touchdown"},
            {"time_s": 0.90, "kind": "left_liftoff"},
        ]

        result = module.match_contact_events(scheduled, observed, tolerance_s=0.08)

        self.assertLess(result["match_ratio"], 1.0)
        self.assertFalse(result["event_order_valid"])


class ReplayGateTests(unittest.TestCase):
    def passing_metrics(self):
        return {
            "finite": True,
            "fall_detected": False,
            "max_joint_rmse_deg": 4.0,
            "max_abs_torso_pitch_deg": 15.0,
            "minimum_height_fraction": 0.65,
            "maximum_unintended_flight_s": 0.020,
            "maximum_sustained_penetration_s": 0.020,
            "event_match_ratio": 0.90,
            "event_order_valid": True,
            "all_event_kinds_present": True,
        }

    def test_replay_gate_is_inclusive_and_requires_contact_integrity(self):
        module = load_module()
        self.assertTrue(module.evaluate_replay(self.passing_metrics())["passed"])

        cases = {
            "max_joint_rmse_deg": 4.0001,
            "max_abs_torso_pitch_deg": 15.0001,
            "minimum_height_fraction": 0.6499,
            "maximum_unintended_flight_s": 0.0201,
            "maximum_sustained_penetration_s": 0.0201,
            "event_match_ratio": 0.8999,
            "event_order_valid": False,
            "all_event_kinds_present": False,
            "fall_detected": True,
            "finite": False,
        }
        for key, value in cases.items():
            with self.subTest(key=key):
                result = module.evaluate_replay({**self.passing_metrics(), key: value})
                self.assertFalse(result["passed"])
                self.assertIn(key, result["failures"])


class ReplayExecutionContractTests(unittest.TestCase):
    def test_two_step_smoke_window_and_output_paths_are_isolated(self):
        module = load_module()
        reference = module.load_multistep_reference(FINAL_REFERENCE)

        self.assertEqual(module.resolve_replay_duration(reference, smoke_steps=2), 2.0)
        smoke_paths = module.replay_output_paths(smoke_steps=2)
        final_paths = module.replay_output_paths(smoke_steps=None)

        self.assertNotEqual(smoke_paths, final_paths)
        self.assertIn("smoke_2step", smoke_paths["csv"].name)
        self.assertEqual(final_paths["csv"].name, "fddp_multistep_nominal_replay.csv")

    def test_real_two_step_smoke_replay_logs_position_and_contact_contract(self):
        module = load_module()

        result = module.run_nominal_replay(smoke_steps=2, write_outputs=False)

        self.assertEqual(result["schema_version"], "fddp-multistep-replay-v1")
        self.assertEqual(result["reference_duration_s"], 2.0)
        self.assertEqual(len(result["scheduled_events"]), 4)
        self.assertFalse(result["tau_ref_sent_to_ctrl"])
        self.assertEqual(result["control_source"], "q_ref_position_proxy_only")
        self.assertGreater(len(result["frame"]), 900)
        for column in (
            "time_s",
            "reference_time_s",
            "torso_pitch_deg",
            "torso_height_m",
            "left_contact_observed",
            "right_contact_observed",
            "scheduled_left_contact",
            "scheduled_right_contact",
            "penetration_depth_m",
            "unintended_flight_duration_s",
            "sustained_penetration_duration_s",
        ):
            self.assertIn(column, result["frame"].columns)
        self.assertIn("event_match_ratio", result["metrics"])
        self.assertIn("maximum_unintended_flight_s", result["metrics"])
        self.assertIn("maximum_sustained_penetration_s", result["metrics"])
        self.assertTrue(
            result["integrity_passed"],
            msg=f"two-step smoke replay failures: {result['failures']}",
        )


if __name__ == "__main__":
    unittest.main()
