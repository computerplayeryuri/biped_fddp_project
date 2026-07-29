import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "fddp" / "multistep_core.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fddp_multistep_core", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MultiStepScheduleTests(unittest.TestCase):
    def test_ten_steps_preserve_five_complete_periodic_cycles(self):
        module = load_module()
        schedule = module.build_step_schedule(step_count=10)

        self.assertEqual(len(schedule), 25)
        self.assertEqual(sum(phase.node_count for phase in schedule), 1000)
        self.assertAlmostEqual(1000 * module.DT, 10.0)
        expected_modes = ["DS", "RIGHT_SS", "DS", "LEFT_SS", "DS"]
        expected_nodes = [20, 70, 20, 70, 20]
        for cycle_id in range(5):
            cycle = schedule[5 * cycle_id : 5 * (cycle_id + 1)]
            self.assertEqual([phase.mode for phase in cycle], expected_modes)
            self.assertEqual([phase.node_count for phase in cycle], expected_nodes)
            self.assertEqual([phase.cycle_id for phase in cycle], [cycle_id] * 5)

        self.assertAlmostEqual(module.planned_forward_displacement(schedule), 0.050)

    def test_phase_arrays_encode_contacts_and_unmerged_cycle_boundary(self):
        module = load_module()
        arrays = module.phase_arrays(module.build_step_schedule(step_count=10))

        self.assertEqual(arrays["phase_name"].shape, (1000,))
        self.assertTrue(arrays["left_contact_active"][19])
        self.assertTrue(arrays["right_contact_active"][19])
        self.assertFalse(arrays["left_contact_active"][20])
        self.assertTrue(arrays["right_contact_active"][20])
        self.assertTrue(arrays["left_contact_active"][110])
        self.assertFalse(arrays["right_contact_active"][110])
        self.assertEqual(arrays["swing_side"][20], "left")
        self.assertEqual(arrays["swing_side"][110], "right")
        self.assertEqual(arrays["mode"][199], "DS")
        self.assertEqual(arrays["mode"][200], "DS")
        self.assertEqual(arrays["cycle_id"][199], 0)
        self.assertEqual(arrays["cycle_id"][200], 1)

    def test_step_count_must_be_positive_even_and_final_run_has_ten_steps(self):
        module = load_module()
        for step_count in (0, 3, 9):
            with self.subTest(step_count=step_count):
                with self.assertRaises(ValueError):
                    module.build_step_schedule(step_count)
        with self.assertRaises(ValueError):
            module.build_step_schedule(8, require_final=True)


class MultiStepSchemaAndGateTests(unittest.TestCase):
    def test_experiment_config_loads_declared_schema_and_planning_thresholds(self):
        module = load_module()
        config = module.load_experiment_config()

        self.assertEqual(config["schema_version"], "fddp-multistep-config-v1")
        self.assertEqual(config["planning"]["final_step_count"], 10)
        self.assertEqual(config["planning"]["final_horizon_nodes"], 1000)
        self.assertEqual(config["planning"]["thresholds"], module.DEFAULT_THRESHOLDS)

    def test_export_metadata_is_versioned_and_marks_tau_analysis_only(self):
        module = load_module()
        metadata = module.export_metadata(step_count=10, horizon_nodes=1000)

        self.assertEqual(metadata["schema_version"], "fddp-multistep-v1")
        self.assertEqual(metadata["tau_ref_hardware_usage"], "analysis_only")
        self.assertEqual(metadata["command_interface"], "q_cmd=q_ref+delta_q")
        self.assertEqual(metadata["step_count"], 10)
        self.assertEqual(metadata["horizon_nodes"], 1000)
        self.assertAlmostEqual(metadata["duration_s"], 10.0)

    def test_planning_gate_accepts_exact_boundaries_and_reports_failures(self):
        module = load_module()
        metrics = {
            key: threshold
            for key, threshold in module.DEFAULT_THRESHOLDS.items()
        }
        accepted = module.evaluate_gate(metrics, module.DEFAULT_THRESHOLDS)
        self.assertTrue(accepted["passed"])
        self.assertEqual(accepted["failures"], [])
        self.assertTrue(all(accepted["checks"].values()))

        rejected_metrics = dict(metrics)
        rejected_metrics["solver_stop"] = np.nextafter(
            module.DEFAULT_THRESHOLDS["solver_stop"], np.inf
        )
        rejected_metrics["swing_clearance_m"] = np.nextafter(
            module.DEFAULT_THRESHOLDS["swing_clearance_m"], -np.inf
        )
        rejected = module.evaluate_gate(rejected_metrics, module.DEFAULT_THRESHOLDS)
        self.assertFalse(rejected["passed"])
        self.assertIn("solver_stop", rejected["failures"])
        self.assertIn("swing_clearance_m", rejected["failures"])

    def test_thresholds_and_metrics_reject_unknown_missing_or_nonfinite_values(self):
        module = load_module()
        invalid_thresholds = dict(module.DEFAULT_THRESHOLDS)
        invalid_thresholds["shooting_gap_norm"] = -1.0
        with self.assertRaises(ValueError):
            module.validate_thresholds(invalid_thresholds)

        with self.assertRaises(KeyError):
            module.evaluate_gate({}, module.DEFAULT_THRESHOLDS)

        invalid_metrics = dict(module.DEFAULT_THRESHOLDS)
        invalid_metrics["solver_stop"] = np.nan
        with self.assertRaises(ValueError):
            module.evaluate_gate(invalid_metrics, module.DEFAULT_THRESHOLDS)


if __name__ == "__main__":
    unittest.main()
