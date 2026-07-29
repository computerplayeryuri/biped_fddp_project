import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "mpc" / "17_run_fddp_two_step_c1_parameter_ablation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("two_step_c1_ablation", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TwoStepC1AblationTests(unittest.TestCase):
    def test_candidates_are_balanced_plus_one_factor_variants(self):
        module = load_module()
        candidates = module.build_ablation_candidates()
        self.assertEqual(len(candidates), 7)
        self.assertEqual(len({item["name"] for item in candidates}), 7)
        balanced = candidates[0]
        self.assertEqual(
            (balanced["q"], balanced["r"], balanced["slew"]),
            (1500.0, 20.0, 25.0),
        )
        for candidate in candidates[1:]:
            changed = sum(candidate[key] != balanced[key] for key in ("q", "r", "slew"))
            self.assertEqual(changed, 1)

    def test_acceptance_requires_performance_and_contact_gates(self):
        module = load_module()
        c0 = {"iae_deg_s": 0.50, "peak_abs_error_deg": 2.0}
        passing = {
            "iae_deg_s": 0.40,
            "peak_abs_error_deg": 2.05,
            "max_abs_alpha_deg": 3.0,
            "min_contacts": 2,
            "min_torso_height_m": 0.20,
            "initial_torso_height_m": 0.25,
        }
        summary = module.evaluate_candidate(passing, c0)
        self.assertTrue(summary["acceptance_passed"])
        failing = dict(passing, min_contacts=1)
        self.assertFalse(module.evaluate_candidate(failing, c0)["acceptance_passed"])

    def test_source_uses_two_step_reference_not_com_shift_trajectory(self):
        load_module()
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn("load_two_step_reference", source)
        self.assertNotIn("com_shift_ref.npz", source)

    def test_result_row_uses_candidate_column_for_persisted_schema(self):
        module = load_module()
        row = module.build_result_row(
            {"name": "balanced", "q": 1500.0, "r": 20.0, "slew": 25.0},
            {"iae_deg_s": 0.4},
            {"acceptance_passed": True},
        )
        self.assertEqual(row["candidate"], "balanced")
        self.assertNotIn("name", row)


if __name__ == "__main__":
    unittest.main()
