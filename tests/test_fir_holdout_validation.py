import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "mpc" / "06_identify_qp_mpc_fir_model.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fir_holdout", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FIRHoldoutValidationTests(unittest.TestCase):
    def test_holdout_metrics_predicts_opposite_sign_pulse_without_leakage(self):
        module = load_module()
        train_u = np.array([1.0, 0.0, 0.0])
        train_y = np.array([2.0, 1.0, 0.0])
        validation_u = np.array([-1.0, 0.0, 0.0])
        validation_y = np.array([-2.0, -1.0, 0.0])

        result = module.holdout_metrics(
            train_u,
            train_y,
            validation_u,
            validation_y,
            sample_time_s=1.0,
            fir_length=3,
        )

        self.assertAlmostEqual(result["multi_step_rmse_rad"], 0.0)
        self.assertAlmostEqual(result["one_step_rmse_rad"], 0.0)
        self.assertEqual(result["prediction_horizon_samples"], 3)
        np.testing.assert_allclose(
            result["fir_coefficients_rad_per_rad"],
            [2.0, 1.0, 0.0],
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
