import importlib.util
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "reporting" / "collect_controller_timing_safety.py"


def load_module():
    spec = importlib.util.spec_from_file_location("timing_safety", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ControllerTimingSafetyTests(unittest.TestCase):
    def test_summary_reports_percentiles_and_deadline_misses(self):
        module = load_module()
        result = module.summarize_timing(
            pd.DataFrame({"qp_solve_time_us": [0.0, 100.0, 500.0, 25000.0]}),
            controller_period_s=0.020,
        )
        self.assertEqual(result["update_count"], 3)
        self.assertEqual(result["deadline_miss_count"], 1)
        self.assertGreater(result["p95_qp_solve_time_us"], 500.0)


if __name__ == "__main__":
    unittest.main()
