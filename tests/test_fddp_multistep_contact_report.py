import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fddp"))
MODULE_PATH = ROOT / "fddp" / "05c_report_fddp_multistep_contact_feasibility.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "fddp_multistep_contact_report", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PlanarReportCsvTests(unittest.TestCase):
    def test_node_result_is_repeated_for_each_raw_contact_diagnostic(self):
        module = load_module()
        raw_rows = [
            {
                "node": 4,
                "time_s": 0.04,
                "phase": "double_support_start",
                "cycle_id": 0,
                "side": "left",
                "sole_position_world_m": np.array([0.0, 0.0, 0.0]),
                "wrench": np.array([1, 2, 3, 4, 5, 6], dtype=float),
                "tangential_speed_m_s": 0.0,
            },
            {
                "node": 4,
                "time_s": 0.04,
                "phase": "double_support_start",
                "cycle_id": 0,
                "side": "right",
                "sole_position_world_m": np.array([0.0, 0.02, 0.0]),
                "wrench": np.array([-1, -2, 7, -4, -5, -6], dtype=float),
                "tangential_speed_m_s": 0.0,
            },
        ]
        node_rows = [
            {
                "node": 4,
                "resultant_fz_n": 10.0,
                "friction_utilisation": 0.25,
                "cop_margin_m": 0.003,
                "accepted": True,
                "evaluation_mode": "planar_sagittal_resultant",
            }
        ]
        csv_rows = module._csv_rows(raw_rows, node_rows)
        self.assertEqual(len(csv_rows), 2)
        self.assertEqual([row["side"] for row in csv_rows], ["left", "right"])
        self.assertTrue(all(row["accepted"] for row in csv_rows))
        self.assertTrue(
            all(row["resultant_fz_n"] == 10.0 for row in csv_rows)
        )
        self.assertTrue(
            all(
                row["evaluation_mode"] == "planar_sagittal_resultant"
                for row in csv_rows
            )
        )


if __name__ == "__main__":
    unittest.main()