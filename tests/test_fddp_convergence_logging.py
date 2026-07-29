import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "fddp" / "03n_solve_damped_6d_full_continuation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fddp_convergence", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FDDPConvergenceLoggingTests(unittest.TestCase):
    def test_iteration_row_contains_cost_gap_and_timing_fields(self):
        module = load_module()

        row = module.build_iteration_row(
            direction="forward",
            stage_index=1,
            attempt="primary",
            iteration=2,
            total_cost=1.2,
            max_gap_norm=0.3,
            rms_gap_norm=0.2,
            stop_criterion=1e-5,
            regularization=1e-7,
            step_length_alpha=0.5,
            iteration_wall_time_ms=3.0,
        )

        self.assertEqual(row["direction"], "forward")
        self.assertEqual(row["stage_index"], 1)
        self.assertEqual(row["iteration"], 2)
        self.assertEqual(row["total_cost"], 1.2)
        self.assertEqual(row["max_gap_norm"], 0.3)
        self.assertEqual(row["iteration_wall_time_ms"], 3.0)


if __name__ == "__main__":
    unittest.main()
