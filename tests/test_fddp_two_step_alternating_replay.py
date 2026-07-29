import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "simulation" / "03r_replay_fddp_two_step_alternating_mujoco.py"


def load_module():
    spec = importlib.util.spec_from_file_location("two_step_replay", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TwoStepReplayTests(unittest.TestCase):
    def test_reference_matches_two_step_contract(self):
        module = load_module()
        t, q, dq, com, phase = module.load_reference()
        self.assertEqual(q.shape, (t.size, 6))
        self.assertEqual(dq.shape, (t.size, 6))
        self.assertEqual(com.shape, (t.size, 3))
        self.assertEqual(phase[0], "double_support_start")
        self.assertEqual(phase[20], "right_stance_left_swing")
        self.assertEqual(phase[110], "left_stance_right_swing")
        self.assertEqual(phase[-1], "double_support_landing")
        self.assertIn("tau_ref_usage", MODULE_PATH.read_text())


if __name__ == "__main__":
    unittest.main()
