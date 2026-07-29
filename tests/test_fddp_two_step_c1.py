import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "mpc" / "16_validate_fddp_two_step_c1_fir_qp.py"


def load_module():
    spec = importlib.util.spec_from_file_location("two_step_c1", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TwoStepC1Tests(unittest.TestCase):
    def test_loader_uses_two_step_fddp_reference_and_not_com_shift(self):
        module = load_module()
        t, q = module.load_two_step_reference()
        self.assertEqual(q.shape, (t.size, 6))
        self.assertIn("fddp_two_step_alternating_ref.npz", MODULE_PATH.read_text())
        self.assertNotIn("com_shift_forward_ref.npz", MODULE_PATH.read_text())


if __name__ == "__main__":
    unittest.main()
