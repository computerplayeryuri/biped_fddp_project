import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "mpc" / "14_validate_c1_c2_contact_mismatch.py"


def load_module():
    spec = importlib.util.spec_from_file_location("c1_c2_mismatch", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class C1C2ContactMismatchTests(unittest.TestCase):
    def test_late_touchdown_moves_only_target_event(self):
        module = load_module()
        events = [
            {"time_s": 0.36, "kind": "left_liftoff"},
            {"time_s": 1.00, "kind": "left_touchdown"},
        ]
        shifted = module.inject_event_mismatch(events, "late_touchdown_30ms")
        self.assertAlmostEqual(shifted[0]["time_s"], 0.36)
        self.assertAlmostEqual(shifted[1]["time_s"], 1.03)

    def test_c1_mode_keeps_zero_phase_offset(self):
        module = load_module()
        self.assertEqual(module.phase_offset_for_mode("C1_fixed_phase", 0.02), 0.0)
        self.assertEqual(module.phase_offset_for_mode("C2_phase_sync", 0.02), 0.02)


if __name__ == "__main__":
    unittest.main()
