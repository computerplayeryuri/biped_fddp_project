import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "mpc" / "14_validate_fddp_two_step_natural_contact_c2.py"


def load_module():
    spec = importlib.util.spec_from_file_location("two_step_natural_c2", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TwoStepNaturalContactC2Tests(unittest.TestCase):
    def test_fddp_reference_exposes_four_leg_transition_events(self):
        module = load_module()
        reference = module.load_reference()
        events = module.build_reference_events(reference)
        self.assertEqual(
            [event["kind"] for event in events],
            ["left_liftoff", "left_touchdown", "right_liftoff", "right_touchdown"],
        )
        self.assertEqual([event["index"] for event in events], [20, 90, 110, 180])
        self.assertIn("tau_ref_usage", MODULE_PATH.read_text())


if __name__ == "__main__":
    unittest.main()
