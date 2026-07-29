import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "simulation" / "03t_render_fddp_two_step_c1_disturbance_animation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("two_step_disturbance_animation", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TwoStepDisturbanceAnimationTests(unittest.TestCase):
    def test_animation_uses_two_step_c1_pipeline_and_dedicated_output(self):
        module = load_module()
        source = MODULE_PATH.read_text(encoding="utf-8")

        self.assertIn("16_validate_fddp_two_step_c1_fir_qp.py", source)
        self.assertIn("load_two_step_reference", source)
        self.assertNotIn("com_shift_ref", source)
        self.assertEqual(
            module.OUT.name,
            "fddp_two_step_c1_pitch_disturbance.gif",
        )


if __name__ == "__main__":
    unittest.main()
