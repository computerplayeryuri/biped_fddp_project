from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ReusedIdentificationTests(unittest.TestCase):
    def test_existing_identification_supports_bounded_reference_advance(self):
        module = load(
            ROOT / "mpc" / "16_validate_fddp_two_step_c1_fir_qp.py",
            "test_reusable_two_step_identification",
        )
        module.COMMAND_REFERENCE_ADVANCE_S = 0.03
        t_ref = np.asarray([0.0, 0.5, 1.0])

        self.assertAlmostEqual(
            module.command_reference_time(0.20, t_ref), 0.23
        )
        self.assertAlmostEqual(
            module.command_reference_time(0.99, t_ref), 1.0
        )

    def test_multistep_adapter_retargets_existing_identification_functions(self):
        module = load(
            ROOT / "mpc" / "19_identify_fddp_multistep_fir.py",
            "test_multistep_identification_adapter",
        )
        reused = module.configure_reused_identification()

        self.assertEqual(
            Path(module.REUSE_SOURCE_PATH).name,
            "16_validate_fddp_two_step_c1_fir_qp.py",
        )
        self.assertEqual(reused.NPZ_PATH, module.FINAL_REFERENCE_PATH)
        self.assertEqual(reused.BASELINE_CSV, module.BASELINE_CSV_PATH)
        self.assertEqual(reused.FIR_YAML, module.FIR_YAML_PATH)
        self.assertEqual(reused.AR_YAML, module.AR_YAML_PATH)
        self.assertAlmostEqual(
            reused.COMMAND_REFERENCE_ADVANCE_S,
            module.COMMAND_REFERENCE_ADVANCE_S,
        )
        self.assertAlmostEqual(reused.FIR_SAMPLE_S, 0.020)
        self.assertAlmostEqual(reused.PULSE_WIDTH_S, 0.020)

    def test_shared_controller_uses_the_multistep_models_by_default(self):
        identification = load(
            ROOT / "mpc" / "19_identify_fddp_multistep_fir.py",
            "test_multistep_identification_paths",
        )
        controller = load(
            ROOT / "mpc" / "multistep_controller.py",
            "test_multistep_controller_model_paths",
        )

        self.assertEqual(controller.FIR_YAML_PATH, identification.FIR_YAML_PATH)
        self.assertEqual(controller.AR_YAML_PATH, identification.AR_YAML_PATH)


if __name__ == "__main__":
    unittest.main()
