import importlib.util
import subprocess
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "mpc" / "multistep_controller.py"
VALIDATOR_PATH = ROOT / "mpc" / "18_validate_fddp_multistep_c2_final.py"
FINAL_REFERENCE = (
    ROOT / "transfer" / "trajectories" / "fddp_multistep_10step_ref.npz"
)
OLD_REFERENCE = (
    ROOT / "transfer" / "trajectories" / "fddp_two_step_alternating_ref.npz"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "fddp_multistep_controller", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_validator():
    spec = importlib.util.spec_from_file_location(
        "fddp_multistep_validator", VALIDATOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sample(module, *, time_s=0.0, pitch_rad=0.10, left=True, right=True):
    return module.ControllerInput(
        wall_time_s=time_s,
        root_pitch_rad=pitch_rad,
        root_pitch_rate_rad_s=0.0,
        left_toe_contact=left,
        left_heel_contact=left,
        right_toe_contact=right,
        right_heel_contact=right,
    )


class MultiStepControllerInterfaceTests(unittest.TestCase):
    def test_final_reference_is_required(self):
        module = load_module()

        controller = module.MultiStepController(reference_path=FINAL_REFERENCE)
        self.assertEqual(controller.reference.schema_version, "fddp-multistep-v1")
        self.assertEqual(controller.reference.step_count, 10)

        with self.assertRaises(module.ReferenceIntegrityError):
            module.MultiStepController(reference_path=OLD_REFERENCE)

    def test_all_modes_share_finite_position_command_contract(self):
        module = load_module()
        controller = module.MultiStepController()

        outputs = {
            mode: controller.update(sample(module), mode=mode)
            for mode in ("C0", "C1", "phase_only", "C2_final")
        }
        for mode, output in outputs.items():
            with self.subTest(mode=mode):
                self.assertEqual(output.schema_version, "fddp-controller-v1")
                self.assertEqual(output.mode, mode)
                self.assertEqual(output.q_cmd.shape, (6,))
                self.assertEqual(output.q_ref.shape, (6,))
                self.assertEqual(output.delta_q.shape, (6,))
                self.assertTrue(np.all(np.isfinite(output.q_cmd)))
                self.assertTrue(np.allclose(
                    output.q_cmd, output.q_ref + output.delta_q
                ))
                self.assertLessEqual(
                    abs(output.alpha_rad), np.deg2rad(5.0) + 1.0e-12
                )
                self.assertIsInstance(output.limiter_active, bool)
                self.assertIsInstance(output.saturated, bool)
                self.assertFalse(output.fallback_active)

        for mode in ("C1", "C2_final"):
            self.assertEqual(controller._qps[mode].q, 1500.0)
            self.assertEqual(controller._qps[mode].r, 10.0)
            self.assertEqual(controller._qps[mode].slew, 25.0)
            self.assertEqual(controller._recovery_qps[mode].q, 2500.0)
            self.assertEqual(controller._recovery_qps[mode].r, 5.0)
            self.assertEqual(controller._recovery_qps[mode].slew, 10.0)
        self.assertEqual(controller.recovery_activation_pitch_deg, 3.0)
        self.assertAlmostEqual(
            controller._synchronizers["C2_final"].max_offset_s, 0.100
        )
        self.assertAlmostEqual(
            controller._synchronizers["C2_final"].max_step_s, 0.040
        )
        self.assertEqual(outputs["C0"].alpha_rad, 0.0)
        self.assertEqual(outputs["C0"].phase_offset_s, 0.0)
        self.assertTrue(np.allclose(outputs["C0"].delta_q, 0.0))
        self.assertNotEqual(outputs["C1"].alpha_rad, 0.0)
        self.assertEqual(outputs["C1"].phase_offset_s, 0.0)
        self.assertEqual(outputs["phase_only"].alpha_rad, 0.0)

    def test_phase_modes_retime_the_same_natural_contact_event(self):
        module = load_module()
        controller = module.MultiStepController()
        self.assertAlmostEqual(
            controller.controller_configuration["phase_sync"]["match_window_s"],
            0.220,
        )

        for mode in ("phase_only", "C2_final"):
            controller.update(sample(module), mode=mode)
            first = controller.update(
                sample(module, time_s=0.25, left=False, right=True),
                mode=mode,
            )
            self.assertEqual(first.phase_offset_s, 0.0)
            controller.update(
                sample(module, time_s=0.95, left=True, right=True),
                mode=mode,
            )
            controller.update(
                sample(module, time_s=1.15, left=True, right=False),
                mode=mode,
            )
            output = controller.update(
                sample(module, time_s=1.85, left=True, right=True),
                mode=mode,
            )
            with self.subTest(mode=mode):
                self.assertGreater(output.phase_offset_s, 0.0)
                self.assertEqual(output.matched_events, 4)
                self.assertEqual(output.skipped_events, 0)
                self.assertAlmostEqual(output.reference_time_s, 1.85)
                base_q = controller.bundle.q_at(
                    1.85 + controller.command_reference_advance_s
                )
                phase_q = controller.bundle.q_at(
                    1.85
                    + output.phase_offset_s
                    + controller.command_reference_advance_s
                )
                self.assertTrue(np.allclose(output.q_ref[:3], phase_q[:3]))
                self.assertTrue(np.allclose(output.q_ref[3:], base_q[3:]))

        self.assertEqual(
            controller.update(sample(module, time_s=0.25), mode="C1").phase_offset_s,
            0.0,
        )

    def test_predictive_matching_keeps_nominal_observation_time(self):
        module = load_module()
        synchronizer = module.phase_runtime.ContactPhaseSynchronizer(
            [
                {"time_s": 0.2, "kind": "left_liftoff"},
                {"time_s": 2.2, "kind": "left_liftoff"},
            ],
            predictive=True,
            match_window_s=0.22,
        )

        first = synchronizer.update(
            0.25, [{"time_s": 0.25, "kind": "left_liftoff"}]
        )
        second = synchronizer.update(
            2.2, [{"time_s": 2.2, "kind": "left_liftoff"}]
        )
        self.assertGreater(first[0]["new_offset_s"], 0.0)
    def test_rate_limiter_and_invalid_mode_are_explicit(self):
        module = load_module()
        controller = module.MultiStepController()
        controller.update(sample(module), mode="C0")
        c0 = controller.update(sample(module, time_s=0.50), mode="C0")

        self.assertFalse(c0.limiter_active)
        self.assertTrue(np.allclose(c0.delta_q, 0.0))
        self.assertTrue(np.allclose(
            c0.q_cmd, c0.q_ref + c0.delta_q
        ))

        controller._limiters["C1"].max_step = 1e-6
        limited = controller.update(sample(module), mode="C1")
        self.assertTrue(limited.limiter_active)
        self.assertTrue(np.allclose(
            limited.q_cmd, limited.q_ref + limited.delta_q
        ))
        with self.assertRaisesRegex(ValueError, "mode"):
            controller.update(sample(module), mode="unknown")

    def test_controller_imports_when_mpc_is_the_working_directory(self):
        result = subprocess.run(
            [sys.executable, "-c", "import multistep_controller"],
            cwd=ROOT / "mpc",
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)


class MultiStepControllerAblationTests(unittest.TestCase):
    def passing_metrics(self):
        return {
            "C0": {
                "iae_deg_s": 1.0,
                "peak_abs_error_deg": 2.0,
                "mean_abs_contact_timing_error_s": 0.10,
            },
            "C1": {
                "iae_deg_s": 0.90,
                "peak_abs_error_deg": 2.10,
                "mean_abs_contact_timing_error_s": 0.10,
            },
            "phase_only": {
                "iae_deg_s": 1.0,
                "peak_abs_error_deg": 2.0,
                "mean_abs_contact_timing_error_s": 0.07,
            },
            "C2_final": {
                "iae_deg_s": 0.918,
                "peak_abs_error_deg": 2.205,
                "mean_abs_contact_timing_error_s": 0.07,
            },
        }

    def test_ablation_gate_is_inclusive_and_rejects_c1_underperformance(self):
        validator = load_validator()

        accepted = validator.evaluate_ablation(self.passing_metrics())
        self.assertTrue(accepted["passed"])

        failed = self.passing_metrics()
        failed["C1"]["iae_deg_s"] = 0.9001
        decision = validator.evaluate_ablation(failed)
        self.assertFalse(decision["passed"])
        self.assertIn("c1_iae_improvement", decision["failures"])

    def test_two_step_smoke_reuses_one_protocol_for_all_four_modes(self):
        validator = load_validator()

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            report = validator.run_ablation(smoke_steps=2, write_outputs=False)

        self.assertEqual(
            report["schema_version"],
            "fddp-multistep-controller-ablation-v1",
        )
        self.assertEqual(
            set(report["mode_metrics"]),
            {"C0", "C1", "phase_only", "C2_final"},
        )
        self.assertEqual(report["disturbance"]["time_s"], 0.40)
        self.assertEqual(set(report["frames"]), set(report["mode_metrics"]))


if __name__ == "__main__":
    unittest.main()
