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


class MultiStepRecoveryTests(unittest.TestCase):
    def test_recovery_bounds_alpha_and_foothold(self):
        module = load(
            ROOT / "mpc" / "19_tune_fddp_multistep_push_recovery.py",
            "test_multistep_recovery_bounds",
        )
        limits = module.RecoveryLimits()
        bounded = module.bound_recovery(
            np.deg2rad(9.0), 0.030, limits
        )

        self.assertAlmostEqual(abs(bounded.alpha_rad), np.deg2rad(5.0))
        self.assertAlmostEqual(abs(bounded.foothold_y_m), 0.015)
        self.assertTrue(
            module.recovery_terms_respect_limits(bounded, limits)
        )

    def test_staged_candidates_reuse_earlier_authority_first(self):
        module = load(
            ROOT / "mpc" / "19_tune_fddp_multistep_push_recovery.py",
            "test_multistep_recovery_order",
        )
        candidates = module.build_recovery_candidates()

        self.assertEqual(
            [candidate.stage for candidate in candidates],
            [
                "fir_qp_retune",
                "phase_gain_scheduling",
                "bounded_foothold",
            ],
        )

    def test_selection_is_first_candidate_passing_both_directions(self):
        module = load(
            ROOT / "mpc" / "19_tune_fddp_multistep_push_recovery.py",
            "test_multistep_recovery_selection",
        )
        rows = [
            {
                "candidate": "first",
                "forward_success_rate": 0.90,
                "backward_success_rate": 0.89,
            },
            {
                "candidate": "second",
                "forward_success_rate": 0.90,
                "backward_success_rate": 0.90,
            },
            {
                "candidate": "third",
                "forward_success_rate": 1.0,
                "backward_success_rate": 1.0,
            },
        ]

        selected = module.select_first_accepted(rows, minimum_rate=0.90)
        self.assertEqual(selected["candidate"], "second")

    def test_multistep_controller_accepts_recovery_gain_overrides(self):
        module = load(
            ROOT / "mpc" / "multistep_controller.py",
            "test_multistep_recovery_overrides",
        )
        controller = module.MultiStepController(
            fir_qp_candidate={
                "name": "recovery_fast",
                "q": 2500.0,
                "r": 5.0,
                "slew": 10.0,
            },
            phase_sync={
                "max_offset_s": 0.100,
                "max_step_s": 0.050,
                "correction_gain": 1.0,
            },
        )

        self.assertEqual(controller._qps["C2_final"].q, 2500.0)
        self.assertEqual(controller._qps["C2_final"].r, 5.0)
        self.assertEqual(controller._qps["C2_final"].slew, 10.0)
        self.assertEqual(
            controller._synchronizers["C2_final"].correction_gain,
            1.0,
        )

    def test_recovery_gain_schedule_activates_only_above_pitch_threshold(self):
        module = load(
            ROOT / "mpc" / "multistep_controller.py",
            "test_multistep_recovery_schedule",
        )
        controller = module.MultiStepController()
        nominal_pitch = controller.bundle.pitch_ref_at(0.0)

        normal = controller.update(
            module.ControllerInput(
                wall_time_s=0.0,
                root_pitch_rad=nominal_pitch + np.deg2rad(2.9),
            ),
            mode="C1",
        )
        recovery = controller.update(
            module.ControllerInput(
                wall_time_s=0.02,
                root_pitch_rad=(
                    controller.bundle.pitch_ref_at(0.02)
                    + np.deg2rad(4.0)
                ),
            ),
            mode="C1",
        )

        self.assertEqual(normal.controller_state, "normal")
        self.assertEqual(recovery.controller_state, "recovery")

    def test_recovery_candidate_maps_to_scheduled_fast_controller(self):
        module = load(
            ROOT / "mpc" / "19_tune_fddp_multistep_push_recovery.py",
            "test_multistep_recovery_candidate_mapping",
        )
        kwargs = module._controller_kwargs(
            module.build_recovery_candidates()[0]
        )

        self.assertNotIn("fir_qp_candidate", kwargs)
        self.assertEqual(
            kwargs["recovery_fir_qp_candidate"]["r"], 5.0
        )
        self.assertEqual(kwargs["recovery_activation_pitch_deg"], 3.0)

    def test_recovery_sweep_exports_recovery_activation_fraction(self):
        module = load(
            ROOT / "mpc" / "19_tune_fddp_multistep_push_recovery.py",
            "test_multistep_recovery_activation_export",
        )
        stable = {
            "maximum_unintended_flight_s": 0.0,
            "minimum_contacts": 2,
            "minimum_height_fraction": 0.98,
            "peak_abs_error_deg": 10.0,
            "event_match_ratio": 0.95,
            "event_order_valid": True,
            "max_abs_alpha_deg": 5.0,
            "max_abs_delta_q_deg": 1.0,
            "max_abs_phase_offset_ms": 20.0,
            "recovery_active_fraction": 0.25,
            "saturation_fraction": 0.0,
        }

        class FakeRunner:
            @staticmethod
            def _run_mode(*args, **kwargs):
                return {"metrics": dict(stable)}

        module._load_runner = lambda: FakeRunner
        _, rows = module._run_candidate(
            module.build_recovery_candidates()[0],
            [0],
            module.RecoveryLimits(),
        )

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(
            row["recovery_active_fraction"] == 0.25 for row in rows
        ))

    def test_recovery_trial_gate_reuses_formal_flight_duration_limit(self):
        module = load(
            ROOT / "mpc" / "19_tune_fddp_multistep_push_recovery.py",
            "test_multistep_recovery_gate",
        )
        stable = {
            "maximum_unintended_flight_s": 0.020,
            "minimum_contacts": 0,
            "minimum_height_fraction": 0.98,
            "peak_abs_error_deg": 10.0,
            "event_match_ratio": 0.95,
            "event_order_valid": True,
            "max_abs_alpha_deg": 5.0,
            "saturation_fraction": 0.0,
        }

        self.assertTrue(module.evaluate_trial(stable)["passed"])
        rejected = module.evaluate_trial({
            **stable, "maximum_unintended_flight_s": 0.0201
        })
        self.assertFalse(rejected["passed"])
        self.assertIn("unintended_flight_above_limit", rejected["failures"])

    def test_existing_ablation_runner_accepts_bounded_push_and_seed(self):
        module = load(
            ROOT / "mpc" / "18_validate_fddp_multistep_c2_final.py",
            "test_multistep_push_runner",
        )
        result = module._run_mode(
            "C2_final",
            0.10,
            push_force_n=4.0,
            push_start_s=0.0,
            push_duration_s=0.05,
            seed=7,
            initial_joint_noise_std_rad=1.0e-4,
        )

        self.assertEqual(result["metrics"]["push_force_n"], 4.0)
        self.assertEqual(result["metrics"]["seed"], 7)
        self.assertTrue(np.isfinite(result["metrics"]["peak_abs_error_deg"]))
        self.assertIn("controller_state", result["frame"].columns)
        self.assertIn(
            "recovery_active_fraction", result["metrics"]
        )


    def test_existing_ablation_runner_applies_push_to_mujoco_plant(self):
        module = load(
            ROOT / "mpc" / "18_validate_fddp_multistep_c2_final.py",
            "test_multistep_push_is_physical",
        )
        baseline = module._run_mode(
            "C0",
            0.10,
            push_force_n=0.0,
            push_start_s=0.0,
            push_duration_s=0.05,
        )
        pushed = module._run_mode(
            "C0",
            0.10,
            push_force_n=4.0,
            push_start_s=0.0,
            push_duration_s=0.05,
        )

        self.assertEqual(pushed["metrics"]["push_force_n"], 4.0)
        self.assertTrue(np.any(np.isclose(
            pushed["frame"]["applied_push_force_n"].to_numpy(), 4.0
        )))
        pitch_delta = np.max(np.abs(
            pushed["frame"]["pitch_measured_rad"].to_numpy()
            - baseline["frame"]["pitch_measured_rad"].to_numpy()
        ))
        self.assertGreater(pitch_delta, 1.0e-6)

    def test_alpha_projection_respects_current_joint_headroom(self):
        module = load(
            ROOT / "mpc" / "multistep_controller.py",
            "test_multistep_alpha_feasible_projection",
        )
        controller = module.MultiStepController()
        limiter = controller._limiters["C2_final"]
        q_ref = 0.5 * (limiter.low + limiter.high)
        q_ref[0] = limiter.high[0] - 1.0e-4

        projected = module.project_alpha_to_joint_limits(
            np.deg2rad(5.0),
            q_ref,
            limiter.low,
            limiter.high,
        )
        q_raw = q_ref + module.runtime.G * projected

        self.assertLess(projected, np.deg2rad(5.0))
        self.assertTrue(np.all(q_raw >= limiter.low - 1.0e-12))
        self.assertTrue(np.all(q_raw <= limiter.high + 1.0e-12))
if __name__ == "__main__":
    unittest.main()
