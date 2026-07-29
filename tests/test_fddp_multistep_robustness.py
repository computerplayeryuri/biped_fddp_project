from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


import mujoco
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "mpc" / "robustness_statistics.py"

RUNNER_PATH = ROOT / "mpc" / "18_validate_fddp_multistep_c2_final.py"
BATCH_PATH = ROOT / "mpc" / "21_run_fddp_multistep_robustness_matrix.py"


def load_path_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_runner(name: str):
    return load_path_module(name, RUNNER_PATH)


def load_batch(name: str):
    return load_path_module(name, BATCH_PATH)


def initialise_reference_pose(runner, step_height_m: float):
    model = runner._build_robustness_model(step_height_m=step_height_m)
    data = mujoco.MjData(model)
    mapping = runner.proxy.resolve_joint_and_actuator_mapping(model)
    controller = runner.controller_runtime.MultiStepController()
    mujoco.mj_resetData(model, data)
    for address, value in zip(
        mapping["qpos_addresses"], controller.reference.q_ref[0]
    ):
        data.qpos[address] = value
    mujoco.mj_forward(model, data)
    return model, data



def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MultiStepRobustnessStatisticsTests(unittest.TestCase):
    def test_step_obstacle_geometry_clears_initial_toe_by_five_mm(self):
        runner = load_runner("test_valid_step_geometry")
        geometry = runner.terrain_runtime.step_obstacle_geometry()

        self.assertAlmostEqual(geometry["initial_toe_y_m"], 0.0725)
        self.assertAlmostEqual(geometry["obstacle_rear_edge_y_m"], 0.0775)
        self.assertAlmostEqual(geometry["initial_clearance_m"], 0.005)
        self.assertAlmostEqual(geometry["obstacle_center_y_m"], 0.0895)

    def test_step_obstacle_geometry_rejects_insufficient_clearance(self):
        runner = load_runner("test_invalid_step_geometry")

        with self.assertRaisesRegex(ValueError, "clearance"):
            runner.terrain_runtime.step_obstacle_geometry(0.0049)

    def test_valid_6mm_obstacle_has_no_initial_foot_contact(self):
        runner = load_runner("test_valid_6mm_initial_contact")
        model, data = initialise_reference_pose(runner, step_height_m=0.006)
        ids = runner._resolve_foot_terrain_contact_ids(model)
        snapshot = runner._obstacle_contact_snapshot(data, ids)

        self.assertFalse(snapshot["detected"])
        self.assertEqual(snapshot["contact_count"], 0)
        self.assertEqual(snapshot["maximum_penetration_m"], 0.0)

    def test_6mm_replay_encounters_obstacle_after_initialisation(self):
        runner = load_runner("test_6mm_obstacle_encounter")
        result = runner._run_mode("C2_final", 10.0, step_height_m=0.006)
        metrics = result["metrics"]

        self.assertTrue(metrics["step_obstacle_contact_detected"])
        self.assertGreater(metrics["step_obstacle_contact_sample_count"], 0)
        self.assertGreater(metrics["first_step_obstacle_contact_time_s"], 0.0)
        self.assertLessEqual(metrics["peak_abs_error_deg"], 15.0)
        self.assertEqual(metrics["event_match_ratio"], 1.0)

    def test_step_trial_requires_real_obstacle_encounter(self):
        batch = load_batch("test_step_encounter_gate")
        scenario = next(
            item
            for item in batch.statistics.build_scenarios()
            if item.label == "step_6mm"
        )

        self.assertEqual(
            batch._scenario_specific_failures(
                scenario, {"step_obstacle_contact_detected": False}
            ),
            ["step_obstacle_not_encountered"],
        )
        self.assertEqual(
            batch._scenario_specific_failures(
                scenario, {"step_obstacle_contact_detected": True}
            ),
            [],
        )

    def test_registry_is_complete_unique_and_deterministic(self):
        module = load_module("test_multistep_robustness_registry")
        scenarios = module.build_scenarios()
        nominal = next(item for item in scenarios if item.label == "nominal")
        self.assertEqual(nominal.command_delay_s, 0.020)
        self.assertEqual(
            nominal.actuator_time_constant_s, 0.030
        )

        self.assertEqual(len(scenarios), 32)
        self.assertEqual(len({item.label for item in scenarios}), 32)
        self.assertTrue(all(item.seed_count >= 30 for item in scenarios))
        self.assertTrue(all(item.modes == ("C2_final",) for item in scenarios))
        labels = {item.label for item in scenarios}
        self.assertTrue({
            "nominal",
            "push_minus_4N",
            "push_plus_4N",
            "friction_mu_0p40",
            "slope_minus_5deg",
            "slope_plus_5deg",
            "step_6mm",
            "command_delay_20ms",
            "sensor_delay_40ms",
            "actuator_gain_minus_20pct",
            "actuator_gain_plus_20pct",
            "actuator_tau_50ms",
            "joint_noise_0p002rad",
            "pitch_noise_0p5deg",
            "combined_A",
            "combined_B",
            "combined_C",
        }.issubset(labels))
        self.assertEqual(
            module.scenario_seeds(scenarios[0], 3),
            module.scenario_seeds(scenarios[0], 3),
        )
        self.assertNotEqual(
            module.scenario_seeds(scenarios[0], 3),
            module.scenario_seeds(scenarios[1], 3),
        )

    def test_wilson_interval_uses_requested_confidence(self):
        module = load_module("test_multistep_robustness_wilson")
        low, high = module.wilson_interval(27, 30, confidence=0.95)

        self.assertLess(low, 0.90)
        self.assertGreater(high, 0.90)
        self.assertGreater(high, low)
        with self.assertRaises(ValueError):
            module.wilson_interval(31, 30)

    def test_aggregation_keeps_failures_and_reports_wilson_bounds(self):
        module = load_module("test_multistep_robustness_aggregate")
        rows = [
            {
                "scenario": "nominal",
                "mode": "C2_final",
                "seed": seed,
                "passed": seed != 2,
                "failure_category": "" if seed != 2 else "pitch_limit",
                "event_match_ratio": 1.0,
            }
            for seed in range(3)
        ]

        aggregates = module.aggregate_trials(rows)
        aggregate = aggregates[0]

        self.assertEqual(aggregate["trial_count"], 3)
        self.assertEqual(aggregate["success_count"], 2)
        self.assertEqual(aggregate["failure_count"], 1)
        self.assertLess(
            aggregate["wilson_95_low"],
            aggregate["success_rate"],
        )
        self.assertGreater(
            aggregate["wilson_95_high"],
            aggregate["success_rate"],
        )
        with self.assertRaises(ValueError):
            module.aggregate_trials([
                {**rows[-1], "failure_category": ""}
            ])

    def test_acceptance_requires_nominal_critical_and_aggregate_gates(self):
        module = load_module("test_multistep_robustness_acceptance")
        aggregates = []
        for scenario in module.build_scenarios():
            aggregates.append({
                "scenario": scenario.label,
                "mode": "C2_final",
                "critical": scenario.critical,
                "trial_count": 30,
                "success_count": 30,
                "failure_count": 0,
                "success_rate": 1.0,
            })

        accepted = module.evaluate_matrix(aggregates)
        self.assertTrue(accepted["passed"])

        rejected_rows = [
            {
                **row,
                "success_count": 26,
                "failure_count": 4,
                "success_rate": 26 / 30,
            }
            if row["scenario"] == "push_plus_4N" else row
            for row in aggregates
        ]
        rejected = module.evaluate_matrix(rejected_rows)
        self.assertFalse(rejected["passed"])
        self.assertIn("critical_scenarios", rejected["failures"])

    def test_phase_aligned_metrics_use_controller_counts(self):
        runner_path = (
            ROOT / "mpc" / "18_validate_fddp_multistep_c2_final.py"
        )
        spec = importlib.util.spec_from_file_location(
            "test_multistep_phase_aligned_metrics", runner_path
        )
        runner = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)
        frame = pd.DataFrame([
            {"matched_events": 17, "skipped_events": 3}
        ])
        raw = {
            "matched_count": 13,
            "scheduled_count": 20,
            "match_ratio": 0.65,
            "event_order_valid": True,
        }

        metrics = runner._event_metrics(
            "C2_final", frame, [{} for _ in range(20)], raw
        )

        self.assertEqual(metrics["matched_contact_events"], 17)
        self.assertEqual(metrics["scheduled_contact_events"], 20)
        self.assertAlmostEqual(metrics["event_match_ratio"], 0.85)
        self.assertAlmostEqual(metrics["raw_event_match_ratio"], 0.65)

    def test_phase_aligned_timing_uses_synchronizer_records(self):
        runner_path = (
            ROOT / "mpc" / "18_validate_fddp_multistep_c2_final.py"
        )
        spec = importlib.util.spec_from_file_location(
            "test_multistep_phase_aligned_timing", runner_path
        )
        runner = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)
        synchronizer = runner.SimpleNamespace(matched=[
            {"timing_error_s": 0.010},
            {"timing_error_s": -0.020},
        ])
        controller = runner.SimpleNamespace(
            _synchronizers={"C2_final": synchronizer}
        )

        metrics = runner._contact_timing_metrics(
            "C2_final", controller, np.asarray([0.050, 0.070])
        )

        self.assertAlmostEqual(metrics["mean_abs_contact_timing_error_s"], 0.060)
        self.assertAlmostEqual(metrics["phase_aligned_mean_abs_contact_timing_error_s"], 0.015)
        self.assertAlmostEqual(metrics["raw_mean_abs_contact_timing_error_s"], 0.060)



    def test_existing_runner_accepts_all_robustness_overrides(self):
        runner_path = (
            ROOT / "mpc" / "18_validate_fddp_multistep_c2_final.py"
        )
        spec = importlib.util.spec_from_file_location(
            "test_multistep_robustness_runner", runner_path
        )
        runner = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)

        result = runner._run_mode(
            "C2_final",
            0.10,
            friction_mu=0.50,
            slope_deg=3.0,
            step_height_m=0.004,
            command_delay_s=0.012,
            sensor_delay_s=0.020,
            actuator_gain=0.80,
            actuator_time_constant_s=0.050,
            initial_pitch_noise_std_rad=0.01,
        )

        configured = result["metrics"]["robustness_configuration"]
        self.assertEqual(configured["friction_mu"], 0.50)
        self.assertEqual(configured["sensor_delay_s"], 0.020)
        self.assertEqual(configured["actuator_gain"], 0.80)
        self.assertIn("sensor_measurement_time_s", result["frame"].columns)

    def test_scenario_model_changes_friction_gravity_and_step_geometry(self):
        runner_path = (
            ROOT / "mpc" / "18_validate_fddp_multistep_c2_final.py"
        )
        spec = importlib.util.spec_from_file_location(
            "test_multistep_robustness_model", runner_path
        )
        runner = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)

        model = runner._build_robustness_model(
            friction_mu=0.40,
            slope_deg=-3.0,
            step_height_m=0.004,
        )

        floor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        step_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "stage9_left_toe_step",
        )
        self.assertGreaterEqual(step_id, 0)
        self.assertAlmostEqual(model.geom_friction[floor_id, 0], 0.40)
        self.assertFalse(np.isclose(model.opt.gravity[1], 0.0))

    def test_batch_runner_reuses_trial_gate_and_keeps_every_seed(self):
        script_path = (
            ROOT / "mpc" / "21_run_fddp_multistep_robustness_matrix.py"
        )
        spec = importlib.util.spec_from_file_location(
            "test_multistep_robustness_batch", script_path
        )
        batch = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = batch
        spec.loader.exec_module(batch)
        scenario = batch.statistics.build_scenarios()[0]

        report = batch.run_matrix(
            scenarios=[scenario],
            seed_count=1,
            duration_s=0.10,
            write_outputs=False,
        )

        self.assertEqual(len(report["trial_rows"]), 1)
        self.assertEqual(len(report["aggregates"]), 1)
        self.assertEqual(report["trial_rows"][0]["scenario"], "nominal")
        self.assertIn("failure_category", report["trial_rows"][0])
        self.assertEqual(report["aggregates"][0]["trial_count"], 1)
    def test_terrain_contact_fallback_recognises_both_feet(self):
        runner_path = (
            ROOT / "mpc" / "18_validate_fddp_multistep_c2_final.py"
        )
        spec = importlib.util.spec_from_file_location(
            "test_multistep_terrain_contact_fallback", runner_path
        )
        runner = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)
        model = runner._build_robustness_model(step_height_m=0.004)
        data = mujoco.MjData(model)
        controller = runner.controller_runtime.MultiStepController()
        mapping = runner.proxy.resolve_joint_and_actuator_mapping(model)
        for address, value in zip(
            mapping["qpos_addresses"], controller.reference.q_ref[0]
        ):
            data.qpos[address] = value
        mujoco.mj_forward(model, data)

        contact_ids = runner._resolve_foot_terrain_contact_ids(model)
        active = runner._geometry_leg_contacts(data, contact_ids)

        self.assertTrue(active["left"])
        self.assertTrue(active["right"])
if __name__ == "__main__":
    unittest.main()
