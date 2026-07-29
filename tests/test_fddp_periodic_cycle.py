import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pinocchio as pin
import yaml


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "fddp" / "05a_solve_fddp_periodic_cycle.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fddp_periodic_cycle", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def identity_state() -> np.ndarray:
    x = np.zeros(25, dtype=float)
    x[2] = 0.31
    x[6] = 1.0
    return x


def translated_pose(forward_y: float) -> pin.SE3:
    pose = pin.SE3.Identity()
    pose.translation[1] = forward_y
    return pose


class PeriodicCycleReferenceTests(unittest.TestCase):
    def test_periodic_state_reference_translates_base_without_joint_reset(self):
        module = load_module()
        x0 = identity_state()
        references = module.build_periodic_state_references(
            x0,
            nq=13,
            horizon_nodes=200,
            planned_forward_y=0.010,
        )

        self.assertEqual(references.shape, (201, 25))
        np.testing.assert_allclose(references[0], x0)
        self.assertAlmostEqual(references[-1, 1] - references[0, 1], 0.010)
        np.testing.assert_allclose(references[-1, 7:13], references[0, 7:13])
        np.testing.assert_allclose(references[-1, 13:], 0.0)

    def test_periodic_problem_keeps_contact_switches_and_translated_terminal_target(self):
        module = load_module()
        model = pin.buildModelFromUrdf(
            str(module.URDF_PATH), pin.JointModelFreeFlyer()
        )
        x0 = module.load_x0(module.X0_PATH, model.nq + model.nv)
        problem, _, metadata = module.build_periodic_cycle_problem(model, x0)

        def contact_names(action_model):
            return set(action_model.differential.contacts.contacts.todict())

        self.assertEqual(
            contact_names(problem.runningModels[20]), {module.RIGHT_CONTACT_NAME}
        )
        self.assertEqual(
            contact_names(problem.runningModels[110]), {module.LEFT_CONTACT_NAME}
        )
        self.assertEqual(metadata["seed_provenance"], "accepted_fddp_two_step")
        self.assertAlmostEqual(
            metadata["state_references"][-1, 1]
            - metadata["state_references"][0, 1],
            module.PLANNED_CYCLE_FORWARD_Y_M,
        )
        terminal_cost = problem.terminalModel.differential.costs.costs.todict()[
            "static_x0_state_regularisation"
        ].cost
        np.testing.assert_allclose(
            terminal_cost.residual.reference,
            metadata["state_references"][-1],
        )

    def test_periodic_problem_uses_sole_centered_contact_frames(self):
        module = load_module()
        model = pin.buildModelFromUrdf(
            str(module.URDF_PATH), pin.JointModelFreeFlyer()
        )
        x0 = module.load_x0(module.X0_PATH, model.nq + model.nv)
        problem, _, metadata = module.build_periodic_cycle_problem(model, x0)

        self.assertEqual(
            model.frames[metadata["left_frame_id"]].name,
            "left_sole_contact",
        )
        self.assertEqual(
            model.frames[metadata["right_frame_id"]].name,
            "right_sole_contact",
        )
        expected_offset = np.array([0.0, 0.0, -0.043])
        np.testing.assert_allclose(
            model.frames[metadata["left_frame_id"]].placement.translation,
            expected_offset,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            model.frames[metadata["right_frame_id"]].placement.translation,
            expected_offset,
            atol=1.0e-12,
        )

        left_contact = (
            problem.runningModels[0]
            .differential.contacts.contacts.todict()[module.LEFT_CONTACT_NAME]
        )
        right_contact = (
            problem.runningModels[0]
            .differential.contacts.contacts.todict()[module.RIGHT_CONTACT_NAME]
        )
        self.assertEqual(left_contact.contact.id, metadata["left_frame_id"])
        self.assertEqual(right_contact.contact.id, metadata["right_frame_id"])

    def test_periodic_problem_adds_planar_resultant_cone_cost(self):
        module = load_module()
        model = pin.buildModelFromUrdf(
            str(module.URDF_PATH), pin.JointModelFreeFlyer()
        )
        x0 = module.load_x0(module.X0_PATH, model.nq + model.nv)
        problem, _, metadata = module.build_periodic_cycle_problem(model, x0)
        cost_name = "planar_support_resultant_cone"

        ds_costs = problem.runningModels[0].differential.costs.costs.todict()
        ss_costs = problem.runningModels[20].differential.costs.costs.todict()
        self.assertIn(cost_name, ds_costs)
        self.assertIn(cost_name, ss_costs)
        self.assertEqual(
            set(ds_costs[cost_name].cost.residual.active_contact_names),
            {module.LEFT_CONTACT_NAME, module.RIGHT_CONTACT_NAME},
        )
        self.assertEqual(
            tuple(ss_costs[cost_name].cost.residual.active_contact_names),
            (module.RIGHT_CONTACT_NAME,),
        )
        self.assertEqual(
            metadata["contact_feasibility_mode"],
            "planar_sagittal_resultant",
        )

    def test_periodic_planar_cone_force_jacobian_matches_control_finite_difference(self):
        module = load_module()
        model = pin.buildModelFromUrdf(
            str(module.URDF_PATH), pin.JointModelFreeFlyer()
        )
        x0 = module.load_x0(module.X0_PATH, model.nq + model.nv)
        problem, _, _ = module.build_periodic_cycle_problem(
            model,
            x0,
            terminal_state_weight=1.0,
            boundary_control_weight=0.0,
        )
        xs_seed, us_seed, _ = module.build_periodic_seed(model, x0)

        node = 2
        cost_name = "planar_support_resultant_cone"
        action = problem.runningModels[node]
        self.assertIn(cost_name, action.differential.costs.costs.todict())
        x = xs_seed[node].copy()
        u = us_seed[node].copy()
        data = action.createData()
        action.calc(data, x, u)
        action.calcDiff(data, x, u)
        analytic = np.asarray(
            data.differential.costs.costs.todict()[cost_name].residual.Ru,
            dtype=float,
        ).copy()

        epsilon = 1.0e-6
        finite_difference = np.zeros_like(analytic)
        for column in range(u.size):
            u_plus = u.copy()
            u_minus = u.copy()
            u_plus[column] += epsilon
            u_minus[column] -= epsilon
            plus_data = action.createData()
            minus_data = action.createData()
            action.calc(plus_data, x, u_plus)
            action.calc(minus_data, x, u_minus)
            plus = np.asarray(
                plus_data.differential.costs.costs.todict()[cost_name].residual.r,
                dtype=float,
            )
            minus = np.asarray(
                minus_data.differential.costs.costs.todict()[cost_name].residual.r,
                dtype=float,
            )
            finite_difference[:, column] = (plus - minus) / (2.0 * epsilon)

        self.assertGreater(np.linalg.norm(finite_difference), 1.0)
        self.assertGreater(np.linalg.norm(analytic), 1.0)
        np.testing.assert_allclose(
            analytic,
            finite_difference,
            rtol=2.0e-4,
            atol=2.0e-5,
        )

    def test_periodic_planar_cone_records_stricter_internal_cop_margin(self):
        module = load_module()
        model = pin.buildModelFromUrdf(
            str(module.URDF_PATH), pin.JointModelFreeFlyer()
        )
        x0 = module.load_x0(module.X0_PATH, model.nq + model.nv)
        problem, _, metadata = module.build_periodic_cycle_problem(model, x0)

        self.assertIn("optimisation_cop_margin_m", metadata)
        self.assertGreater(
            metadata["optimisation_cop_margin_m"],
            metadata["minimum_cop_margin_m"],
        )
        residual = (
            problem.runningModels[0]
            .differential.costs.costs.todict()["planar_support_resultant_cone"]
            .cost.residual
        )
        self.assertAlmostEqual(
            residual.inner_support_min_y_m - residual.support_min_y_m,
            metadata["optimisation_cop_margin_m"],
        )

    def test_periodic_seed_uses_only_accepted_two_step_and_preserves_feasibility(self):
        module = load_module()
        model = pin.buildModelFromUrdf(
            str(module.URDF_PATH), pin.JointModelFreeFlyer()
        )
        x0 = module.load_x0(module.X0_PATH, model.nq + model.nv)
        xs_seed, us_seed, provenance = module.build_periodic_seed(model, x0)
        problem, state, _ = module.build_periodic_cycle_problem(
            model,
            x0,
            terminal_state_weight=1.0,
            boundary_control_weight=0.0,
        )
        with np.load(module.TWO_STEP_TRAJECTORY_PATH, allow_pickle=False) as loaded:
            source_xs = np.asarray(loaded["state_ref"], dtype=float)
            source_us = np.asarray(loaded["tau_ref"], dtype=float)

        self.assertEqual(xs_seed.shape, (201, model.nq + model.nv))
        self.assertEqual(us_seed.shape, (200, model.nv - 6))
        np.testing.assert_allclose(xs_seed, source_xs)
        np.testing.assert_allclose(us_seed, source_us)
        problem.calc([x.copy() for x in xs_seed], [u.copy() for u in us_seed])
        gaps = [
            np.linalg.norm(state.diff(xs_seed[k + 1], data.xnext))
            for k, data in enumerate(problem.runningDatas)
        ]
        self.assertLessEqual(max(gaps), 1.0e-7)
        self.assertEqual(provenance["source"], "accepted_fddp_two_step")
        self.assertNotIn("scripted", str(provenance).lower())

    def test_homotopy_adds_symmetric_boundary_control_cost_after_state_stages(self):
        module = load_module()
        stages = module.build_periodic_homotopy_stages()
        self.assertEqual(
            [
                (stage.terminal_state_weight, stage.boundary_control_weight)
                for stage in stages
            ],
            [
                (1.0, 0.0),
                (10.0, 0.0),
                (100.0, 0.0),
                (1000.0, 0.0),
                (1000.0, 1.0),
                (1000.0, 10.0),
                (1000.0, 30.0),
            ],
        )

        model = pin.buildModelFromUrdf(
            str(module.URDF_PATH), pin.JointModelFreeFlyer()
        )
        x0 = module.load_x0(module.X0_PATH, model.nq + model.nv)
        problem, _, metadata = module.build_periodic_cycle_problem(
            model,
            x0,
            terminal_state_weight=1000.0,
            boundary_control_weight=30.0,
        )

        def cost_names(action_model):
            return set(action_model.differential.costs.costs.todict())

        self.assertIn(
            module.PERIODIC_BOUNDARY_CONTROL_COST_NAME,
            cost_names(problem.runningModels[0]),
        )
        self.assertNotIn(
            module.PERIODIC_BOUNDARY_CONTROL_COST_NAME,
            cost_names(problem.runningModels[1]),
        )
        self.assertIn(
            module.PERIODIC_BOUNDARY_CONTROL_COST_NAME,
            cost_names(problem.runningModels[199]),
        )
        self.assertEqual(metadata["terminal_state_weight"], 1000.0)
        self.assertEqual(metadata["boundary_control_weight"], 30.0)

    def test_one_iteration_solver_smoke_returns_gated_artifacts(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result = module.solve_periodic_cycle(
                max_iterations=1,
                output_dir=root / "logs",
                trajectory_dir=root / "trajectories",
            )

            self.assertEqual(
                result["exported"], result["acceptance"]["passed"]
            )
            self.assertIn("solver_stop", result["diagnostics"])
            self.assertIn("shooting_gap_norm", result["diagnostics"])
            self.assertIn("contact_violation_count", result["diagnostics"])
            self.assertIn(
                "maximum_friction_utilisation", result["diagnostics"]
            )
            self.assertIn("minimum_cop_margin_m", result["diagnostics"])
            self.assertIn(
                "maximum_tangential_speed_m_s", result["diagnostics"]
            )
            self.assertTrue(result["outputs"]["diagnostic_yaml"].is_file())
            self.assertTrue(result["outputs"]["iteration_csv"].is_file())
            if not result["acceptance"]["passed"]:
                self.assertFalse(result["outputs"]["trajectory_npz"].exists())


class PeriodicCycleBoundaryTests(unittest.TestCase):
    def passing_metrics(self, module) -> dict:
        x_start = identity_state()
        x_terminal = x_start.copy()
        x_terminal[1] += 0.010
        start_pose = pin.SE3.Identity()
        terminal_pose = translated_pose(0.010)
        metrics = module.compute_cycle_boundary_metrics(
            x_start=x_start,
            x_terminal=x_terminal,
            u_start=np.zeros(6),
            u_terminal=np.full(6, 0.10),
            left_start=start_pose,
            left_terminal=terminal_pose,
            right_start=start_pose,
            right_terminal=terminal_pose,
            com_start=np.array([0.0, 0.0, 0.25]),
            com_terminal=np.array([0.0, 0.010, 0.25]),
            com_velocity_start=np.zeros(3),
            com_velocity_terminal=np.zeros(3),
            planned_forward_y=0.010,
            nq=13,
            start_contact_mode="DS",
            terminal_contact_mode="DS",
        )
        metrics.update(
            {
                "solver_stop": 1.0e-7,
                "shooting_gap_norm": 1.0e-7,
                "active_support_translation_error_m": 5.0e-4,
                "active_support_rotation_error_rad": np.deg2rad(0.5),
                "landing_translation_error_m": 1.0e-3,
                "landing_rotation_error_rad": np.deg2rad(1.0),
                "swing_clearance_m": 0.010,
                "joint_limit_margin_rad": 0.010,
                "contact_support_node_count": module.HORIZON_NODES,
                "contact_violation_count": 0,
                "minimum_contact_normal_force_n": 1.0,
                "maximum_friction_utilisation": 0.90,
                "minimum_cop_margin_m": 0.002,
                "maximum_tangential_speed_m_s": 0.001,
            }
        )
        return metrics

    def test_exact_cycle_boundaries_pass_after_forward_translation_removal(self):
        module = load_module()
        metrics = self.passing_metrics(module)

        self.assertAlmostEqual(metrics["base_translation_residual_m"], 0.0)
        self.assertAlmostEqual(metrics["com_translation_residual_m"], 0.0)
        self.assertAlmostEqual(metrics["relative_foot_translation_error_m"], 0.0)
        acceptance = module.evaluate_cycle_acceptance(metrics)
        self.assertTrue(acceptance["passed"])
        self.assertEqual(acceptance["failures"], [])

    def test_contact_feasibility_failure_withholds_cycle_acceptance(self):
        module = load_module()
        metrics = self.passing_metrics(module)
        metrics.update(
            {
                "contact_support_node_count": module.HORIZON_NODES,
                "contact_violation_count": 1,
                "minimum_contact_normal_force_n": 10.0,
                "maximum_friction_utilisation": 0.91,
                "minimum_cop_margin_m": 0.0019,
                "maximum_tangential_speed_m_s": 0.0,
            }
        )

        acceptance = module.evaluate_cycle_acceptance(metrics)

        self.assertFalse(acceptance["passed"])
        self.assertIn("contact_violation_count", acceptance["failures"])
        self.assertIn("maximum_friction_utilisation", acceptance["failures"])
        self.assertIn("minimum_cop_margin_m", acceptance["failures"])

    def test_each_periodic_or_planning_gate_can_withhold_acceptance(self):
        module = load_module()
        passing = self.passing_metrics(module)
        failing_values = {
            "joint_position_difference_rad": 1.01e-3,
            "joint_velocity_difference_rad_s": 1.01e-2,
            "base_height_difference_m": 1.01e-3,
            "base_pitch_difference_rad": 1.01e-3,
            "com_velocity_difference_m_s": 1.01e-3,
            "relative_foot_translation_error_m": 1.01e-3,
            "relative_foot_rotation_error_rad": np.deg2rad(0.201),
            "control_discontinuity": 0.101,
            "solver_stop": 1.01e-7,
            "shooting_gap_norm": 1.01e-7,
            "active_support_translation_error_m": 5.01e-4,
            "active_support_rotation_error_rad": np.deg2rad(0.501),
            "landing_translation_error_m": 1.01e-3,
            "landing_rotation_error_rad": np.deg2rad(1.01),
            "swing_clearance_m": 0.0099,
            "joint_limit_margin_rad": 0.0099,
            "contact_support_node_count": module.HORIZON_NODES - 1,
            "contact_violation_count": 1,
            "minimum_contact_normal_force_n": -1.1e-5,
            "maximum_friction_utilisation": 0.9001,
            "minimum_cop_margin_m": 0.0019,
            "maximum_tangential_speed_m_s": 0.00101,
        }
        for name, bad_value in failing_values.items():
            with self.subTest(name=name):
                rejected = module.evaluate_cycle_acceptance(
                    {**passing, name: bad_value}
                )
                self.assertFalse(rejected["passed"])
                self.assertIn(name, rejected["failures"])

        rejected_contact = module.evaluate_cycle_acceptance(
            {**passing, "contact_mode_compatible": False}
        )
        self.assertFalse(rejected_contact["passed"])
        self.assertIn("contact_mode_compatible", rejected_contact["failures"])

    def test_failed_cycle_writes_diagnostics_but_never_nominal_npz(self):
        module = load_module()
        metrics = self.passing_metrics(module)
        metrics["solver_stop"] = 1.0e-3
        acceptance = module.evaluate_cycle_acceptance(metrics)

        with tempfile.TemporaryDirectory() as temporary_directory:
            outputs = module.write_cycle_artifacts(
                {
                    "diagnostics": metrics,
                    "acceptance": acceptance,
                    "iteration_history": [
                        {"iteration": 0, "cost": 10.0, "stop": 1.0}
                    ],
                },
                accepted=False,
                output_dir=Path(temporary_directory),
            )
            self.assertTrue(outputs["diagnostic_yaml"].is_file())
            self.assertTrue(outputs["iteration_csv"].is_file())
            self.assertFalse(outputs["trajectory_npz"].exists())
            summary = yaml.safe_load(outputs["diagnostic_yaml"].read_text())
            self.assertFalse(summary["exported_nominal_trajectory"])
            self.assertFalse(summary["acceptance"]["passed"])

    def test_existing_two_step_is_feasible_but_not_periodic_boundary_evidence(self):
        module = load_module()
        model = pin.buildModelFromUrdf(
            str(module.URDF_PATH), pin.JointModelFreeFlyer()
        )
        x0 = module.load_x0(module.X0_PATH, model.nq + model.nv)
        problem, state, metadata = module.build_periodic_cycle_problem(model, x0)
        with np.load(module.TWO_STEP_TRAJECTORY_PATH, allow_pickle=False) as loaded:
            xs = np.asarray(loaded["state_ref"], dtype=float)
            us = np.asarray(loaded["tau_ref"], dtype=float)
        solver = SimpleNamespace(iter=166, stop=2.84e-8, cost=0.56)

        diagnostics, com = module.compute_periodic_diagnostics(
            model=model,
            problem=problem,
            state=state,
            xs=xs,
            us=us,
            metadata=metadata,
            solver=solver,
        )
        self.assertEqual(com.shape, (201, 3))
        self.assertLessEqual(diagnostics["shooting_gap_norm"], 1.0e-7)
        self.assertGreater(diagnostics["joint_position_difference_rad"], 1.0e-3)
        self.assertGreater(diagnostics["base_pitch_difference_rad"], 1.0e-3)
        self.assertFalse(module.evaluate_cycle_acceptance(diagnostics)["passed"])


if __name__ == "__main__":
    unittest.main()
