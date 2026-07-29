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
MODULE_PATH = ROOT / "fddp" / "05b_solve_fddp_multistep_continuation.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "fddp_multistep_continuation", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ContinuationContractTests(unittest.TestCase):
    def test_ladder_and_seed_provenance_forbid_scripted_warm_starts(self):
        module = load_module()
        self.assertEqual(module.continuation_ladder(10), (2, 4, 6, 10))
        self.assertEqual(module.required_seed_for(4), "accepted_fddp_periodic_2step")
        self.assertEqual(module.required_seed_for(6), "accepted_fddp_4step")
        self.assertEqual(module.required_seed_for(10), "accepted_fddp_6step")

        valid = {
            "source": "accepted_fddp_4step",
            "source_step_count": 4,
            "accepted": True,
            "tau_ref_hardware_usage": "analysis_only",
        }
        module.validate_seed_metadata(valid, target_steps=6)
        with self.assertRaises(module.SeedIntegrityError):
            module.validate_seed_metadata(
                {**valid, "source": "stage8_scripted"}, target_steps=6
            )
        with self.assertRaises(module.SeedIntegrityError):
            module.validate_seed_metadata(
                {**valid, "accepted": False}, target_steps=6
            )

    def test_control_seed_extends_only_with_accepted_periodic_fddp_controls(self):
        module = load_module()
        periodic = np.arange(200 * 6, dtype=float).reshape(200, 6)
        four_step = module.extend_control_seed(
            previous_us=periodic,
            previous_steps=2,
            target_steps=4,
            periodic_us=periodic,
        )
        self.assertEqual(four_step.shape, (400, 6))
        np.testing.assert_allclose(four_step[:200], periodic)
        np.testing.assert_allclose(four_step[200:], periodic)

        six_step = module.extend_control_seed(
            previous_us=four_step,
            previous_steps=4,
            target_steps=6,
            periodic_us=periodic,
        )
        ten_step = module.extend_control_seed(
            previous_us=six_step,
            previous_steps=6,
            target_steps=10,
            periodic_us=periodic,
        )
        self.assertEqual(six_step.shape, (600, 6))
        self.assertEqual(ten_step.shape, (1000, 6))

    def test_segment_gate_uses_inclusive_thresholds_and_final_duration(self):
        module = load_module()
        metrics = {
            "solver_stop": 1.0e-7,
            "shooting_gap_norm": 1.0e-7,
            "active_support_translation_error_m": 5.0e-4,
            "active_support_rotation_error_rad": np.deg2rad(0.5),
            "landing_translation_error_m": 1.0e-3,
            "landing_rotation_error_rad": np.deg2rad(1.0),
            "swing_clearance_m": 0.010,
            "joint_limit_margin_rad": 0.010,
            "forward_displacement_m": 0.040,
            "duration_s": 10.0,
            "contact_support_node_count": 1000,
            "contact_violation_count": 0,
            "minimum_contact_normal_force_n": 10.0,
            "maximum_friction_utilisation": 0.90,
            "minimum_cop_margin_m": 0.002,
            "maximum_tangential_speed_m_s": 0.001,
        }
        self.assertTrue(module.evaluate_segment(metrics, step_count=10)["passed"])
        rejected = module.evaluate_segment(
            {**metrics, "duration_s": 9.999}, step_count=10
        )
        self.assertFalse(rejected["passed"])
        self.assertIn("duration_s", rejected["failures"])
        rejected_contact = module.evaluate_segment(
            {**metrics, "maximum_friction_utilisation": 0.9001},
            step_count=10,
        )
        self.assertFalse(rejected_contact["passed"])
        self.assertIn(
            "maximum_friction_utilisation", rejected_contact["failures"]
        )

    def test_solver_configuration_keeps_gate_and_allows_high_regularization(self):
        module = load_module()
        solver = SimpleNamespace(th_stop=None, reg_max=1.0e8)
        module.configure_solver(solver)
        self.assertEqual(solver.th_stop, 1.0e-7)
        self.assertEqual(solver.reg_max, 1.0e14)



class ContinuationProblemTests(unittest.TestCase):
    def test_sole_contact_frames_are_idempotent_and_match_mjcf_ground_plane(self):
        module = load_module()
        model = pin.buildModelFromUrdf(
            str(module.URDF_PATH), pin.JointModelFreeFlyer()
        )
        left_id, right_id = module.ensure_sole_contact_frames(model)
        again = module.ensure_sole_contact_frames(model)
        self.assertEqual((left_id, right_id), again)
        self.assertEqual(model.frames[left_id].name, module.LEFT_SOLE_FRAME)
        self.assertEqual(model.frames[right_id].name, module.RIGHT_SOLE_FRAME)
        np.testing.assert_allclose(
            model.frames[left_id].placement.translation, [0.0, 0.0, -0.043]
        )
        np.testing.assert_allclose(
            model.frames[right_id].placement.translation, [0.0, 0.0, -0.043]
        )

    def test_active_contacts_share_one_planar_resultant_cone_cost(self):
        module = load_module()
        model = pin.buildModelFromUrdf(
            str(module.URDF_PATH), pin.JointModelFreeFlyer()
        )
        x0 = module.load_x0(module.X0_PATH, model.nq + model.nv)
        problem, _, metadata = module.build_multistep_problem(
            model, x0, step_count=4
        )

        def costs(node):
            return problem.runningModels[node].differential.costs.costs.todict()

        ds_costs = costs(0)
        ss_costs = costs(20)
        self.assertIn(module.PLANAR_SUPPORT_CONE_COST_NAME, ds_costs)
        self.assertIn(module.PLANAR_SUPPORT_CONE_COST_NAME, ss_costs)
        self.assertFalse(
            any("contact_force_regularisation" in name for name in ds_costs)
        )
        self.assertFalse(
            any("contact_wrench_cone" in name for name in ds_costs)
        )
        ds_residual = ds_costs[module.PLANAR_SUPPORT_CONE_COST_NAME].cost.residual
        ss_residual = ss_costs[module.PLANAR_SUPPORT_CONE_COST_NAME].cost.residual
        self.assertIsInstance(ds_residual, module.ResidualModelPlanarSupportCone)
        self.assertEqual(ds_residual.nr, 5)
        self.assertEqual(set(ds_residual.active_contact_names), {
            module.LEFT_CONTACT_NAME,
            module.RIGHT_CONTACT_NAME,
        })
        self.assertEqual(
            tuple(ss_residual.active_contact_names), (module.RIGHT_CONTACT_NAME,)
        )
        action = problem.runningModels[0]
        action_data = action.createData()
        action.calc(action_data, x0, np.zeros(action.differential.actuation.nu))
        self.assertEqual(metadata["contact_feasibility_mode"], "planar_sagittal_resultant")
        self.assertEqual(metadata["left_frame_id"], model.getFrameId(module.LEFT_SOLE_FRAME))
        self.assertEqual(metadata["right_frame_id"], model.getFrameId(module.RIGHT_SOLE_FRAME))

    def test_contact_weight_homotopy_and_internal_cop_margin_are_explicit(self):
        module = load_module()
        expected_weights = {4: 1.0e3, 6: 1.0e4, 10: 1.0e5}
        for step_count, expected_weight in expected_weights.items():
            with self.subTest(step_count=step_count):
                model = pin.buildModelFromUrdf(
                    str(module.URDF_PATH), pin.JointModelFreeFlyer()
                )
                x0 = module.load_x0(module.X0_PATH, model.nq + model.nv)
                problem, _, metadata = module.build_multistep_problem(
                    model, x0, step_count=step_count
                )
                self.assertEqual(
                    metadata["planar_support_cone_weight"], expected_weight
                )
                self.assertEqual(metadata["minimum_cop_margin_m"], 0.002)
                self.assertEqual(metadata["optimisation_cop_margin_m"], 0.0027)
                item = (
                    problem.runningModels[0]
                    .differential.costs.costs.todict()[
                        module.PLANAR_SUPPORT_CONE_COST_NAME
                    ]
                )
                self.assertEqual(item.weight, expected_weight)
                residual = item.cost.residual
                self.assertAlmostEqual(
                    residual.inner_support_min_y_m - residual.support_min_y_m,
                    0.0027,
                )

    def test_planar_cone_force_jacobian_matches_control_finite_difference(self):
        module = load_module()
        model = pin.buildModelFromUrdf(
            str(module.URDF_PATH), pin.JointModelFreeFlyer()
        )
        x0 = module.load_x0(module.X0_PATH, model.nq + model.nv)
        problem, state, _ = module.build_multistep_problem(
            model, x0, step_count=4
        )
        xs_seed, us_seed, _ = module.build_continuation_seed(
            problem=problem,
            state=state,
            target_steps=4,
            source_path=module.PERIODIC_TRAJECTORY_PATH,
        )

        node = 2
        action = problem.runningModels[node]
        x = xs_seed[node].copy()
        u = us_seed[node].copy()
        data = action.createData()
        action.calc(data, x, u)
        action.calcDiff(data, x, u)
        cost_data = data.differential.costs.costs.todict()[
            module.PLANAR_SUPPORT_CONE_COST_NAME
        ]
        analytic = np.asarray(cost_data.residual.Ru, dtype=float).copy()

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
                plus_data.differential.costs.costs.todict()[
                    module.PLANAR_SUPPORT_CONE_COST_NAME
                ].residual.r,
                dtype=float,
            )
            minus = np.asarray(
                minus_data.differential.costs.costs.todict()[
                    module.PLANAR_SUPPORT_CONE_COST_NAME
                ].residual.r,
                dtype=float,
            )
            finite_difference[:, column] = (plus - minus) / (2.0 * epsilon)

        self.assertGreater(np.linalg.norm(finite_difference), 1.0)
        self.assertGreater(np.linalg.norm(analytic), 1.0)
        np.testing.assert_allclose(
            analytic, finite_difference, rtol=2.0e-4, atol=2.0e-5
        )


    def test_four_step_problem_repeats_contacts_and_translated_cycle_references(self):
        module = load_module()
        model = pin.buildModelFromUrdf(
            str(module.URDF_PATH), pin.JointModelFreeFlyer()
        )
        x0 = module.load_x0(module.X0_PATH, model.nq + model.nv)
        problem, _, metadata = module.build_multistep_problem(
            model, x0, step_count=4
        )

        def contact_names(action_model):
            return set(action_model.differential.contacts.contacts.todict())

        self.assertEqual(problem.T, 400)
        self.assertEqual(contact_names(problem.runningModels[20]), {module.RIGHT_CONTACT_NAME})
        self.assertEqual(contact_names(problem.runningModels[110]), {module.LEFT_CONTACT_NAME})
        self.assertEqual(contact_names(problem.runningModels[220]), {module.RIGHT_CONTACT_NAME})
        self.assertEqual(contact_names(problem.runningModels[310]), {module.LEFT_CONTACT_NAME})
        self.assertAlmostEqual(
            metadata["state_references"][-1, 1]
            - metadata["state_references"][0, 1],
            0.020,
        )
        self.assertEqual(metadata["cycle_id_by_node"][199], 0)
        self.assertEqual(metadata["cycle_id_by_node"][200], 1)

    def test_four_step_seed_stitches_finite_blocks_with_bounded_boundary_gap(self):
        module = load_module()
        model = pin.buildModelFromUrdf(
            str(module.URDF_PATH), pin.JointModelFreeFlyer()
        )
        x0 = module.load_x0(module.X0_PATH, model.nq + model.nv)
        problem, state, _ = module.build_multistep_problem(model, x0, step_count=4)
        xs_seed, us_seed, provenance = module.build_continuation_seed(
            problem=problem,
            state=state,
            target_steps=4,
            source_path=module.PERIODIC_TRAJECTORY_PATH,
        )

        self.assertEqual(xs_seed.shape, (401, model.nq + model.nv))
        self.assertEqual(us_seed.shape, (400, model.nv - 6))
        self.assertTrue(np.all(np.isfinite(xs_seed)))
        self.assertTrue(np.all(np.isfinite(us_seed)))
        problem.calc([x.copy() for x in xs_seed], [u.copy() for u in us_seed])
        gaps = [
            np.linalg.norm(state.diff(xs_seed[k + 1], data.xnext))
            for k, data in enumerate(problem.runningDatas)
        ]
        self.assertLessEqual(max(gaps), 1.0e-3)
        self.assertGreater(max(gaps), 1.0e-7)
        self.assertEqual(int(np.argmax(gaps)), 199)
        with np.load(module.PERIODIC_TRAJECTORY_PATH, allow_pickle=False) as archive:
            expected_seam = np.asarray(archive["state_ref"][0], dtype=float).copy()
        expected_seam[1] += module.STEP_FORWARD_Y_M
        np.testing.assert_allclose(xs_seed[200], expected_seam, atol=1.0e-12)
        self.assertEqual(provenance["source"], "accepted_fddp_periodic_2step")
        self.assertEqual(
            provenance["seed_method"],
            "accepted_fddp_periodic_start_seam_stitch",
        )
        self.assertEqual(
            provenance["seam_state_policy"], "translated_periodic_start"
        )

    def test_ten_step_seed_repairs_every_added_periodic_cycle_seam(self):
        module = load_module()
        model = pin.buildModelFromUrdf(
            str(module.URDF_PATH), pin.JointModelFreeFlyer()
        )
        x0 = module.load_x0(module.X0_PATH, model.nq + model.nv)
        problem, state, _ = module.build_multistep_problem(
            model, x0, step_count=10
        )
        source_path = (
            module.PROJECT_ROOT
            / "transfer"
            / "trajectories"
            / "fddp_multistep_6step_accepted_seed.npz"
        )
        xs_seed, us_seed, _ = module.build_continuation_seed(
            problem=problem,
            state=state,
            target_steps=10,
            source_path=source_path,
        )

        self.assertEqual(xs_seed.shape, (1001, model.nq + model.nv))
        self.assertEqual(us_seed.shape, (1000, model.nv - 6))
        problem.calc([x.copy() for x in xs_seed], [u.copy() for u in us_seed])
        gaps = np.asarray(
            [
                np.linalg.norm(state.diff(xs_seed[k + 1], data.xnext))
                for k, data in enumerate(problem.runningDatas)
            ]
        )
        self.assertLessEqual(float(np.max(gaps)), 1.0e-3)
        self.assertLessEqual(float(gaps[599]), 1.0e-3)
        self.assertLessEqual(float(gaps[799]), 1.0e-3)
        self.assertLessEqual(float(gaps[800]), 1.0e-7)

        with np.load(module.PERIODIC_TRAJECTORY_PATH, allow_pickle=False) as archive:
            periodic_start = np.asarray(archive["state_ref"][0], dtype=float)
        expected_first = periodic_start.copy()
        expected_first[1] += 3 * module.STEP_FORWARD_Y_M
        expected_second = periodic_start.copy()
        expected_second[1] += 4 * module.STEP_FORWARD_Y_M
        np.testing.assert_allclose(xs_seed[600], expected_first, atol=1.0e-12)
        np.testing.assert_allclose(xs_seed[800], expected_second, atol=1.0e-12)

    def test_planning_candidate_never_creates_final_reference(self):
        module = load_module()
        trajectory = {
            "schema_version": np.asarray("fddp-multistep-v1"),
            "acceptance_state": np.asarray(
                "planning_accepted_contact_pending"
            ),
            "tau_ref_hardware_usage": np.asarray("analysis_only"),
            "state_ref": np.zeros((1001, 25)),
            "tau_ref": np.zeros((1000, 6)),
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidate = root / "fddp_multistep_10step_planning_candidate.npz"
            final = root / "fddp_multistep_10step_ref.npz"
            summary = root / "candidate_summary.yaml"
            exported = module.gated_export_planning_candidate(
                trajectory=trajectory,
                acceptance={"passed": True, "failures": []},
                candidate_path=candidate,
                summary_path=summary,
                final_reference_path=final,
            )
            self.assertTrue(exported)
            self.assertTrue(candidate.is_file())
            self.assertTrue(summary.is_file())
            self.assertFalse(final.exists())
            summary_data = yaml.safe_load(summary.read_text())
            self.assertEqual(
                summary_data["acceptance_state"],
                "planning_accepted_contact_pending",
            )

    def test_four_step_diagnostics_cover_every_landing_and_swing(self):
        module = load_module()
        model = pin.buildModelFromUrdf(
            str(module.URDF_PATH), pin.JointModelFreeFlyer()
        )
        x0 = module.load_x0(module.X0_PATH, model.nq + model.nv)
        problem, state, metadata = module.build_multistep_problem(
            model, x0, step_count=4
        )
        xs_seed, us_seed, _ = module.build_continuation_seed(
            problem=problem,
            state=state,
            target_steps=4,
            source_path=module.PERIODIC_TRAJECTORY_PATH,
        )
        solver = SimpleNamespace(iter=1, stop=1.0, cost=10.0)
        diagnostics, com = module.compute_multistep_diagnostics(
            model=model,
            problem=problem,
            state=state,
            xs=xs_seed,
            us=us_seed,
            metadata=metadata,
            solver=solver,
        )

        self.assertEqual(com.shape, (401, 3))
        self.assertEqual(diagnostics["landing_count"], 4)
        self.assertEqual(len(diagnostics["landing_translation_errors_m"]), 4)
        self.assertEqual(len(diagnostics["swing_clearances_m"]), 4)
        self.assertLessEqual(diagnostics["shooting_gap_norm"], 1.0e-3)
        self.assertGreater(diagnostics["shooting_gap_norm"], 1.0e-7)
        self.assertAlmostEqual(diagnostics["duration_s"], 4.0)
        self.assertEqual(diagnostics["contact_support_node_count"], 400)
        self.assertEqual(
            {row["cycle_id"] for row in diagnostics["contact_feasibility"]["rows"]},
            {0, 1},
        )

    def test_one_iteration_segment_smoke_returns_failed_gated_result(self):
        module = load_module()
        result = module.solve_segment(
            step_count=4,
            source_path=module.PERIODIC_TRAJECTORY_PATH,
            max_iterations=1,
        )

        self.assertFalse(result["acceptance"]["passed"])
        self.assertFalse(result["exported"])
        self.assertIn("solver_stop", result["diagnostics"])
        self.assertIn("shooting_gap_norm", result["diagnostics"])
        self.assertIn("contact_violation_count", result["diagnostics"])
        self.assertIn("minimum_cop_margin_m", result["diagnostics"])
        self.assertEqual(result["seed_provenance"]["source_step_count"], 2)


    def test_one_iteration_ladder_stops_and_preserves_failure_without_candidate(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result = module.solve_ladder(
                targets=(4, 6, 10),
                max_iterations=1,
                log_dir=root / "logs",
                trajectory_dir=root / "trajectories",
            )

            self.assertFalse(result["passed"])
            self.assertEqual(result["attempted_targets"], [4])
            self.assertEqual(result["accepted_targets"], [])
            self.assertTrue(result["diagnostic_yaml"].is_file())
            self.assertTrue(result["iteration_csv"].is_file())
            self.assertFalse(
                (
                    root
                    / "trajectories"
                    / "fddp_multistep_10step_planning_candidate.npz"
                ).exists()
            )
            self.assertFalse(
                (
                    root
                    / "trajectories"
                    / "fddp_multistep_4step_accepted_seed.npz"
                ).exists()
            )



if __name__ == "__main__":
    unittest.main()
