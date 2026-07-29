import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pinocchio as pin


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "fddp" / "04b_solve_fddp_single_step_right_swing.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fddp_single_step_right", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FDDPSingleStepRightGaitTests(unittest.TestCase):
    def test_phase_schedule_is_ds_left_stance_ss_ds_with_120_nodes(self):
        module = load_module()

        phases = module.build_phase_schedule()

        self.assertEqual(
            [phase.name for phase in phases],
            [
                "double_support",
                "left_stance_right_swing",
                "double_support_landing",
            ],
        )
        self.assertEqual([phase.node_count for phase in phases], [20, 70, 30])
        self.assertEqual(sum(phase.node_count for phase in phases), 120)
        self.assertTrue(phases[0].left_contact_active)
        self.assertTrue(phases[0].right_contact_active)
        self.assertTrue(phases[1].left_contact_active)
        self.assertFalse(phases[1].right_contact_active)
        self.assertTrue(phases[2].left_contact_active)
        self.assertTrue(phases[2].right_contact_active)

    def test_swing_pose_starts_lifts_and_lands_at_declared_foothold(self):
        module = load_module()

        start = pin.SE3.Identity()
        poses = module.swing_reference_poses(start)

        self.assertEqual(len(poses), module.HORIZON_NODES + 1)
        self.assertAlmostEqual(poses[0].translation[1], 0.0)
        self.assertAlmostEqual(poses[-1].translation[1], module.STEP_FORWARD_Y_M)
        self.assertGreater(max(pose.translation[2] for pose in poses), 0.010)
        self.assertAlmostEqual(poses[-1].translation[2], 0.0)

    def test_invalid_clearance_is_rejected(self):
        module = load_module()

        with self.assertRaisesRegex(ValueError, "SWING_CLEARANCE_Z_M"):
            module.validate_configuration(clearance_m=0.0)

    def test_problem_applies_left_only_contact_during_right_swing(self):
        module = load_module()
        model = pin.buildModelFromUrdf(
            str(module.URDF_PATH), pin.JointModelFreeFlyer()
        )
        x0 = module.load_x0(module.X0_PATH, model.nq + model.nv)

        problem, _, _ = module.build_single_step_problem(model, x0)

        def contact_names(action_model):
            return set(action_model.differential.contacts.contacts.todict().keys())

        self.assertEqual(
            contact_names(problem.runningModels[0]),
            {module.LEFT_CONTACT_NAME, module.RIGHT_CONTACT_NAME},
        )
        self.assertEqual(
            contact_names(problem.runningModels[20]),
            {module.LEFT_CONTACT_NAME},
        )
        self.assertEqual(
            contact_names(problem.runningModels[90]),
            {module.LEFT_CONTACT_NAME, module.RIGHT_CONTACT_NAME},
        )
        self.assertEqual(
            contact_names(problem.terminalModel),
            {module.LEFT_CONTACT_NAME, module.RIGHT_CONTACT_NAME},
        )

    def test_problem_tracks_active_supports_and_swing_foot_in_se3(self):
        module = load_module()
        model = pin.buildModelFromUrdf(
            str(module.URDF_PATH), pin.JointModelFreeFlyer()
        )
        x0 = module.load_x0(module.X0_PATH, model.nq + model.nv)
        problem, _, _ = module.build_single_step_problem(model, x0)

        def cost_names(action_model):
            return set(action_model.differential.costs.costs.todict().keys())

        self.assertTrue(
            {module.LEFT_SUPPORT_COST_NAME, module.RIGHT_SUPPORT_COST_NAME}
            <= cost_names(problem.runningModels[0])
        )
        self.assertTrue(
            {module.LEFT_SUPPORT_COST_NAME, module.SWING_COST_NAME}
            <= cost_names(problem.runningModels[20])
        )
        self.assertTrue(
            {module.LEFT_SUPPORT_COST_NAME, module.RIGHT_SUPPORT_COST_NAME}
            <= cost_names(problem.runningModels[90])
        )

    def test_acceptance_gate_requires_every_single_step_threshold(self):
        module = load_module()
        accepted = module.evaluate_acceptance(
            {
                "max_shooting_gap_norm": 1.0e-7,
                "max_active_support_translation_error_m": 5.0e-4,
                "max_active_support_rotation_error_rad": np.deg2rad(0.5),
                "landing_translation_error_m": 1.0e-3,
                "landing_rotation_error_rad": np.deg2rad(1.0),
                "min_swing_clearance_m": 0.010,
                "min_joint_limit_margin_rad": 1.0e-2,
                "achieved_forward_com_displacement_m": 0.002,
            }
        )
        self.assertTrue(accepted["passed"])
        self.assertEqual(accepted["failures"], [])

        rejected = module.evaluate_acceptance(
            {
                **accepted["diagnostics"],
                "min_swing_clearance_m": 0.009,
            }
        )
        self.assertFalse(rejected["passed"])
        self.assertIn("swing clearance below threshold", rejected["failures"])

    def test_static_seed_uses_x0_without_gait_warm_start(self):
        module = load_module()
        x0 = np.arange(25, dtype=float)
        xs_seed, us_seed = module.build_static_seed(x0, control_dim=6)

        self.assertEqual(xs_seed.shape, (module.HORIZON_NODES + 1, x0.size))
        self.assertEqual(us_seed.shape, (module.HORIZON_NODES, 6))
        self.assertTrue(np.all(xs_seed == x0[None, :]))
        self.assertTrue(np.all(us_seed == 0.0))

    def test_yaml_safe_value_converts_numpy_scalars_recursively(self):
        module = load_module()
        converted = module.yaml_safe_value(
            {
                "rotation": np.float64(np.deg2rad(0.5)),
                "counts": [np.int64(3)],
                "flag": np.bool_(True),
            }
        )

        self.assertIsInstance(converted["rotation"], float)
        self.assertEqual(converted["counts"], [3])
        self.assertEqual(converted["flag"], True)

    def test_problem_applies_joint_safety_barrier_at_every_node(self):
        module = load_module()
        model = pin.buildModelFromUrdf(
            str(module.URDF_PATH), pin.JointModelFreeFlyer()
        )
        x0 = module.load_x0(module.X0_PATH, model.nq + model.nv)
        problem, _, _ = module.build_single_step_problem(model, x0)

        def cost_names(action_model):
            return set(action_model.differential.costs.costs.todict().keys())

        self.assertIn(
            module.JOINT_SAFETY_COST_NAME, cost_names(problem.runningModels[0])
        )
        self.assertIn(
            module.JOINT_SAFETY_COST_NAME, cost_names(problem.runningModels[20])
        )
        self.assertIn(
            module.JOINT_SAFETY_COST_NAME, cost_names(problem.terminalModel)
        )

    def test_joint_barrier_uses_buffer_above_export_safety_threshold(self):
        module = load_module()
        self.assertGreater(
            module.JOINT_BARRIER_MARGIN_RAD,
            module.MIN_JOINT_LIMIT_MARGIN_RAD,
        )


if __name__ == "__main__":
    unittest.main()
