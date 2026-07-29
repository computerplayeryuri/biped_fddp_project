import importlib.util
import sys
import unittest
from pathlib import Path

import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "fddp" / "04c_solve_fddp_two_step_alternating.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fddp_two_step", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FDDPTwoStepAlternatingTests(unittest.TestCase):
    def test_fixed_schedule_is_ds_left_ss_ds_right_ss_ds(self):
        module = load_module()
        phases = module.build_phase_schedule()
        self.assertEqual(
            [phase.name for phase in phases],
            [
                "double_support_start",
                "right_stance_left_swing",
                "double_support_mid",
                "left_stance_right_swing",
                "double_support_landing",
            ],
        )
        self.assertEqual([phase.node_count for phase in phases], [20, 70, 20, 70, 20])
        self.assertEqual(sum(phase.node_count for phase in phases), module.HORIZON_NODES)
        self.assertTrue(phases[0].right_contact_active)
        self.assertTrue(phases[0].left_contact_active)
        self.assertTrue(phases[1].right_contact_active)
        self.assertFalse(phases[1].left_contact_active)
        self.assertTrue(phases[2].right_contact_active)
        self.assertTrue(phases[2].left_contact_active)
        self.assertFalse(phases[3].right_contact_active)
        self.assertTrue(phases[3].left_contact_active)
        self.assertTrue(phases[4].right_contact_active)
        self.assertTrue(phases[4].left_contact_active)

    def test_left_then_right_swing_references_lift_and_land_forward(self):
        module = load_module()
        left_start = pin.SE3.Identity()
        right_start = pin.SE3.Identity()
        left_poses, right_poses = module.swing_reference_poses(left_start, right_start)
        self.assertEqual(len(left_poses), module.HORIZON_NODES + 1)
        self.assertEqual(len(right_poses), module.HORIZON_NODES + 1)
        self.assertAlmostEqual(left_poses[0].translation[1], 0.0)
        self.assertAlmostEqual(left_poses[90].translation[1], module.STEP_FORWARD_Y_M)
        self.assertGreater(max(p.translation[2] for p in left_poses[20:91]), 0.010)
        self.assertAlmostEqual(left_poses[90].translation[2], 0.0)
        self.assertAlmostEqual(right_poses[109].translation[1], 0.0)
        self.assertAlmostEqual(right_poses[-1].translation[1], module.STEP_FORWARD_Y_M)
        self.assertGreater(max(p.translation[2] for p in right_poses[110:181]), 0.010)
        self.assertAlmostEqual(right_poses[-1].translation[2], 0.0)

    def test_swing_reference_has_early_takeoff_clearance_for_80ms_event_window(self):
        module = load_module()
        left_poses, right_poses = module.swing_reference_poses(
            pin.SE3.Identity(), pin.SE3.Identity()
        )

        self.assertGreaterEqual(left_poses[28].translation[2], 0.002)
        self.assertGreaterEqual(right_poses[118].translation[2], 0.002)

    def test_invalid_clearance_is_rejected(self):
        module = load_module()
        with self.assertRaisesRegex(ValueError, "SWING_CLEARANCE_Z_M"):
            module.validate_configuration(clearance_m=0.0)


if __name__ == "__main__":
    unittest.main()


class FDDPTwoStepProblemTests(unittest.TestCase):
    def test_problem_switches_support_and_tracks_each_swing_foot(self):
        module = load_module()
        model = pin.buildModelFromUrdf(
            str(module.URDF_PATH), pin.JointModelFreeFlyer()
        )
        x0 = module.load_x0(module.X0_PATH, model.nq + model.nv)
        problem, _, _ = module.build_two_step_problem(model, x0)

        def contact_names(action_model):
            return set(action_model.differential.contacts.contacts.todict().keys())

        def cost_names(action_model):
            return set(action_model.differential.costs.costs.todict().keys())

        self.assertEqual(
            contact_names(problem.runningModels[20]),
            {module.RIGHT_CONTACT_NAME},
        )
        self.assertTrue(
            {module.RIGHT_SUPPORT_COST_NAME, module.LEFT_SWING_COST_NAME}
            <= cost_names(problem.runningModels[20])
        )
        self.assertEqual(
            contact_names(problem.runningModels[110]),
            {module.LEFT_CONTACT_NAME},
        )
        self.assertTrue(
            {module.LEFT_SUPPORT_COST_NAME, module.RIGHT_SWING_COST_NAME}
            <= cost_names(problem.runningModels[110])
        )


class FDDPTwoStepAcceptanceTests(unittest.TestCase):
    def test_acceptance_requires_both_landings_and_both_swing_clearances(self):
        module = load_module()
        accepted = module.evaluate_acceptance(
            {
                "max_shooting_gap_norm": 1.0e-7,
                "max_active_support_translation_error_m": 5.0e-4,
                "max_active_support_rotation_error_rad": 0.5 * 3.141592653589793 / 180.0,
                "first_landing_translation_error_m": 1.0e-3,
                "first_landing_rotation_error_rad": 1.0 * 3.141592653589793 / 180.0,
                "second_landing_translation_error_m": 1.0e-3,
                "second_landing_rotation_error_rad": 1.0 * 3.141592653589793 / 180.0,
                "min_left_swing_clearance_m": 0.010,
                "min_right_swing_clearance_m": 0.010,
                "min_joint_limit_margin_rad": 1.0e-2,
                "achieved_forward_com_displacement_m": 0.002,
            }
        )
        self.assertTrue(accepted["passed"])
        rejected = module.evaluate_acceptance(
            {**accepted["diagnostics"], "min_right_swing_clearance_m": 0.009}
        )
        self.assertFalse(rejected["passed"])
        self.assertIn("right swing clearance below threshold", rejected["failures"])

    def test_static_seed_uses_x0_without_scripted_warm_start(self):
        module = load_module()
        import numpy as np
        x0 = np.arange(25, dtype=float)
        xs_seed, us_seed = module.build_static_seed(x0, control_dim=6)
        self.assertEqual(xs_seed.shape, (module.HORIZON_NODES + 1, x0.size))
        self.assertEqual(us_seed.shape, (module.HORIZON_NODES, 6))
        self.assertTrue(np.all(xs_seed == x0[None, :]))
