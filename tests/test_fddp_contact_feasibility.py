import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "fddp" / "contact_feasibility.py"
FOOT_POLYGON = np.array(
    [
        [-0.0425, -0.0725],
        [0.0425, -0.0725],
        [0.0425, 0.0725],
        [-0.0425, 0.0725],
    ],
    dtype=float,
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "fddp_contact_feasibility", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ContactFeasibilityMathTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_known_wrench_uses_pinocchio_force_order_and_cop_sign(self):
        wrench = np.array([10.0, 0.0, 100.0, 0.5, 1.0, 0.0])
        metrics = self.module.wrench_metrics(
            wrench, mu=0.8, polygon=FOOT_POLYGON
        )
        self.assertAlmostEqual(metrics["friction_utilisation"], 0.125)
        self.assertTrue(np.allclose(metrics["cop_xy_m"], [-0.01, 0.005]))
        self.assertFalse(metrics["unilateral_violation"])
        self.assertGreaterEqual(metrics["cop_margin_m"], 0.002)

    def test_polygon_margin_is_positive_inside_zero_on_edge_negative_outside(self):
        self.assertAlmostEqual(
            self.module.point_polygon_signed_margin([0.0, 0.0], FOOT_POLYGON),
            0.0425,
        )
        self.assertAlmostEqual(
            self.module.point_polygon_signed_margin([0.0425, 0.0], FOOT_POLYGON),
            0.0,
        )
        self.assertAlmostEqual(
            self.module.point_polygon_signed_margin([0.0500, 0.0], FOOT_POLYGON),
            -0.0075,
        )

    def test_zero_and_negative_normal_force_are_not_accepted(self):
        zero = self.module.wrench_metrics(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], mu=0.8, polygon=FOOT_POLYGON
        )
        negative = self.module.wrench_metrics(
            [0.0, 0.0, -1.0, 0.0, 0.0, 0.0], mu=0.8,
            polygon=FOOT_POLYGON,
        )
        self.assertTrue(zero["normal_force_invalid"])
        self.assertTrue(negative["unilateral_violation"])
        self.assertTrue(np.isinf(zero["friction_utilisation"]))
        self.assertTrue(np.isnan(zero["cop_margin_m"]))

    def test_threshold_equalities_are_inclusive(self):
        rows = [
            {
                "node": 0,
                "side": "left",
                "wrench": [72.0, 0.0, 100.0, 0.0, 0.0, 0.0],
                "tangential_speed_m_s": 0.001,
            }
        ]
        config = {
            "friction_coefficient": 0.8,
            "minimum_normal_force_n": -1.0e-5,
            "maximum_friction_utilisation": 0.90,
            "minimum_cop_margin_m": 0.002,
            "maximum_tangential_speed_m_s": 0.001,
        }
        report = self.module.evaluate_contact_feasibility(
            rows, config, polygon=FOOT_POLYGON
        )
        self.assertTrue(report["accepted"])
        self.assertEqual(report["violation_count"], 0)

    def test_ds_load_distribution_sums_to_one_per_node(self):
        rows = [
            {"node": 7, "side": "left", "wrench": [0, 0, 30, 0, 0, 0],
             "tangential_speed_m_s": 0.0},
            {"node": 7, "side": "right", "wrench": [0, 0, 70, 0, 0, 0],
             "tangential_speed_m_s": 0.0},
        ]
        config = {
            "friction_coefficient": 0.8,
            "minimum_normal_force_n": -1.0e-5,
            "maximum_friction_utilisation": 0.90,
            "minimum_cop_margin_m": 0.002,
            "maximum_tangential_speed_m_s": 0.001,
        }
        report = self.module.evaluate_contact_feasibility(
            rows, config, polygon=FOOT_POLYGON
        )
        fractions = [row["normal_load_fraction"] for row in report["rows"]]
        self.assertTrue(np.allclose(fractions, [0.3, 0.7]))
        self.assertAlmostEqual(sum(fractions), 1.0)

    def test_slip_boundary_and_friction_contract(self):
        self.assertFalse(self.module.classify_slip(0.001, maximum=0.001))
        self.assertTrue(self.module.classify_slip(0.0010001, maximum=0.001))
        self.module.assert_friction_contract(0.8, 0.8)
        with self.assertRaisesRegex(ValueError, "friction"):
            self.module.assert_friction_contract(0.8, 0.79)
    def test_lwa_wrench_is_shifted_from_ankle_origin_to_sole_center(self):
        # F=[10, 0, 100], M_sole=[0.5, 1.0, 0].  With the sole 40 mm
        # below the ankle frame, M_ankle=M_sole+r x F=[0.5, 0.6, 0].
        at_sole = self.module.contact_wrench_at_sole(
            [10.0, 0.0, 100.0, 0.5, 0.6, 0.0],
            frame_rotation_world=np.eye(3),
            sole_offset_local=[0.0, 0.0, -0.04],
        )
        self.assertTrue(
            np.allclose(at_sole, [10.0, 0.0, 100.0, 0.5, 1.0, 0.0])
        )

    def test_lwa_velocity_is_shifted_to_sole_before_slip_measurement(self):
        speed = self.module.sole_tangential_speed(
            [0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            frame_rotation_world=np.eye(3),
            sole_offset_local=[0.0, 0.0, -0.04],
        )
        self.assertAlmostEqual(speed, 0.08)
    def test_contact_frame_to_sole_offset_removes_existing_urdf_frame_shift(self):
        offset = self.module.contact_frame_to_sole_offset(
            sole_offset_from_ankle_local=[0.0, 0.0, -0.043],
            contact_frame_offset_from_ankle_local=[0.0, 0.0, -0.040],
        )
        self.assertTrue(np.allclose(offset, [0.0, 0.0, -0.003]))



class PlanarSupportResultantTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    @staticmethod
    def _contact(*, side, y, wrench):
        return {
            "side": side,
            "sole_position_world_m": [0.0, y, 0.0],
            "wrench": wrench,
        }

    def test_single_support_boundaries_are_inclusive(self):
        contact = self._contact(
            side="right",
            y=0.0,
            wrench=[999.0, 72.0, 100.0, 7.05, -500.0, 300.0],
        )
        result = self.module.planar_support_resultant(
            [contact],
            friction_coefficient=0.8,
            maximum_friction_utilisation=0.90,
            foot_half_length_m=0.0725,
            minimum_cop_margin_m=0.002,
        )
        self.assertTrue(result["accepted"])
        np.testing.assert_allclose(
            result["cone_values"],
            [100.0, 0.0, 144.0, 14.1, 0.0],
            atol=2.0e-14,
        )
        self.assertAlmostEqual(result["friction_utilisation"], 0.90)
        self.assertAlmostEqual(result["cop_margin_m"], 0.002)
        self.assertEqual(result["evaluation_mode"], "planar_sagittal_resultant")
        self.assertEqual(result["out_of_plane_policy"], "diagnostic_only")

    def test_ds_internal_wrench_nullspace_does_not_change_gate(self):
        baseline = [
            self._contact(side="left", y=0.0, wrench=[0, 10, 40, 1, 0, 0]),
            self._contact(side="right", y=0.02, wrench=[0, -5, 60, 2, 0, 0]),
        ]
        redistributed = [
            self._contact(
                side="left", y=0.0, wrench=[20, 1010, 1040, 51, 70, 80]
            ),
            self._contact(
                side="right", y=0.02, wrench=[-20, -1005, -940, -28, -70, -80]
            ),
        ]
        kwargs = dict(
            friction_coefficient=0.8,
            maximum_friction_utilisation=0.90,
            foot_half_length_m=0.0725,
            minimum_cop_margin_m=0.002,
        )
        first = self.module.planar_support_resultant(baseline, **kwargs)
        second = self.module.planar_support_resultant(redistributed, **kwargs)
        np.testing.assert_allclose(second["cone_values"], first["cone_values"])
        self.assertAlmostEqual(second["resultant_fy_n"], first["resultant_fy_n"])
        self.assertAlmostEqual(second["resultant_fz_n"], first["resultant_fz_n"])
        self.assertAlmostEqual(
            second["resultant_mx_world_nm"], first["resultant_mx_world_nm"]
        )
        self.assertTrue(second["per_foot_load_split_nonidentifiable"])

    def test_planar_violations_are_classified_independently(self):
        kwargs = dict(
            friction_coefficient=0.8,
            maximum_friction_utilisation=0.90,
            foot_half_length_m=0.0725,
            minimum_cop_margin_m=0.002,
        )
        negative = self.module.planar_support_resultant(
            [self._contact(side="left", y=0.0, wrench=[0, 0, -1, 0, 0, 0])],
            **kwargs,
        )
        friction = self.module.planar_support_resultant(
            [self._contact(side="left", y=0.0, wrench=[0, 73, 100, 0, 0, 0])],
            **kwargs,
        )
        cop = self.module.planar_support_resultant(
            [self._contact(side="left", y=0.0, wrench=[0, 0, 100, 8, 0, 0])],
            **kwargs,
        )
        self.assertTrue(negative["unilateral_violation"])
        self.assertTrue(friction["friction_violation"])
        self.assertTrue(cop["cop_violation"])
        self.assertFalse(negative["accepted"])
        self.assertFalse(friction["accepted"])
        self.assertFalse(cop["accepted"])

    def test_lateral_components_are_logged_but_do_not_change_planar_gate(self):
        nominal = self._contact(
            side="left", y=0.0, wrench=[0, 0, 100, 0, 0, 0]
        )
        lateral = self._contact(
            side="left", y=0.0, wrench=[5000, 0, 100, 0, 9000, -7000]
        )
        kwargs = dict(
            friction_coefficient=0.8,
            maximum_friction_utilisation=0.90,
            foot_half_length_m=0.0725,
            minimum_cop_margin_m=0.002,
        )
        first = self.module.planar_support_resultant([nominal], **kwargs)
        second = self.module.planar_support_resultant([lateral], **kwargs)
        np.testing.assert_allclose(first["cone_values"], second["cone_values"])
        self.assertTrue(second["accepted"])
        self.assertEqual(second["out_of_plane_policy"], "diagnostic_only")
        self.assertAlmostEqual(second["out_of_plane_resultant_fx_n"], 5000.0)
        self.assertAlmostEqual(second["out_of_plane_resultant_ty_nm"], 9000.0)
        self.assertAlmostEqual(second["out_of_plane_resultant_tz_nm"], -7000.0)
class PlanarSupportReportTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.config = {
            "friction_coefficient": 0.8,
            "minimum_normal_force_n": -1.0e-5,
            "maximum_friction_utilisation": 0.90,
            "minimum_cop_margin_m": 0.002,
            "maximum_tangential_speed_m_s": 0.001,
        }

    def test_grouped_node_report_uses_combined_resultant_and_raw_diagnostics(self):
        rows = [
            {
                "node": 0,
                "time_s": 0.0,
                "phase": "double_support_start",
                "side": "left",
                "sole_position_world_m": [0.0, 0.0, 0.0],
                "wrench": [900.0, 10.0, 40.0, 1.0, 600.0, 500.0],
                "tangential_speed_m_s": 0.0005,
            },
            {
                "node": 0,
                "time_s": 0.0,
                "phase": "double_support_start",
                "side": "right",
                "sole_position_world_m": [0.0, 0.02, 0.0],
                "wrench": [-900.0, -5.0, 60.0, 2.0, -600.0, -500.0],
                "tangential_speed_m_s": 0.001,
            },
            {
                "node": 1,
                "time_s": 0.01,
                "phase": "right_single_support",
                "side": "right",
                "sole_position_world_m": [0.0, 0.02, 0.0],
                "wrench": [0.0, 0.0, 100.0, 2.0, 0.0, 0.0],
                "tangential_speed_m_s": 0.0,
            },
        ]
        report = self.module.evaluate_planar_support_feasibility(
            rows, self.config, foot_half_length_m=0.0725
        )
        self.assertTrue(report["accepted"])
        self.assertEqual(report["evaluation_mode"], "planar_sagittal_resultant")
        self.assertEqual(report["support_node_count"], 2)
        self.assertEqual(report["active_contact_row_count"], 3)
        self.assertEqual(report["violation_count"], 0)
        self.assertTrue(report["rows"][0]["per_foot_load_split_nonidentifiable"])
        self.assertEqual(report["rows"][0]["out_of_plane_policy"], "diagnostic_only")
        self.assertAlmostEqual(report["rows"][0]["maximum_tangential_speed_m_s"], 0.001)
        self.assertAlmostEqual(report["rows"][1]["cop_y_m"], 0.04)

class ContactPromotionGateTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def _candidate(self, path):
        np.savez(
            path,
            schema_version=np.array("fddp-multistep-v1"),
            acceptance_state=np.array("planning_accepted_contact_pending"),
            step_count=np.array(10),
            tau_ref_hardware_usage=np.array("analysis_only"),
            q=np.zeros((1001, 6)),
        )
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_contact_failure_withholds_final_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            candidate = directory / "candidate.npz"
            final = directory / "final.npz"
            digest = self._candidate(candidate)
            with self.assertRaisesRegex(RuntimeError, "contact"):
                self.module.promote_candidate_reference(
                    candidate,
                    final,
                    expected_sha256=digest,
                    contact_report={"accepted": False},
                )
            self.assertFalse(final.exists())

    def test_pass_promotes_atomically_with_final_acceptance_state(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            candidate = directory / "candidate.npz"
            final = directory / "final.npz"
            digest = self._candidate(candidate)
            summary = self.module.promote_candidate_reference(
                candidate,
                final,
                expected_sha256=digest,
                contact_report={"accepted": True},
            )
            self.assertTrue(final.exists())
            with np.load(final, allow_pickle=False) as archive:
                self.assertEqual(
                    str(archive["acceptance_state"]),
                    "planning_and_contact_accepted",
                )
                self.assertEqual(str(archive["tau_ref_hardware_usage"]), "analysis_only")
            self.assertEqual(summary["source_sha256"], digest)
            self.assertEqual(
                summary["final_sha256"],
                hashlib.sha256(final.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
