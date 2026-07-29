import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "model" / "02_audit_urdf_mjcf_consistency.py"


def load_module():
    spec = importlib.util.spec_from_file_location("urdf_mjcf_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class URDFMJCFConsistencyAuditTests(unittest.TestCase):
    def test_body_mapping_folds_both_contact_lumps_without_dropping_links(self):
        module = load_module()
        mapping = module.build_body_mapping()
        sources = [source for item in mapping.values() for source in item]
        self.assertEqual(len(mapping), 14)
        self.assertEqual(len(sources), 16)
        self.assertEqual(len(set(sources)), 16)
        self.assertEqual(
            mapping["left_foot_link"],
            ["left_foot_link", "left_foot_contact_lump"],
        )
        self.assertEqual(
            mapping["right_foot_link"],
            ["right_foot_link", "right_foot_contact_lump"],
        )

    def test_folded_left_foot_inertia_matches_mjcf_rounding(self):
        module = load_module()
        urdf_root, mjcf_root = module.load_roots()
        folded = module.combine_urdf_inertials(
            urdf_root,
            ["left_foot_link", "left_foot_contact_lump"],
            "left_foot_link",
        )
        mjcf = module.mjcf_body_inertial(mjcf_root, "left_foot_link")
        self.assertAlmostEqual(folded["mass"], mjcf["mass"], places=9)
        np.testing.assert_allclose(folded["com"], mjcf["com"], atol=1e-12)
        np.testing.assert_allclose(
            np.diag(folded["inertia"]),
            np.diag(mjcf["inertia"]),
            atol=5e-9,
        )

    def test_audit_classifies_differences_without_claiming_full_equivalence(self):
        module = load_module()
        report = module.audit_models()
        self.assertFalse(report["summary"]["full_equivalence"])
        self.assertEqual(report["summary"]["unexpected_mismatch_count"], 0)
        self.assertGreater(report["summary"]["matched_count"], 40)
        categories = {row["category"] for row in report["rows"]}
        self.assertIn("actuator_effort", categories)
        self.assertIn("collision_enablement", categories)
        self.assertIn("contact_parameters", categories)

    def test_zero_pose_base_height_places_foot_bottom_on_floor(self):
        module = load_module()
        self.assertAlmostEqual(module.derive_standing_base_height(), 0.288, places=12)


if __name__ == "__main__":
    unittest.main()
