from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from reporting import generate_robustness_ppt_figures as evidence
from reporting import render_robustness_ppt_figures as plotting


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "results" / "logs"


class RobustnessScenarioFigureTests(unittest.TestCase):
    def setUp(self) -> None:
        trials, aggregate, campaign = evidence.load_scenario_evidence(LOG_DIR)
        self.profile = evidence.build_scenario_profile(trials, aggregate, campaign)

    def test_labels_are_human_readable_and_complete(self) -> None:
        labels = [plotting.scenario_display_label(name) for name in self.profile["scenario"]]

        self.assertEqual(len(labels), 32)
        self.assertEqual(len(set(labels)), 32)
        self.assertEqual(labels[0], "Nominal")
        self.assertIn("Combined A", labels)
        self.assertIn("Combined B", labels)
        self.assertEqual(labels[-1], "Combined C")
        self.assertIn("Friction μ = 0.40", labels)
        self.assertIn("Actuator τ = 50 ms", labels)

    def test_scenario_figure_exports_all_evidence_and_uncertainty(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            outputs, metadata = plotting.plot_robustness_scenario_profile(
                self.profile,
                Path(temporary_directory),
            )

            self.assertEqual({path.suffix for path in outputs}, {".png", ".pdf"})
            for path in outputs:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 1_000)
            png_path = next(path for path in outputs if path.suffix == ".png")
            with Image.open(png_path) as image:
                self.assertGreaterEqual(image.width, 3_000)
                self.assertEqual(image.mode, "RGBA")

        self.assertEqual(metadata["scenario_count"], "32")
        self.assertEqual(metadata["trial_count"], "960")
        self.assertEqual(metadata["passed"], "960/960")
        self.assertEqual(metadata["success_rate"], "100.0%")
        self.assertEqual(metadata["per_scenario_wilson_low"], "88.65%")
        self.assertIn(metadata["most_demanding_scenario"], set(self.profile["scenario"]))
        self.assertTrue(bool(self.profile["critical"].astype(bool).any()))
        self.assertTrue(bool((~self.profile["critical"].astype(bool)).any()))


if __name__ == "__main__":
    unittest.main()
