from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from reporting import generate_robustness_ppt_figures as evidence

try:
    from reporting import render_robustness_ppt_figures as plotting
except ImportError:
    plotting = None


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "results" / "logs"


class RobustnessPushFigureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertIsNotNone(
            plotting,
            "reporting.render_robustness_ppt_figures must be implemented",
        )

    def test_push_figure_exports_slide_png_and_vector_pdf(self) -> None:
        trials, selection = evidence.load_push_evidence(LOG_DIR)
        with TemporaryDirectory() as temporary_directory:
            outputs, metadata = plotting.plot_bidirectional_push_profile(
                trials,
                selection,
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

        self.assertEqual(metadata["forward_passed"], "30/30")
        self.assertEqual(metadata["backward_passed"], "30/30")
        self.assertEqual(metadata["forward_mean_peak_deg"], "9.02")
        self.assertEqual(metadata["backward_mean_peak_deg"], "13.71")


if __name__ == "__main__":
    unittest.main()
