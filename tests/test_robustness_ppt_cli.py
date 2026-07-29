from __future__ import annotations

import unittest
from pathlib import Path

from reporting import render_robustness_ppt_figures as plotting


ROOT = Path(__file__).resolve().parents[1]


class RobustnessFigureCliTests(unittest.TestCase):
    def test_default_project_root_is_repository_root(self) -> None:
        expected = ROOT.resolve()
        actual = Path(plotting.__file__).resolve().parents[2]

        self.assertEqual(actual, expected)
        plotting.main(["--figure", "push"])
        self.assertTrue(
            (expected / "results/figures/presentation/robustness_bidirectional_push_profile.png").exists()
        )


if __name__ == "__main__":
    unittest.main()
