from __future__ import annotations

import unittest
from pathlib import Path

import matplotlib.pyplot as plt

from reporting import generate_robustness_ppt_figures as evidence
from reporting import render_robustness_ppt_figures as plotting


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "results" / "logs"


class RobustnessScenarioAxisLabelTests(unittest.TestCase):
    def test_success_axis_does_not_clear_scenario_labels(self) -> None:
        trials, aggregate, campaign = evidence.load_scenario_evidence(LOG_DIR)
        profile = evidence.build_scenario_profile(trials, aggregate, campaign)
        block = profile.iloc[:16].reset_index(drop=True)
        figure = plt.figure()
        pitch_axis = figure.add_subplot(1, 2, 1)
        success_axis = figure.add_subplot(1, 2, 2, sharey=pitch_axis)

        plotting._plot_scenario_block(
            pitch_axis,
            success_axis,
            block,
            "Scenarios 1–16",
        )
        visible_labels = [label.get_text() for label in pitch_axis.get_yticklabels()]
        plt.close(figure)

        self.assertIn("Nominal", visible_labels)
        self.assertIn("Push -4 N", visible_labels)
        self.assertIn("Friction μ = 0.40", visible_labels)


if __name__ == "__main__":
    unittest.main()
