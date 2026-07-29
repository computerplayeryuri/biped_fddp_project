from __future__ import annotations

import unittest
from pathlib import Path

try:
    from reporting import generate_robustness_ppt_figures as figures
except ImportError:
    figures = None


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "results" / "logs"


class RobustnessEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertIsNotNone(
            figures,
            "reporting.generate_robustness_ppt_figures must be implemented",
        )

    def test_push_summary_matches_recorded_trials(self) -> None:
        trials, selection = figures.load_push_evidence(LOG_DIR)
        summary = figures.summarise_push_trials(trials, pitch_gate_deg=15.0)

        self.assertEqual(len(trials), 60)
        self.assertEqual(int(summary.loc["forward", "passed"]), 30)
        self.assertEqual(int(summary.loc["backward", "passed"]), 30)
        self.assertAlmostEqual(
            float(summary.loc["forward", "mean_peak_deg"]),
            9.024600,
            places=5,
        )
        self.assertAlmostEqual(
            float(summary.loc["backward", "mean_peak_deg"]),
            13.712496,
            places=5,
        )
        self.assertAlmostEqual(
            float(summary.loc["backward", "margin_deg"]),
            15.0 - 13.745362,
            places=5,
        )
        self.assertTrue(selection["accepted"])

    def test_scenario_profile_preserves_all_trials_and_summary_order(self) -> None:
        trials, aggregate, campaign = figures.load_scenario_evidence(LOG_DIR)
        profile = figures.build_scenario_profile(trials, aggregate, campaign)

        self.assertEqual(len(profile), 32)
        self.assertEqual(len(trials), 960)
        self.assertEqual(int(profile["passed"].sum()), 960)
        self.assertTrue((profile["success_rate"] == 1.0).all())
        self.assertAlmostEqual(
            float(profile.iloc[0]["wilson_low"]),
            0.886486606826,
            places=10,
        )
        self.assertEqual(campaign["trial_count"], 960)
        self.assertEqual(profile.iloc[0]["scenario"], "nominal")
        self.assertEqual(profile.iloc[-1]["scenario"], "combined_C")

    def test_loader_rejects_missing_required_columns(self) -> None:
        import pandas as pd

        with self.assertRaisesRegex(ValueError, "missing required columns"):
            figures.summarise_push_trials(
                pd.DataFrame({"direction": ["forward"]}),
                pitch_gate_deg=15.0,
            )


if __name__ == "__main__":
    unittest.main()
