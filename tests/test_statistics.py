from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from scipy import stats

from pulse_run_analysis.data import derive_outcomes
from pulse_run_analysis.statistics import (
    cliffs_delta,
    games_howell,
    hedges_g,
    holm_adjust,
    welch_anova,
)


class StatisticsTests(unittest.TestCase):
    def test_welch_anova_matches_scipy(self) -> None:
        groups = {
            "A": np.array([1.2, 1.7, 2.1, 2.5, 2.9]),
            "B": np.array([1.0, 1.1, 1.4, 1.8, 2.2, 2.7]),
            "C": np.array([2.2, 2.8, 3.1, 3.7]),
        }
        result = welch_anova(groups)
        expected = stats.f_oneway(*groups.values(), equal_var=False)
        self.assertAlmostEqual(result["statistic"], float(expected.statistic), places=12)
        self.assertAlmostEqual(result["p_raw"], float(expected.pvalue), places=12)

    def test_holm_adjustment(self) -> None:
        adjusted = holm_adjust([0.01, 0.04, 0.03, 0.20])
        np.testing.assert_allclose(adjusted, [0.04, 0.09, 0.09, 0.20])

    def test_effect_size_directions(self) -> None:
        high = np.array([4.0, 5.0, 6.0, 7.0])
        low = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertGreater(hedges_g(high, low), 0)
        self.assertGreater(cliffs_delta(high, low), 0)
        self.assertAlmostEqual(cliffs_delta(low, high), -cliffs_delta(high, low))

    def test_games_howell_contains_all_pairs(self) -> None:
        groups = {
            "A": np.array([1.0, 1.5, 2.0, 2.4]),
            "B": np.array([2.0, 2.1, 2.5, 3.0]),
            "C": np.array([4.0, 4.2, 4.5, 5.0]),
        }
        result = games_howell(groups, 0.95, 200, 17)
        self.assertEqual(len(result), 3)
        self.assertTrue(result["p_pairwise"].between(0, 1).all())
        self.assertTrue(
            (result["difference_ci_low"] <= result["difference_ci_high"]).all()
        )

    def test_derived_outcomes_use_raw_counts_and_active_duration(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "screening_status": "included",
                    "ddaAlgorithm": "FixedTime",
                    "experience": 1,
                    "movementAnalytics.sampledDurationSeconds": 200.0,
                    "movementAnalytics.headHorizontalDistance": 30.0,
                    "actualRunDurationSeconds": 215.0,
                    "finalScore": 300.0,
                    "totalPatternsSpawned": 10,
                    "veryEasyPatternsSpawned": 1,
                    "easyPatternsSpawned": 2,
                    "mediumPatternsSpawned": 3,
                    "hardPatternsSpawned": 3,
                    "expertPatternsSpawned": 1,
                    "targetsHit": 10,
                    "targetsMissed": 10,
                    "totalShotsFired": 40,
                    "correctColorHits": 8,
                    "wallContactTimeSeconds": 10.0,
                    "totalDepletedTimeSeconds": 20.0,
                    "actualMaxScrollSpeedReached": 5.5,
                    "actualMinPatternGapReached": 2.2,
                    "postRunFeedback.difficultyRating": 5,
                    "postRunFeedback.enjoymentRating": 4,
                    "postRunFeedback.frustrationRating": 2,
                    "postRunFeedback.completed": True,
                    "postRunFeedback.motionSickness": False,
                }
            ]
        )
        derived = derive_outcomes(frame).iloc[0]
        self.assertAlmostEqual(derived["resolved_targets"], 20)
        self.assertAlmostEqual(derived["normalized_target_score_pct"], 75.0)
        self.assertAlmostEqual(derived["wall_contact_pct"], 5.0)
        self.assertAlmostEqual(derived["zero_stability_pct"], 10.0)
        self.assertAlmostEqual(derived["hard_expert_pattern_pct"], 40.0)
        self.assertAlmostEqual(derived["difficulty_deviation"], 2.0)
        self.assertTrue(derived["feedback_complete"])


if __name__ == "__main__":
    unittest.main()
