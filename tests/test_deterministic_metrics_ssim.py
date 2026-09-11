"""Tests for the full-field RMSE/SSIM diagnostic and eta-learning cost report
(src/evaluation/deterministic_metrics.py), PASO 8."""

import unittest

import numpy as np

from src.evaluation import deterministic_metrics as DM


class StructuralSimilarityTest(unittest.TestCase):
    def test_identical_fields_give_ssim_one(self):
        rng = np.random.default_rng(0)
        field = rng.gamma(1.0, 2.0, size=(32, 32))
        self.assertAlmostEqual(DM.structural_similarity(field, field), 1.0, places=6)

    def test_very_different_fields_give_low_ssim(self):
        rng = np.random.default_rng(1)
        a = rng.gamma(1.0, 2.0, size=(32, 32))
        b = rng.gamma(1.0, 2.0, size=(32, 32)) + 50.0  # shifted mean, uncorrelated
        self.assertLess(DM.structural_similarity(a, b), 0.5)

    def test_ignores_nan_pixels(self):
        a = np.array([[1.0, 2.0, np.nan], [3.0, 4.0, 5.0]])
        b = np.array([[1.0, 2.0, 100.0], [3.0, 4.0, np.nan]])
        value = DM.structural_similarity(a, b)
        self.assertTrue(np.isfinite(value))


class FullFieldRmseSsimTest(unittest.TestCase):
    def test_returns_both_metrics(self):
        rng = np.random.default_rng(2)
        obs = rng.gamma(1.0, 2.0, size=(16, 16))
        pred = obs + rng.normal(0.0, 0.1, obs.shape)
        out = DM.full_field_rmse_ssim(pred, obs)
        self.assertIn("rmse", out)
        self.assertIn("ssim", out)
        self.assertGreater(out["ssim"], 0.5)


class EtaLearningCostReportTest(unittest.TestCase):
    def test_reports_modest_increase_within_expectation(self):
        baseline = {"rmse": 2.878, "ssim": 0.947}
        eta = {"rmse": 3.112, "ssim": 0.930}  # matches paper's own 8.13% example
        report = DM.eta_learning_cost_report(baseline, eta)
        self.assertAlmostEqual(report["rmse_increase_pct"], 8.13, places=1)
        self.assertFalse(report["miscalibrated"])

    def test_flags_a_much_larger_increase_as_miscalibrated(self):
        baseline = {"rmse": 3.0, "ssim": 0.95}
        eta = {"rmse": 6.0, "ssim": 0.80}  # +100%, far above 2x the 8% reference
        report = DM.eta_learning_cost_report(baseline, eta)
        self.assertTrue(report["miscalibrated"])

    def test_ssim_decrease_is_signed_positive_when_ssim_drops(self):
        baseline = {"rmse": 3.0, "ssim": 0.95}
        eta = {"rmse": 3.1, "ssim": 0.90}
        report = DM.eta_learning_cost_report(baseline, eta)
        self.assertAlmostEqual(report["ssim_decrease"], 0.05, places=6)


if __name__ == "__main__":
    unittest.main()
