"""Tests for per-pixel calibrated distributions
(src/evaluation/pixel_distributions.py), using synthetic data with known
ground truth (PASO 9)."""

import unittest

import numpy as np
from scipy import stats

from src.evaluation import pixel_distributions as D


class AndersonDarlingStatisticTest(unittest.TestCase):
    def test_true_distribution_gives_small_statistic(self):
        rng = np.random.default_rng(0)
        data = stats.weibull_min(2.0, scale=5.0).rvs(500, random_state=rng)
        true_dist = stats.weibull_min(2.0, scale=5.0)
        stat = D.anderson_darling_statistic(data, true_dist)
        self.assertTrue(np.isfinite(stat))
        self.assertLess(stat, 5.0)  # well-fit data: small A^2

    def test_badly_mismatched_distribution_gives_large_statistic(self):
        rng = np.random.default_rng(0)
        data = stats.weibull_min(0.5, scale=1.0).rvs(500, random_state=rng)  # heavy tail
        wrong_dist = stats.norm(loc=data.mean(), scale=data.std())  # symmetric, wrong shape entirely
        stat_wrong = D.anderson_darling_statistic(data, wrong_dist)
        true_dist = stats.weibull_min(0.5, scale=1.0)
        stat_true = D.anderson_darling_statistic(data, true_dist)
        self.assertGreater(stat_wrong, stat_true)


class FitFamilyTest(unittest.TestCase):
    def test_recovers_known_weibull_parameters(self):
        rng = np.random.default_rng(1)
        true_shape, true_scale = 2.0, 5.0
        data = stats.weibull_min(true_shape, scale=true_scale).rvs(800, random_state=rng)
        fitted = D.fit_family("weibull_min", data, n_bootstrap=20, random_state=1)
        got_shape, _, got_scale = fitted.params
        self.assertAlmostEqual(got_shape, true_shape, delta=0.3)
        self.assertAlmostEqual(got_scale, true_scale, delta=0.5)

    def test_aic_bic_formulas(self):
        rng = np.random.default_rng(2)
        data = stats.gamma(a=3.0, scale=2.0).rvs(300, random_state=rng)
        fitted = D.fit_family("gamma", data, n_bootstrap=10, random_state=2)
        n = len(data)
        self.assertAlmostEqual(fitted.aic, 2 * fitted.n_params - 2 * fitted.loglik, places=6)
        self.assertAlmostEqual(fitted.bic, fitted.n_params * np.log(n) - 2 * fitted.loglik, places=6)

    def test_bootstrap_pvalue_is_a_valid_probability(self):
        rng = np.random.default_rng(3)
        data = stats.lognorm(s=0.5, scale=3.0).rvs(200, random_state=rng)
        fitted = D.fit_family("lognorm", data, n_bootstrap=25, random_state=3)
        self.assertGreaterEqual(fitted.ad_pvalue, 0.0)
        self.assertLessEqual(fitted.ad_pvalue, 1.0)


class WeibullMixture2Test(unittest.TestCase):
    def test_em_recovers_a_bimodal_scale_mixture_better_than_single_weibull(self):
        rng = np.random.default_rng(4)
        # Two well-separated Weibull components -> a single Weibull should
        # fit poorly (in log-likelihood) relative to the two-component
        # mixture recovering the bimodal structure.
        comp1 = stats.weibull_min(3.0, scale=2.0).rvs(400, random_state=rng)
        comp2 = stats.weibull_min(3.0, scale=30.0).rvs(400, random_state=rng)
        data = np.concatenate([comp1, comp2])

        mixture, mixture_loglik = D.fit_weibull_mixture2(data, max_iter=40, random_state=4)
        single_params = stats.weibull_min.fit(data, floc=0)
        single_loglik = float(np.sum(stats.weibull_min.logpdf(data, *single_params)))

        self.assertGreater(mixture_loglik, single_loglik)
        self.assertEqual(mixture.pdf(np.array([5.0])).shape, (1,))

    def test_cdf_and_ppf_are_approximately_inverse(self):
        rng = np.random.default_rng(5)
        data = np.concatenate([
            stats.weibull_min(2.0, scale=2.0).rvs(200, random_state=rng),
            stats.weibull_min(2.0, scale=20.0).rvs(200, random_state=rng),
        ])
        mixture, _ = D.fit_weibull_mixture2(data, max_iter=30, random_state=5)
        for q in (0.1, 0.5, 0.9):
            x = mixture.ppf(q)
            self.assertAlmostEqual(mixture.cdf(np.array([x]))[0], q, delta=0.02)


class SelectFamilyTest(unittest.TestCase):
    def _fake(self, aic, ad_pvalue):
        return D.FittedFamily(
            name="fake", params=(), n_params=2, loglik=0.0, aic=aic, bic=aic,
            ad_statistic=0.0, ad_pvalue=ad_pvalue, dist=None,
        )

    def test_picks_lowest_aic_among_passing_families(self):
        fitted = {
            "a": self._fake(aic=10.0, ad_pvalue=0.5),
            "b": self._fake(aic=5.0, ad_pvalue=0.2),   # lowest AIC, passes gof
            "c": self._fake(aic=1.0, ad_pvalue=0.01),  # lowest AIC overall, REJECTED
        }
        name, passed = D.select_family(fitted, alpha=0.05)
        self.assertEqual(name, "b")
        self.assertTrue(passed)

    def test_falls_back_to_lowest_aic_when_all_rejected(self):
        fitted = {
            "a": self._fake(aic=10.0, ad_pvalue=0.01),
            "b": self._fake(aic=5.0, ad_pvalue=0.001),
        }
        name, passed = D.select_family(fitted, alpha=0.05)
        self.assertEqual(name, "b")
        self.assertFalse(passed)


class ZeroInflatedFitTest(unittest.TestCase):
    def test_end_to_end_synthetic_zero_inflated_precipitation(self):
        rng = np.random.default_rng(6)
        n_zero, n_rain = 700, 300
        zeros = np.zeros(n_zero)
        rain = stats.weibull_min(1.5, scale=8.0).rvs(n_rain, random_state=rng)
        values = np.concatenate([zeros, rain])
        rng.shuffle(values)

        fit = D.fit_pixel_distribution(
            values, families=("weibull_min", "gamma"), n_bootstrap=15, random_state=6
        )
        self.assertAlmostEqual(fit.p_zero, n_zero / (n_zero + n_rain), delta=0.03)

        # Below p_zero -> ppf is exactly zero (the point mass).
        self.assertEqual(fit.ppf(fit.p_zero / 2.0), 0.0)
        # Above p_zero -> ppf is strictly positive and increasing.
        low = fit.ppf(fit.p_zero + 0.01)
        high = fit.ppf(0.99)
        self.assertGreater(low, 0.0)
        self.assertGreater(high, low)

        # Exceedance probability is monotonically decreasing in threshold.
        p_low = fit.exceedance_probability(1.0)
        p_high = fit.exceedance_probability(20.0)
        self.assertGreaterEqual(p_low, p_high)
        self.assertGreaterEqual(p_low, 0.0)
        self.assertLessEqual(p_high, 1.0)

    def test_quantile_map_helper(self):
        rng = np.random.default_rng(7)
        values = np.concatenate([np.zeros(200), stats.gamma(a=2.0, scale=3.0).rvs(300, random_state=rng)])
        fit = D.fit_pixel_distribution(values, families=("gamma",), n_bootstrap=10, random_state=7)
        result = D.quantile_map(fit, [0.9, 0.99])
        self.assertIn(0.9, result)
        self.assertLess(result[0.9], result[0.99])


if __name__ == "__main__":
    unittest.main()
