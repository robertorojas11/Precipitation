"""Reference-value tests for src/evaluation/probabilistic_metrics.py."""

import unittest

import numpy as np

from src.evaluation import probabilistic_metrics as PM
from src.evaluation import deterministic_metrics as DM
from src.evaluation import baselines


class CrpsTest(unittest.TestCase):
    def test_single_member_crps_equals_mae_exactly(self):
        # Both papers: for a deterministic forecast CRPS reduces to MAE
        # (P1 M181, P2 M247). This must be exact, not approximate.
        rng = np.random.default_rng(1)
        obs = rng.gamma(1.2, 2.0, size=(1, 24, 24))
        pred = obs + rng.normal(0.0, 1.5, obs.shape)
        crps = PM.crps_ensemble(pred[None], obs)          # ensemble of 1
        mae = DM.mae(pred, obs)
        self.assertAlmostEqual(crps, mae, places=12)

    def test_two_member_closed_form(self):
        # ensemble {0, 2}, obs 1:
        #   skill  = mean(|0-1|, |2-1|) = 1
        #   spread = (|0-0|+|0-2|+|2-0|+|2-2|) / 4 = 1
        #   CRPS   = 1 - 0.5 * 1 = 0.5
        ensemble = np.array([[[0.0]], [[2.0]]])           # (2, 1, 1)
        obs = np.array([[1.0]])                            # (1, 1)
        self.assertAlmostEqual(PM.crps_ensemble(ensemble, obs), 0.5, places=12)

    def test_perfect_ensemble_has_zero_crps(self):
        obs = np.array([[3.0, 1.0], [0.0, 7.0]])
        ensemble = np.stack([obs, obs, obs])
        self.assertAlmostEqual(PM.crps_ensemble(ensemble, obs), 0.0, places=12)


class FssTest(unittest.TestCase):
    def test_identical_fields_give_one(self):
        field = np.array([[0.0, 5.0], [5.0, 0.0]])
        self.assertEqual(PM.fractions_skill_score(field, field, 1.0, 1), 1.0)

    def test_complete_miss_gives_zero(self):
        obs = np.full((8, 8), 10.0)
        pred = np.zeros((8, 8))
        # scale 0: f_pred = 0, f_obs = 1 -> num = 1, den = 1 -> FSS = 0
        self.assertAlmostEqual(PM.fractions_skill_score(pred, obs, 1.0, 0), 0.0, places=12)

    def test_ensemble_fraction_exceeding(self):
        ens = np.array([[[0.0, 2.0]], [[2.0, 2.0]], [[0.0, 0.0]]])
        got = PM.ensemble_fraction_exceeding(ens, 1.0)
        np.testing.assert_allclose(got, [[1 / 3, 2 / 3]])


class PercentileMaeTest(unittest.TestCase):
    def test_constant_shift_shows_in_every_percentile(self):
        rng = np.random.default_rng(2)
        obs = rng.gamma(1.0, 1.0, size=(40, 40))
        pred = obs + 2.0
        out = PM.percentile_mae(pred, obs)
        self.assertAlmostEqual(out[95.0], 2.0, places=6)
        self.assertAlmostEqual(out[99.0], 2.0, places=6)


class RankHistogramTest(unittest.TestCase):
    def test_counts_sum_to_valid_pixels_and_length_is_m_plus_one(self):
        rng = np.random.default_rng(3)
        ensemble = rng.normal(size=(10, 6, 6))
        obs = rng.normal(size=(6, 6))
        hist = PM.rank_histogram(ensemble, obs)
        self.assertEqual(len(hist), 11)
        self.assertEqual(hist.sum(), 36)

    def test_observation_above_all_members_lands_in_last_bin(self):
        ensemble = np.zeros((5, 3, 3))
        obs = np.full((3, 3), 100.0)
        hist = PM.rank_histogram(ensemble, obs)
        self.assertEqual(hist[-1], 9)


class RapsdTest(unittest.TestCase):
    def test_returns_one_dimensional_spectrum(self):
        rng = np.random.default_rng(4)
        spectrum = PM.rapsd(rng.random((32, 32)))
        self.assertEqual(spectrum.ndim, 1)
        self.assertGreater(spectrum.size, 3)

    def test_normalized_spectrum_sums_to_one(self):
        rng = np.random.default_rng(5)
        spectrum = PM.rapsd(rng.random((32, 32)), normalize=True)
        self.assertAlmostEqual(spectrum.sum(), 1.0, places=6)


class EccentricityTest(unittest.TestCase):
    def test_horizontal_stripes_are_anisotropic(self):
        # A field constant along x, varying along y -> autocorrelation elongated
        # along x -> non-zero eccentricity, orientation near horizontal.
        y = np.linspace(0, 6 * np.pi, 64)
        field = np.tile(np.sin(y)[:, None], (1, 64))
        result = PM.linear_eccentricity(field, max_lag_px=20)
        # EXPERIMENTAL metric: assert the anisotropy is detected, not a value.
        self.assertGreater(result.eccentricity, 0.25)
        self.assertGreater(result.major_axis, result.minor_axis)
        self.assertGreater(result.linear_eccentricity, 0.0)


class BiasAndBaselineTest(unittest.TestCase):
    def test_bias_sign_convention_p1_eq13(self):
        # obs mean 4, pred mean 3 -> (4 - 3) / 4 = 0.25
        obs = np.full((4, 4), 4.0)
        pred = np.full((4, 4), 3.0)
        self.assertAlmostEqual(DM.bias(pred, obs), 0.25, places=12)

    def test_bilinear_baseline_upsamples_to_target_shape(self):
        lr = np.arange(16, dtype=float).reshape(4, 4)
        up = baselines.bilinear_interpolation(lr, (16, 16))
        self.assertEqual(up.shape, (16, 16))

    def test_rainfarm_is_not_silently_skipped(self):
        with self.assertRaises(NotImplementedError):
            baselines.rainfarm()


if __name__ == "__main__":
    unittest.main()
