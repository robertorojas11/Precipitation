"""Tests for Shapley attribution (src/evaluation/shapley_attribution.py),
checked against games with analytically known Shapley values."""

import random
import unittest

from src.evaluation import shapley_attribution as S


class ExactShapleyAdditiveGameTest(unittest.TestCase):
    """For a purely additive game v(S) = sum_{i in S} w_i, the Shapley value
    of player i is exactly w_i, regardless of interactions with other
    players (a standard textbook result) -- this makes the additive case a
    clean closed-form check."""

    def test_matches_individual_weights(self):
        weights = [3.0, -1.5, 2.0, 0.5]

        def value_fn(subset):
            return sum(weights[i] for i in subset)

        phi = S.exact_shapley_values(value_fn, len(weights))
        for got, expected in zip(phi, weights):
            self.assertAlmostEqual(got, expected, places=8)

    def test_values_sum_to_full_minus_empty(self):
        weights = [1.0, 2.0, 3.0]

        def value_fn(subset):
            return sum(weights[i] for i in subset) + 10.0  # nonzero empty value

        phi = S.exact_shapley_values(value_fn, len(weights))
        full = value_fn(frozenset(range(3)))
        empty = value_fn(frozenset())
        self.assertAlmostEqual(sum(phi), full - empty, places=8)


class ExactShapleySymmetryGameTest(unittest.TestCase):
    def test_interchangeable_players_get_equal_credit(self):
        # v(S) = 1 if |S| >= 1 else 0: any single "AND-of-one" voting game
        # where all three players are symmetric -> equal Shapley values.
        def value_fn(subset):
            return 1.0 if len(subset) >= 1 else 0.0

        phi = S.exact_shapley_values(value_fn, 3)
        self.assertAlmostEqual(phi[0], phi[1], places=8)
        self.assertAlmostEqual(phi[1], phi[2], places=8)
        self.assertAlmostEqual(sum(phi), 1.0, places=8)  # full value - empty value = 1


class SamplingShapleyConvergenceTest(unittest.TestCase):
    def test_converges_to_exact_additive_values(self):
        weights = [2.0, -3.0, 1.0, 0.5, 4.0]

        def value_fn(subset):
            return sum(weights[i] for i in subset)

        rng = random.Random(0)
        phi = S.sampling_shapley_values(value_fn, len(weights), n_samples=500, rng=rng)
        for got, expected in zip(phi, weights):
            self.assertAlmostEqual(got, expected, places=6)  # exact for additive games


class DispatcherTest(unittest.TestCase):
    def test_uses_exact_below_threshold(self):
        calls = {"n": 0}

        def value_fn(subset):
            calls["n"] += 1
            return len(subset)

        S.shapley_values(value_fn, 3, max_exact_channels=14)
        self.assertEqual(calls["n"], 2 ** 3)  # every subset visited exactly once (cached)

    def test_uses_sampling_above_threshold(self):
        def value_fn(subset):
            return len(subset)

        phi = S.shapley_values(
            value_fn, 20, max_exact_channels=14, n_samples=50, rng=random.Random(1)
        )
        self.assertEqual(len(phi), 20)


class ChannelAttributionTest(unittest.TestCase):
    def test_matches_hand_computed_additive_attribution(self):
        weights = {"cp": 1.0, "lsp": 2.0, "land_sea_mask": -0.5}
        names = list(weights)

        def predict_fn(channels):
            return sum(w * v for w, v in zip(weights.values(), channels))

        sample = [10.0, 5.0, 1.0]
        baseline = [0.0, 0.0, 0.0]
        result = S.channel_attribution(predict_fn, sample, baseline, names)

        expected = [weights["cp"] * 10.0, weights["lsp"] * 5.0, weights["land_sea_mask"] * 1.0]
        for got, exp in zip(result["shapley_values"], expected):
            self.assertAlmostEqual(got, exp, places=6)
        self.assertAlmostEqual(result["base_value"], 0.0, places=8)
        self.assertAlmostEqual(result["prediction"], sum(expected), places=6)
        self.assertAlmostEqual(result["sum_of_shapley_values"], result["prediction"], places=6)


if __name__ == "__main__":
    unittest.main()
