"""Tests for the eta-learning regularizer (src/training/eta_learning.py),
checked against hand-computable Wasserstein-1 values and the paper's
Algorithm 2 tail-index selection."""

import unittest

import torch

from src.training import eta_learning as E


class SpatialMaximumObservableTest(unittest.TestCase):
    def test_reduces_trailing_spatial_dims(self):
        field = torch.zeros(3, 1, 4, 4)
        field[0, 0, 2, 2] = 5.0
        field[1] = -1.0  # every entry negative -> max is -1.0
        field[2, 0, 3, 3] = 9.0
        out = E.spatial_maximum_observable(field)
        self.assertEqual(out.shape, (3, 1))
        torch.testing.assert_close(out.squeeze(-1), torch.tensor([5.0, -1.0, 9.0]))


class TailQuantileLevelsTest(unittest.TestCase):
    def test_levels_are_within_tau_and_one(self):
        levels = E.tail_quantile_levels(500, tau=0.9, generator=torch.Generator().manual_seed(0))
        self.assertEqual(levels.shape, (500,))
        self.assertTrue((levels >= 0.9).all())
        self.assertTrue((levels <= 1.0).all())

    def test_rejects_invalid_tau(self):
        with self.assertRaises(ValueError):
            E.tail_quantile_levels(10, tau=1.0)


class Wasserstein1Test(unittest.TestCase):
    def test_identical_samples_give_zero_distance(self):
        samples = torch.linspace(0.0, 10.0, 100)
        w1 = E.wasserstein1_1d(samples, samples.clone())
        self.assertAlmostEqual(w1.item(), 0.0, places=4)

    def test_shifted_uniforms_match_closed_form(self):
        # Two same-shaped continuous uniforms shifted by a constant c have a
        # closed-form W1 distance of exactly |c| (the quantile functions
        # differ everywhere by exactly c).
        torch.manual_seed(0)
        n = 20000
        a = torch.rand(n) * 10.0            # Uniform(0, 10)
        shift = 3.0
        b = a.clone() + shift               # same distribution, shifted
        # Resample b independently so it's a genuinely different empirical
        # sample of Uniform(shift, 10+shift), not a paired transform.
        b = torch.rand(n) * 10.0 + shift
        w1 = E.wasserstein1_1d(a, b, n_q=2000)
        self.assertAlmostEqual(w1.item(), shift, places=1)

    def test_eta_tail_loss_zero_when_model_matches_reference_in_tail(self):
        torch.manual_seed(1)
        # Same distribution -> tail quantiles should match closely.
        model_obs = torch.randn(20000) * 2.0 + 5.0
        reference_obs = torch.randn(20000) * 2.0 + 5.0
        loss = E.eta_tail_loss(model_obs, reference_obs, tau=0.95, n_q=500,
                                generator=torch.Generator().manual_seed(2))
        self.assertLess(loss.item(), 0.5)

    def test_eta_tail_loss_large_when_reference_tail_is_much_heavier(self):
        torch.manual_seed(1)
        model_obs = torch.randn(20000) * 1.0 + 5.0
        reference_obs = torch.randn(20000) * 1.0 + 50.0  # much heavier / shifted tail
        loss = E.eta_tail_loss(model_obs, reference_obs, tau=0.95, n_q=500,
                                generator=torch.Generator().manual_seed(2))
        self.assertGreater(loss.item(), 10.0)


class BalanceLambdaTest(unittest.TestCase):
    def test_matches_ratio_formula(self):
        self.assertAlmostEqual(E.balance_lambda(4.0, 2.0), 2.0, places=6)

    def test_handles_zero_w1_gradient_without_dividing_by_zero(self):
        value = E.balance_lambda(1.0, 0.0)
        self.assertTrue(value > 0 and value < float("inf"))


class UpdateTailIndexTest(unittest.TestCase):
    def test_nearest_rank_selection_matches_hand_computed_example(self):
        # Values already sorted: index i (0-based) has rank (i+1)/10.
        values = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        # q=0.5 -> ceil(10*0.5)=5 -> rank index 4 (0-based) -> value 50.0
        # q=1.0 -> ceil(10*1.0)=10 -> rank index 9 -> value 100.0
        # q=0.05 -> ceil(0.5)=1 -> rank index 0 -> value 10.0
        q = torch.tensor([0.05, 0.5, 1.0])
        idx = E.update_tail_index(values, q)
        torch.testing.assert_close(values[idx], torch.tensor([10.0, 50.0, 100.0]))

    def test_works_on_unsorted_input(self):
        values = torch.tensor([50.0, 10.0, 100.0, 30.0, 20.0, 90.0, 40.0, 80.0, 60.0, 70.0])
        q = torch.tensor([1.0])
        idx = E.update_tail_index(values, q)
        self.assertEqual(values[idx].item(), 100.0)


class EtaLearningScheduleTest(unittest.TestCase):
    def test_eta_inactive_during_erm_pretraining(self):
        schedule = E.EtaLearningSchedule(erm_pretrain_steps=5, refresh_every=2)
        for _ in range(5):
            self.assertFalse(schedule.is_eta_active())
            self.assertFalse(schedule.should_refresh_tail_index())
            schedule.advance()
        self.assertTrue(schedule.is_eta_active())

    def test_refresh_cadence_after_activation(self):
        schedule = E.EtaLearningSchedule(erm_pretrain_steps=0, refresh_every=3)
        # step 0: should refresh (never refreshed yet)
        self.assertTrue(schedule.should_refresh_tail_index())
        schedule.mark_refreshed()
        schedule.advance()  # step=1
        self.assertFalse(schedule.should_refresh_tail_index())
        schedule.advance()  # step=2
        self.assertFalse(schedule.should_refresh_tail_index())
        schedule.advance()  # step=3
        self.assertTrue(schedule.should_refresh_tail_index())


if __name__ == "__main__":
    unittest.main()
