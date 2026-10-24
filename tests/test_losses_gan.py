"""Hand-checkable tests for the P1 adversarial losses (src/training/losses_gan.py)."""

import math
import unittest

import torch

from src.training import losses_gan as L


class DiscriminatorLossTest(unittest.TestCase):
    def test_equal_real_and_fake_logits_give_ln2(self):
        # Relativistic margin real-fake == 0 regardless of label_smoothing:
        # sigmoid(0) = 0.5, and BCE(0, target) == -log(0.5) == ln2 for *any*
        # target (the target cancels out at x=0). This is the "discriminator
        # can't tell them apart at all" case the real training run hit.
        real = torch.zeros(4, 1, 8, 8)
        fake = torch.zeros(4, 1, 8, 8)
        loss = L.discriminator_loss(real, fake)
        self.assertAlmostEqual(loss.item(), math.log(2.0), places=5)

    def test_separating_real_from_fake_lowers_loss_below_the_tied_case(self):
        real = torch.full((2, 1, 4, 4), 3.0)
        fake = torch.full((2, 1, 4, 4), -3.0)  # margin = 6, near the label-smoothed optimum
        tied = L.discriminator_loss(torch.zeros_like(real), torch.zeros_like(fake)).item()
        self.assertLess(L.discriminator_loss(real, fake).item(), tied)

    def test_label_smoothing_bounds_the_optimal_margin(self):
        # With label_smoothing=0.1 the loss-minimising margin is
        # logit(0.9) = ln(9) ~= 2.197, not +inf -- pushing the margin far
        # past that must make the loss *worse*, which is exactly what
        # label smoothing is for (never ask D for an unreachable certainty).
        near_optimum = torch.tensor(math.log(9.0))
        far_past = torch.tensor(50.0)
        zero = torch.zeros(())
        loss_near = L.discriminator_loss(near_optimum, zero)
        loss_far = L.discriminator_loss(far_past, zero)
        self.assertLess(loss_near.item(), loss_far.item())


class GeneratorAdversarialLossTest(unittest.TestCase):
    def test_equal_real_and_fake_logits_give_ln2(self):
        fake = torch.zeros(3, 1, 5, 5)
        real = torch.zeros(3, 1, 5, 5)
        self.assertAlmostEqual(
            L.generator_adversarial_loss(fake, real).item(), math.log(2.0), places=5
        )

    def test_fooling_discriminator_lowers_loss_below_the_tied_case(self):
        fake = torch.full((3, 1, 5, 5), 3.0)
        real = torch.full((3, 1, 5, 5), -3.0)  # fake looks much more real than real
        tied = L.generator_adversarial_loss(torch.zeros_like(fake), torch.zeros_like(real)).item()
        self.assertLess(L.generator_adversarial_loss(fake, real).item(), tied)


class EnsembleL1LossTest(unittest.TestCase):
    def test_matches_abs_of_mean_minus_target(self):
        # members mean = 2, target = 5  ->  |5 - 2| = 3   (P1 Eq. 3)
        members = torch.stack([
            torch.full((2, 1, 4, 4), 1.0),
            torch.full((2, 1, 4, 4), 2.0),
            torch.full((2, 1, 4, 4), 3.0),
        ])
        target = torch.full((2, 1, 4, 4), 5.0)
        self.assertAlmostEqual(L.ensemble_l1_loss(members, target).item(), 3.0, places=6)

    def test_mask_restricts_the_average(self):
        members = torch.zeros(3, 1, 1, 4)               # (N, B, H, W)
        target = torch.tensor([[[10.0, 10.0, 0.0, 0.0]]])   # (B, H, W)
        mask = torch.tensor([[[True, True, False, False]]])
        # only the two masked cells count -> |10 - 0| = 10
        self.assertAlmostEqual(L.ensemble_l1_loss(members, target, mask).item(), 10.0, places=6)

    def test_requires_at_least_three_members(self):
        members = torch.zeros(2, 1, 4, 4)
        with self.assertRaises(ValueError):
            L.ensemble_l1_loss(members, torch.zeros(1, 4, 4))


class GeneratorLossTest(unittest.TestCase):
    def test_is_sum_of_adversarial_and_ensemble_l1(self):
        fake = torch.zeros(2, 1, 4, 4)
        real = torch.zeros(2, 1, 4, 4)                      # tied -> adversarial = ln 2
        members = torch.stack([torch.full((2, 1, 4, 4), 1.0)] * 3)  # mean 1
        target = torch.full((2, 1, 4, 4), 4.0)             # l1 = 3
        total, parts = L.generator_loss(fake, real, members, target)
        self.assertAlmostEqual(parts["generator_adversarial"].item(), math.log(2.0), places=5)
        self.assertAlmostEqual(parts["generator_ensemble_l1"].item(), 3.0, places=5)
        self.assertAlmostEqual(total.item(), math.log(2.0) + 3.0, places=5)


if __name__ == "__main__":
    unittest.main()
