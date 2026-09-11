"""Hand-checkable tests for the P1 adversarial losses (src/training/losses_gan.py)."""

import math
import unittest

import torch

from src.training import losses_gan as L


class DiscriminatorLossTest(unittest.TestCase):
    def test_zero_logits_give_two_ln2(self):
        # sigmoid(0) = 0.5; BCE(0.5, 1) = BCE(0.5, 0) = -ln(0.5) = ln 2.
        real = torch.zeros(4, 1, 8, 8)
        fake = torch.zeros(4, 1, 8, 8)
        loss = L.discriminator_loss(real, fake)
        self.assertAlmostEqual(loss.item(), 2.0 * math.log(2.0), places=5)

    def test_confident_correct_discriminator_has_low_loss(self):
        real = torch.full((2, 1, 4, 4), 10.0)   # D(real) -> ~1
        fake = torch.full((2, 1, 4, 4), -10.0)  # D(fake) -> ~0
        self.assertLess(L.discriminator_loss(real, fake).item(), 1e-3)


class GeneratorAdversarialLossTest(unittest.TestCase):
    def test_zero_logits_give_ln2(self):
        fake = torch.zeros(3, 1, 5, 5)
        self.assertAlmostEqual(
            L.generator_adversarial_loss(fake).item(), math.log(2.0), places=5
        )

    def test_fooling_discriminator_drives_loss_to_zero(self):
        fake = torch.full((3, 1, 5, 5), 12.0)
        self.assertLess(L.generator_adversarial_loss(fake).item(), 1e-4)


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
        fake = torch.zeros(2, 1, 4, 4)                      # adversarial = ln 2
        members = torch.stack([torch.full((2, 1, 4, 4), 1.0)] * 3)  # mean 1
        target = torch.full((2, 1, 4, 4), 4.0)             # l1 = 3
        total, parts = L.generator_loss(fake, members, target)
        self.assertAlmostEqual(parts["generator_adversarial"].item(), math.log(2.0), places=5)
        self.assertAlmostEqual(parts["generator_ensemble_l1"].item(), 3.0, places=5)
        self.assertAlmostEqual(total.item(), math.log(2.0) + 3.0, places=5)


if __name__ == "__main__":
    unittest.main()
