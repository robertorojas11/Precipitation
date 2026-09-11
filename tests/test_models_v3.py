"""Shape / behaviour contract for the Candidate A + baseline models."""

import unittest

import torch

from src.models.generator import ConditionalGenerator, GeneratorConfig
from src.models.discriminator import ConditionalDiscriminator, DiscriminatorConfig
from src.models.deterministic_resnet import DeterministicResNet, ResNetConfig


def _tiny_gen():
    return ConditionalGenerator(GeneratorConfig(base_width=8, depth=4, noise_channels=2))


class GeneratorTest(unittest.TestCase):
    def test_output_is_single_channel_non_negative_same_grid(self):
        g = _tiny_gen()
        lr = torch.rand(2, 2, 20, 24)
        inv = torch.rand(2, 2, 20, 24)
        out = g(lr, inv)
        self.assertEqual(out.shape, (2, 1, 20, 24))
        self.assertGreaterEqual(out.min().item(), 0.0)  # Softplus tail (P1 M135)

    def test_seeded_ensemble_is_reproducible_and_members_differ(self):
        torch.manual_seed(0)
        g = _tiny_gen()
        lr, inv = torch.rand(1, 2, 16, 16), torch.rand(1, 2, 16, 16)
        a = g.sample(lr, inv, n_members=3, seed=11)
        b = g.sample(lr, inv, n_members=3, seed=11)
        self.assertEqual(a.shape, (3, 1, 1, 16, 16))
        self.assertTrue(torch.allclose(a, b))
        self.assertFalse(torch.allclose(a[0], a[1]))

    def test_sample_does_not_disturb_caller_rng(self):
        g = _tiny_gen()
        lr, inv = torch.rand(1, 2, 8, 8), torch.rand(1, 2, 8, 8)
        torch.manual_seed(99)
        expected = torch.randn(4)
        torch.manual_seed(99)
        g.sample(lr, inv, n_members=3, seed=5)
        self.assertTrue(torch.allclose(expected, torch.randn(4)))


class DiscriminatorTest(unittest.TestCase):
    def test_returns_logit_map_and_backprops_into_generator(self):
        g = _tiny_gen()
        d = ConditionalDiscriminator(DiscriminatorConfig(hr_widths=(8, 8), lr_widths=(8, 4)))
        lr, inv = torch.rand(2, 2, 24, 24), torch.rand(2, 2, 24, 24)
        fake = g(lr, inv)
        logit = d(fake, lr, inv)
        self.assertEqual(logit.shape[0], 2)
        logit.mean().backward()
        grads = [p.grad is not None for p in g.parameters()]
        self.assertTrue(any(grads))

    def test_input_noise_only_applied_in_training_mode(self):
        d = ConditionalDiscriminator(DiscriminatorConfig(hr_widths=(8,), lr_widths=(8,)))
        d.eval()
        hr, lr, inv = torch.rand(1, 1, 16, 16), torch.rand(1, 2, 16, 16), torch.rand(1, 2, 16, 16)
        with torch.no_grad():
            self.assertTrue(torch.allclose(d(hr, lr, inv), d(hr, lr, inv)))


class DeterministicResNetTest(unittest.TestCase):
    def test_deterministic_and_non_negative(self):
        r = DeterministicResNet(ResNetConfig(base_width=8, depth=4))
        r.eval()
        lr, inv = torch.rand(2, 2, 20, 20), torch.rand(2, 2, 20, 20)
        tp = torch.rand(2, 1, 20, 20)
        with torch.no_grad():
            a = r(lr, inv, tp)
            b = r(lr, inv, tp)
        self.assertTrue(torch.allclose(a, b))
        self.assertGreaterEqual(a.min().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
