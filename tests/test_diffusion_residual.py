"""Tests for the EDM preconditioning, loss, schedule, and Heun+churn sampler
(Candidate B stage 2, src/models/diffusion_residual.py), checked against the
closed-form Avila 2026 Eq. 3-7."""

import math
import unittest

import torch

from src.models.diffusion_residual import (
    EDMConfig,
    EDMPrecond,
    HeunSamplerConfig,
    SimpleNoiseConditionedUNet,
    edm_loss,
    heun_sample,
    karras_sigma_schedule,
)
from src.models.regression_stage import RegressionStageConfig


class EDMCoefficientsTest(unittest.TestCase):
    def test_coefficients_at_sigma_equal_sigma_data(self):
        config = EDMConfig(sigma_data=0.5)
        precond = EDMPrecond(raw_network=torch.nn.Identity(), config=config)
        sigma = torch.tensor([0.5])
        c_skip, c_out, c_in, c_noise = precond._coefficients(sigma.reshape(-1, 1, 1, 1))
        self.assertAlmostEqual(c_skip.item(), 0.5, places=6)
        self.assertAlmostEqual(c_out.item(), 0.5 / math.sqrt(2.0), places=6)
        self.assertAlmostEqual(c_in.item(), 1.0 / (0.5 * math.sqrt(2.0)), places=6)
        self.assertAlmostEqual(c_noise.item(), 0.25 * math.log(0.5), places=6)

    def test_forward_shape_matches_residual(self):
        stage_config = RegressionStageConfig(
            in_channels=5, out_channels=1, base_width=8, encoder_depths=(1, 1)
        )
        raw = SimpleNoiseConditionedUNet(stage_config)
        precond = EDMPrecond(raw)
        residual = torch.randn(2, 1, 16, 16)
        conditioning = torch.randn(2, 4, 16, 16)
        sigma = torch.tensor([0.1, 5.0])
        out = precond(residual, sigma, conditioning)
        self.assertEqual(out.shape, residual.shape)


class EDMLossTest(unittest.TestCase):
    def test_loss_is_finite_and_scalar(self):
        stage_config = RegressionStageConfig(
            in_channels=5, out_channels=1, base_width=8, encoder_depths=(1,)
        )
        raw = SimpleNoiseConditionedUNet(stage_config)
        precond = EDMPrecond(raw)
        clean_residual = torch.randn(3, 1, 12, 12)
        conditioning = torch.randn(3, 4, 12, 12)
        loss = edm_loss(precond, clean_residual, conditioning, generator=torch.Generator().manual_seed(0))
        self.assertEqual(loss.shape, ())
        self.assertTrue(torch.isfinite(loss))

    def test_loss_backpropagates(self):
        stage_config = RegressionStageConfig(
            in_channels=3, out_channels=1, base_width=8, encoder_depths=(1,)
        )
        raw = SimpleNoiseConditionedUNet(stage_config)
        precond = EDMPrecond(raw)
        clean_residual = torch.randn(2, 1, 10, 10)
        conditioning = torch.randn(2, 2, 10, 10)
        loss = edm_loss(precond, clean_residual, conditioning)
        loss.backward()
        grads = [p.grad is not None for p in precond.parameters()]
        self.assertTrue(all(grads))


class KarrasScheduleTest(unittest.TestCase):
    def test_schedule_matches_avila_defaults_boundaries(self):
        sigmas = karras_sigma_schedule(18, sigma_min=0.002, sigma_max=80.0, rho=7.0)
        self.assertEqual(sigmas.shape[0], 19)  # N steps + trailing zero
        self.assertAlmostEqual(sigmas[0].item(), 80.0, places=3)
        self.assertAlmostEqual(sigmas[-2].item(), 0.002, places=3)
        self.assertEqual(sigmas[-1].item(), 0.0)

    def test_schedule_is_monotonically_decreasing(self):
        sigmas = karras_sigma_schedule(10, sigma_min=0.01, sigma_max=10.0, rho=7.0)
        diffs = sigmas[1:] - sigmas[:-1]
        self.assertTrue((diffs <= 0).all())


class HeunSamplerTest(unittest.TestCase):
    def test_deterministic_churn_zero_is_reproducible(self):
        stage_config = RegressionStageConfig(
            in_channels=3, out_channels=1, base_width=8, encoder_depths=(1,)
        )
        raw = SimpleNoiseConditionedUNet(stage_config)
        precond = EDMPrecond(raw)
        conditioning = torch.randn(1, 2, 8, 8)
        config = HeunSamplerConfig(num_steps=4, s_churn=0.0)

        sample_a = heun_sample(precond, conditioning, (1, 1, 8, 8), config,
                                generator=torch.Generator().manual_seed(7))
        sample_b = heun_sample(precond, conditioning, (1, 1, 8, 8), config,
                                generator=torch.Generator().manual_seed(7))
        self.assertTrue(torch.allclose(sample_a, sample_b))
        self.assertEqual(sample_a.shape, (1, 1, 8, 8))
        self.assertTrue(torch.isfinite(sample_a).all())

    def test_churn_changes_output_via_extra_noise(self):
        stage_config = RegressionStageConfig(
            in_channels=3, out_channels=1, base_width=8, encoder_depths=(1,)
        )
        raw = SimpleNoiseConditionedUNet(stage_config)
        precond = EDMPrecond(raw)
        conditioning = torch.randn(1, 2, 8, 8)

        no_churn = HeunSamplerConfig(num_steps=4, s_churn=0.0)
        with_churn = HeunSamplerConfig(num_steps=4, s_churn=7.0, s_tmax=float("inf"))

        gen1 = torch.Generator().manual_seed(3)
        sample_no_churn = heun_sample(precond, conditioning, (1, 1, 8, 8), no_churn, generator=gen1)
        gen2 = torch.Generator().manual_seed(3)
        sample_with_churn = heun_sample(precond, conditioning, (1, 1, 8, 8), with_churn, generator=gen2)
        self.assertFalse(torch.allclose(sample_no_churn, sample_with_churn))

    def test_gamma_capped_at_sqrt2_minus_one(self):
        # churn/N >> sqrt(2)-1 must still cap gamma (Avila 2026 Eq. 7).
        config = HeunSamplerConfig(num_steps=2, s_churn=1000.0)
        gamma = min(config.s_churn / config.num_steps, math.sqrt(2.0) - 1.0)
        self.assertAlmostEqual(gamma, math.sqrt(2.0) - 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
