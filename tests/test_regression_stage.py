"""Shape / behaviour contract for the Candidate B stage-1 regression U-Net."""

import unittest

import torch

from src.models.regression_stage import RegressionStageConfig, RegressionUNet


class RegressionUNetTest(unittest.TestCase):
    def test_output_shape_matches_input_grid(self):
        config = RegressionStageConfig(
            in_channels=4, out_channels=1, base_width=8, encoder_depths=(1, 1, 1)
        )
        model = RegressionUNet(config)
        x = torch.rand(2, 4, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (2, 1, 32, 32))

    def test_output_is_nonnegative_by_default(self):
        model = RegressionUNet(RegressionStageConfig(in_channels=3, base_width=8, encoder_depths=(1, 1)))
        x = torch.randn(1, 3, 24, 24)
        out = model(x)
        self.assertGreaterEqual(out.min().item(), 0.0)

    def test_signed_output_when_configured(self):
        model = RegressionUNet(
            RegressionStageConfig(in_channels=2, base_width=8, encoder_depths=(1,), nonnegative_output=False)
        )
        x = torch.randn(1, 2, 16, 16) * 10.0
        out = model(x)
        self.assertTrue((out < 0).any() or (out > 0).any())

    def test_handles_non_power_of_two_spatial_size(self):
        model = RegressionUNet(RegressionStageConfig(in_channels=2, base_width=8, encoder_depths=(1, 1)))
        x = torch.rand(1, 2, 25, 37)
        out = model(x)
        self.assertEqual(out.shape, (1, 1, 25, 37))

    def test_backprop_reaches_all_parameters(self):
        model = RegressionUNet(RegressionStageConfig(in_channels=2, base_width=8, encoder_depths=(1, 1)))
        x = torch.rand(1, 2, 16, 16)
        model(x).mean().backward()
        missing = [name for name, p in model.named_parameters() if p.grad is None]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
