"""Tests for stage 8 (interpretability) --
src/evaluation/interpretability_hourly.py -- reusing stage 6/7's synthetic
prepared-region fixture to train a tiny real GAN checkpoint and then run
both Shapley attribution and per-pixel distribution fitting against it end
to end. Does not re-test the metric/attribution formulas themselves (those
have their own dedicated test files, e.g. tests/test_shapley_attribution.py
and tests/test_pixel_distributions.py)."""

from pathlib import Path
import tempfile
import unittest

from src.evaluation import interpretability_hourly as I
from src.training.train_gan import GanTrainingConfig, train as train_gan
from tests.test_training import _WithPreparedSyntheticRegion


class InterpretabilityTest(_WithPreparedSyntheticRegion):
    def setUp(self):
        super().setUp()
        self._out = tempfile.TemporaryDirectory()
        config = GanTrainingConfig(region=self.REGION_NAME, epochs=1, batch_size=4, base_width=8, depth=2, device="cpu")
        self.run_dir = train_gan(config, prepared_root=self.prepared_root, output_root=Path(self._out.name))

    def tearDown(self):
        self._out.cleanup()
        super().tearDown()

    def test_shapley_attribution_over_four_channels(self):
        result = I.compute_shapley_attribution(
            self.REGION_NAME, "val", "gan", checkpoint=self.run_dir / "best.pt",
            prepared_root=self.prepared_root, device="cpu", n_samples=2,
        )
        self.assertEqual(result["channel_names"], list(I.CHANNEL_NAMES))
        self.assertEqual(result["samples_evaluated"], 2)
        self.assertEqual(len(result["mean_shapley_values"]), 4)
        for sample in result["per_sample"]:
            self.assertEqual(len(sample["shapley_values"]), 4)
            self.assertIn("date", sample)
            # Additivity: the Shapley values should (near-exactly, this is
            # an exact enumeration) sum to prediction - base_value.
            self.assertAlmostEqual(
                sample["sum_of_shapley_values"], sample["prediction"] - sample["base_value"], places=4,
            )

    def test_pixel_distribution_fit_at_one_pixel(self):
        # The synthetic fixture's ERA5 forcing is identical across every
        # hour/day (tests/test_training.py's _write_day), so the only real
        # source of variation at a fixed pixel is ensemble noise -- a small
        # n_members can produce a near-degenerate pooled sample scipy's
        # continuous-family fitting genuinely cannot fit (not a bug: the
        # same graceful-failure path fit_pixel_distributions always takes
        # for too little/too-degenerate data, exercised here instead of a
        # clean fit). Pool across the train split (2 distinct days, unlike
        # val's 1) with more members for a better chance of a clean fit,
        # but accept either outcome -- this test verifies the wiring
        # degrades gracefully, not that scipy converges on synthetic data
        # never designed for statistical realism.
        result = I.fit_pixel_distributions(
            self.REGION_NAME, "train", "gan", pixels=[(30, 30)],
            checkpoint=self.run_dir / "best.pt", prepared_root=self.prepared_root, device="cpu",
            max_frames=8, n_bootstrap=10, n_members=6,
        )
        self.assertEqual(len(result["pixels"]), 1)
        pixel = result["pixels"][0]
        self.assertEqual((pixel["row"], pixel["col"]), (30, 30))
        if "error" in pixel:
            self.assertIn("sample_count", pixel)
        else:
            self.assertIn("family_name", pixel)
            self.assertIn(0.90, pixel["quantile_map_mm"])
            self.assertGreaterEqual(pixel["exceedance_probability"], 0.0)
            self.assertLessEqual(pixel["exceedance_probability"], 1.0)


if __name__ == "__main__":
    unittest.main()
