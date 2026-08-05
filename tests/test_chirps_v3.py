import unittest
import numpy as np
import torch
import pipeline_v3

from src.chirps_v3.config import FOLDS, get_fold
from src.chirps_v3.dataset import context_offsets, _terrain_features
from src.chirps_v3.feasibility import masked_coarse_oracle
from src.chirps_v3.metrics import PooledMetrics
from src.chirps_v3.model import TemporalTerrainNet
from src.chirps_v3.training import EMA, r2_aligned_loss


class V3ContractTests(unittest.TestCase):
    def test_rolling_folds_are_strictly_chronological(self):
        for fold in FOLDS:
            self.assertLess(fold.train_end, fold.validation_start)
            self.assertTrue(get_fold(fold.name).contains(fold.validation_end, "validation"))

    def test_centered_context(self):
        self.assertEqual(list(context_offsets(5)), [-2, -1, 0, 1, 2])

    def test_training_resolves_all_prerequisites(self):
        self.assertEqual(pipeline_v3.selected_stages(["train_search"]), [
            "storage_check", "contract", "prepare_folds", "feasibility",
            "search_manifest", "train_search",
        ])


class V3MathTests(unittest.TestCase):
    def test_pooled_r2(self):
        metric = PooledMetrics(); obs = np.arange(4.0); metric.update(obs, obs, np.ones(4, bool))
        self.assertEqual(metric.result()["r2"], 1.0)

    def test_oracle_preserves_constant_field(self):
        target = torch.full((1, 1, 10, 10), 3.0); mask = torch.ones_like(target)
        torch.testing.assert_close(masked_coarse_oracle(target, mask, 5), target)

    def test_masked_loss_ignores_invalid_pixel(self):
        target = torch.ones(1, 1, 3, 3); mask = torch.ones_like(target, dtype=torch.bool); mask[..., -1, -1] = False
        output = {"prediction_mm": target.clone(), "occurrence_logits": torch.full_like(target, 1.3862944),
                  "wet_probability": torch.full_like(target, .8), "log_amount": torch.log1p(target)}
        first, _ = r2_aligned_loss(output, target, mask); output["prediction_mm"][..., -1, -1] = 1e6
        second, _ = r2_aligned_loss(output, target, mask); self.assertAlmostEqual(first.item(), second.item(), places=5)

    def test_model_shapes_and_ema(self):
        model = TemporalTerrainNet(width=8)
        output = model(torch.randn(2, 3, 18, 4, 6), torch.randn(2, 3, 8, 12),
                       torch.randn(2, 7, 16, 24), torch.rand(2, 1, 16, 24), torch.randn(2, 2))
        self.assertEqual(output["prediction_mm"].shape, (2, 1, 16, 24)); ema = EMA(model); ema.update(model)
        self.assertEqual(output["occurrence_logits"].shape, output["wet_probability"].shape)

    def test_loss_is_safe_under_autocast(self):
        model = TemporalTerrainNet(width=8)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = model(torch.randn(1, 1, 18, 4, 6), torch.randn(1, 3, 8, 12),
                           torch.randn(1, 7, 16, 24), torch.rand(1, 1, 16, 24), torch.randn(1, 2))
            loss, _ = r2_aligned_loss(output, torch.rand(1, 1, 16, 24),
                                      torch.ones(1, 1, 16, 24, dtype=torch.bool))
        self.assertTrue(torch.isfinite(loss))

    def test_terrain_feature_shape(self):
        self.assertEqual(_terrain_features(torch.randn(3, 4, 6), (8, 12)).shape, (7, 8, 12))


if __name__ == "__main__": unittest.main()
