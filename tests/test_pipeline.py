import unittest

import numpy as np
import torch

from src.data_preprocessing.quality import (
    aggregate_oya_slots_numpy,
    inspect_precipitation,
    masked_r2,
)
from src.training.train import compute_losses


class PrecipitationQCTests(unittest.TestCase):
    def test_rejects_sentinels_and_implausible_values(self):
        values = np.array([[0.0, 10.0], [3.4028235e38, 1001.0]])
        qc, mask = inspect_precipitation(values, max_daily_mm=1000.0, min_valid_fraction=0.0)
        self.assertFalse(qc.accepted)
        self.assertEqual(qc.sentinel_count, 1)
        self.assertEqual(qc.implausible_count, 1)
        np.testing.assert_array_equal(mask, [[True, True], [False, False]])

    def test_oya_aggregation_requires_thirty_slots(self):
        slots = np.ones((48, 2, 1), dtype=np.float32)
        slots[29:, 1, 0] = np.nan
        daily, count, coverage = aggregate_oya_slots_numpy(slots)
        self.assertAlmostEqual(float(daily[0, 0]), 24.0)
        self.assertEqual(int(count[0, 0]), 48)
        self.assertEqual(int(count[1, 0]), 29)
        self.assertFalse(bool(coverage[1, 0]))
        self.assertTrue(np.isnan(daily[1, 0]))

    def test_masked_r2_ignores_invalid_pixels(self):
        obs = np.array([1.0, 2.0, 3.0, 999.0])
        pred = np.array([1.0, 2.0, 3.0, -999.0])
        self.assertEqual(masked_r2(pred, obs, np.array([1, 1, 1, 0], dtype=bool)), 1.0)


class MaskedLossTests(unittest.TestCase):
    def test_invalid_pixel_does_not_change_loss(self):
        target_norm = torch.zeros((1, 1, 2, 2))
        mask = torch.tensor([[[[True, True], [True, False]]]])
        base = {
            "prediction_mm": torch.ones((1, 1, 2, 2)),
            "occurrence_logits": torch.zeros((1, 1, 2, 2)),
            "positive_log_amount": torch.ones((1, 1, 2, 2)),
        }
        loss_a, _ = compute_losses(base, target_norm, mask, 0.0, 1.0, 2.0)
        changed = {key: value.clone() for key, value in base.items()}
        for value in changed.values():
            value[..., 1, 1] = 1e6
        loss_b, _ = compute_losses(changed, target_norm, mask, 0.0, 1.0, 2.0)
        self.assertAlmostEqual(loss_a.item(), loss_b.item(), places=5)


if __name__ == "__main__":
    unittest.main()
