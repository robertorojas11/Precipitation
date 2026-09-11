"""Streaming physical-unit metrics used consistently by search and evaluation."""

from __future__ import annotations
import math
import numpy as np


class PooledMetrics:
    def __init__(self) -> None:
        self.n = 0
        self.obs_sum = self.obs_sq = self.sse = self.sae = 0.0

    def update(self, prediction, observation, mask) -> None:
        valid = np.asarray(mask, dtype=bool)
        pred = np.asarray(prediction)[valid].astype(np.float64)
        obs = np.asarray(observation)[valid].astype(np.float64)
        finite = np.isfinite(pred) & np.isfinite(obs)
        pred, obs = pred[finite], obs[finite]
        self.n += obs.size
        self.obs_sum += float(obs.sum()); self.obs_sq += float(np.square(obs).sum())
        self.sse += float(np.square(obs - pred).sum()); self.sae += float(np.abs(obs - pred).sum())

    def result(self) -> dict:
        denominator = self.obs_sq - self.obs_sum ** 2 / max(self.n, 1)
        return {"r2": 1 - self.sse / denominator if denominator > 0 else None,
                "rmse": math.sqrt(self.sse / max(self.n, 1)),
                "mae": self.sae / max(self.n, 1), "valid_pixels": self.n}
