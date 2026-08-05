"""Shared temporal encoder and terrain-aware residual decoder."""

from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, source: int, width: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(source, width, 3, padding=1, bias=False),
                                 nn.GroupNorm(min(8, width), width), nn.SiLU(),
                                 nn.Conv2d(width, width, 3, padding=1, bias=False),
                                 nn.GroupNorm(min(8, width), width), nn.SiLU(),
                                 nn.Dropout2d(dropout))
        self.skip = nn.Conv2d(source, width, 1) if source != width else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class TemporalTerrainNet(nn.Module):
    """Encode days with shared weights, fuse time, then decode at 5 km."""

    def __init__(self, atmospheric_channels: int = 18, terrain_channels: int = 7,
                 width: int = 48, dropout: float = 0.1, fusion: str = "attention"):
        super().__init__()
        if fusion not in {"mean", "attention"}:
            raise ValueError("fusion must be mean or attention")
        self.fusion = fusion
        self.day_encoder = Block(atmospheric_channels, width, dropout)
        self.attention = nn.Conv2d(width, 1, 1)
        self.mid = Block(width + 3 + 2, width * 2, dropout)
        self.terrain = Block(terrain_channels, width, dropout)
        self.decoder = nn.Sequential(Block(width * 3 + 1, width * 2, dropout),
                                     Block(width * 2, width, dropout))
        self.residual = nn.Conv2d(width, 1, 1)
        self.occurrence = nn.Conv2d(width, 1, 1)

    def forward(self, atmospheric, physics_10km, terrain_5km, baseline_mm, season):
        batch, days, channels, height, width = atmospheric.shape
        encoded = self.day_encoder(atmospheric.reshape(batch * days, channels, height, width))
        encoded = encoded.reshape(batch, days, -1, height, width)
        if self.fusion == "attention":
            logits = self.attention(encoded.reshape(batch * days, -1, height, width))
            weights = logits.reshape(batch, days, 1, height, width).softmax(1)
            fused = (encoded * weights).sum(1)
        else:
            fused = encoded.mean(1)
        fused = F.interpolate(fused, size=physics_10km.shape[-2:], mode="bilinear", align_corners=False)
        seasonal = season[:, :, None, None].expand(-1, -1, *physics_10km.shape[-2:])
        middle = self.mid(torch.cat((fused, physics_10km, seasonal), 1))
        middle = F.interpolate(middle, size=terrain_5km.shape[-2:], mode="bilinear", align_corners=False)
        terrain = self.terrain(terrain_5km)
        log_baseline = torch.log1p(baseline_mm)
        decoded = self.decoder(torch.cat((middle, terrain, log_baseline), 1))
        log_amount = log_baseline + self.residual(decoded)
        occurrence_logits = self.occurrence(decoded)
        wet_probability = torch.sigmoid(occurrence_logits)
        prediction = wet_probability * torch.expm1(log_amount.clamp(-10, 8)).clamp_min(0)
        return {"prediction_mm": prediction, "occurrence_logits": occurrence_logits,
                "wet_probability": wet_probability, "log_amount": log_amount}
