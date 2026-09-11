"""Deterministic multiscale precipitation model for the clean v2 pipeline."""

from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        groups = min(8, channels)
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.silu(inputs + self.layers(inputs))


class FeatureStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.blocks = nn.Sequential(
            ResidualBlock(out_channels, dropout),
            ResidualBlock(out_channels, dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.blocks(F.silu(self.project(inputs)))


class MultiscalePrecipUNet(nn.Module):
    """Decode 25 km atmospheric fields with 10 km physical conditioning to 5 km."""

    def __init__(
        self,
        *,
        base_width: int = 32,
        dropout: float = 0.0,
        target_mean: float,
        target_std: float,
        atmospheric_channels: int = 18,
    ):
        super().__init__()
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.encoder_25 = FeatureStage(atmospheric_channels, base_width, dropout)
        # 3 physical channels + lat/lon + seasonal sin/cos + ERA5 baseline.
        self.decoder_10 = FeatureStage(base_width + 8, base_width * 2, dropout)
        self.decoder_5 = FeatureStage(base_width * 2 + 5, base_width, dropout)
        self.occurrence_head = nn.Conv2d(base_width, 1, 1)
        self.amount_residual_head = nn.Conv2d(base_width, 1, 1)

    @staticmethod
    def _coordinates(batch: int, height: int, width: int, device, dtype):
        latitude = torch.linspace(1.0, -1.0, height, device=device, dtype=dtype)
        longitude = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        lat = latitude.view(1, 1, height, 1).expand(batch, 1, height, width)
        lon = longitude.view(1, 1, 1, width).expand(batch, 1, height, width)
        return lat, lon

    def forward(
        self,
        inputs_25km: torch.Tensor,
        phys_dem_10km: torch.Tensor,
        era5_precip_5km_norm: torch.Tensor,
        season: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = inputs_25km.shape[0]
        encoded = self.encoder_25(inputs_25km)
        encoded_10 = F.interpolate(encoded, size=(230, 360), mode="bilinear", align_corners=False)
        lat10, lon10 = self._coordinates(batch, 230, 360, encoded.device, encoded.dtype)
        season10 = season[:, :, None, None].expand(batch, 2, 230, 360)
        baseline10 = F.interpolate(era5_precip_5km_norm, size=(230, 360), mode="area")
        decoded_10 = self.decoder_10(
            torch.cat([encoded_10, phys_dem_10km, lat10, lon10, season10, baseline10], dim=1)
        )

        decoded_up = F.interpolate(decoded_10, size=(460, 720), mode="bilinear", align_corners=False)
        lat5, lon5 = self._coordinates(batch, 460, 720, encoded.device, encoded.dtype)
        season5 = season[:, :, None, None].expand(batch, 2, 460, 720)
        features_5 = self.decoder_5(
            torch.cat([decoded_up, lat5, lon5, season5, era5_precip_5km_norm], dim=1)
        )
        occurrence_logits = self.occurrence_head(features_5)
        baseline_log = era5_precip_5km_norm * self.target_std + self.target_mean
        positive_log_amount = F.softplus(baseline_log + self.amount_residual_head(features_5))
        positive_amount = torch.expm1(positive_log_amount).clamp_min(0.0)
        wet_probability = torch.sigmoid(occurrence_logits)
        deterministic_mm = wet_probability * positive_amount
        prediction_norm = (torch.log1p(deterministic_mm) - self.target_mean) / self.target_std
        return {
            "prediction_norm": prediction_norm,
            "prediction_mm": deterministic_mm,
            "occurrence_logits": occurrence_logits,
            "positive_log_amount": positive_log_amount,
        }
