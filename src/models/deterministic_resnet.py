"""Deterministic fully-convolutional ResNet baseline (P2 contestant, M226;
PASO 5.4 of docs/decisions/0002-implementacion-v2.md -- the "third
comparison point" alongside Candidate A and Candidate B).

P2 (RainShift) uses "a fully-convolutional ResNet as our deterministic
baseline ... follow[ing] prior work that has demonstrated strong performance
in precipitation downscaling" (Harris et al. 2022). It is the deterministic
reference point for which "CRPS reduces to MAE" (P1 M181, P2 M247).

Same conditioning contract as ``ConditionalGenerator`` (LR ERA5 already
bilinearly interpolated to the HR grid, plus HR invariant features), but no
noise input and no dropout at inference: one deterministic field per input.
Non-negative output via Softplus. This is a fresh module for the hourly
ERA5->IMERG contract -- it does **not** reuse
``legacy/deterministic_unet/multiscale_unet.py``, which stays archived and
runnable for the daily CHIRPS/Oya task (legacy/README.md).

Per PASO 5.4, this baseline is also evaluated with the eta-learning
regularizer optionally activated (``src/training/eta_learning.py``), to
measure whether it helps even a simple deterministic estimator -- the
eta-learning paper validates this case too, not only generative models.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from src.models.blocks import ResidualBlock2D, coordinate_channels


@dataclass(frozen=True)
class ResNetConfig:
    lr_channels: int = 2
    invariant_channels: int = 2
    context_hours: int = 1
    base_width: int = 64
    depth: int = 8
    add_coordinates: bool = True
    predict_residual_over_input: bool = True

    @property
    def in_channels(self) -> int:
        return (
            self.lr_channels * self.context_hours
            + self.invariant_channels
            + (2 if self.add_coordinates else 0)
        )


class DeterministicResNet(nn.Module):
    def __init__(self, config: ResNetConfig | None = None):
        super().__init__()
        self.config = config or ResNetConfig()
        c = self.config
        self.stem = nn.Conv2d(c.in_channels, c.base_width, 3, padding=1, padding_mode="reflect")
        self.trunk = nn.Sequential(
            *(ResidualBlock2D(c.base_width, c.base_width) for _ in range(c.depth))
        )
        self.head = nn.Conv2d(c.base_width, 1, 1)

    def forward(
        self,
        lr_precip: torch.Tensor,
        invariants: torch.Tensor,
        total_precip_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return deterministic precipitation (B, 1, H, W) in mm/hr, non-negative.

        ``total_precip_input`` (B, 1, H, W) is the interpolated ERA5 total
        precipitation on the HR grid; when given and ``predict_residual_over_
        input`` the network predicts an additive log-space residual over it
        (P2 M220 residual-learning idea), otherwise it predicts the field.
        """
        parts = [lr_precip, invariants]
        if self.config.add_coordinates:
            b, _, h, w = lr_precip.shape
            parts.append(coordinate_channels(b, h, w, lr_precip.device, lr_precip.dtype))
        x = self.trunk(self.stem(torch.cat(parts, dim=1)))
        out = self.head(x)
        if self.config.predict_residual_over_input and total_precip_input is not None:
            base = torch.log1p(total_precip_input.clamp_min(0.0))
            # Softly bound the raw residual before it meets softplus -- a hard
            # clamp was tried first and rejected: clamp's gradient is exactly
            # zero outside its range, so as soon as training pushes `out` past
            # the boundary it freezes the trunk just as dead as the original
            # bug, only shallower. `tanh` instead has a gradient that only
            # *asymptotically* vanishes, so it stays representable over a much
            # wider range. Measured on a real checkpoint (run4, epoch 11): with
            # the target mostly zero rain, plain L1 drives the unbounded
            # pre-activation to mean -30 within the very first epoch --
            # softplus' gradient (sigmoid(x)) underflows to an exact 0.0 that
            # deep, freezing every weight update for the rest of training
            # (predictions stay a literal 0.0 everywhere, train/val loss
            # unchanged to 4 decimals for 10+ epochs straight). With this
            # bound, the same raw -30 output still carries a representable
            # gradient (~5e-7, measured) instead of an exact zero. Scale 10.0
            # matches log1p of this dataset's clip ceiling (~130mm/hr, see
            # norm_stats era5_clip_range) at ~4.9, so it covers a full-scale
            # correction in either direction before saturating.
            out = 10.0 * torch.tanh(out / 10.0)
            return torch.expm1(F.softplus(base + out)).clamp_min(0.0)
        return F.softplus(out)
