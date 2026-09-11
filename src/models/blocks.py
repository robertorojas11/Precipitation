"""Shared convolutional building blocks for the v2-generative models.

These mirror the residual-block design of P1 (spateGAN-ERA5), Fig. 5
"Residual Block": Conv -> Norm -> activation -> Conv -> Norm, plus a
1x1-convolution skip when the channel count changes, and a final activation.
P1 uses 3D convolutions over (time, y, x); this project's Candidate A
downscales hourly -> hourly with no temporal super-resolution
(docs/decisions/0002-implementacion-v2.md §1.1), so we use 2D convolutions
and fold any short temporal context into the channel dimension (see
``generator.py``).
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _norm(channels: int) -> nn.Module:
    # P1 uses instance normalisation in the generator. GroupNorm with
    # num_groups clamped to the channel count is the stable small-batch
    # equivalent used across this repo (legacy/deterministic_unet/multiscale_unet.py).
    return nn.GroupNorm(min(8, channels), channels)


class ResidualBlock2D(nn.Module):
    """P1-style residual block (2D). ``dropout`` enables ensemble perturbation."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode="reflect"),
            _norm(out_channels),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode="reflect"),
            _norm(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), _norm(out_channels))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.skip(x) + self.body(x))


class DownBlock2D(nn.Module):
    """Strided residual downsample used by the discriminator."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.norm = _norm(out_channels)
        self.block = ResidualBlock2D(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(F.silu(self.norm(self.reduce(x))))


class SelfAttention2D(nn.Module):
    """Single-head spatial self-attention, applied at the coarsest internal
    resolution (Avila 2026 §3.1: "self attention is the only mechanism that
    lets distant grid points coordinate with one another", used so an eyewall
    on one side of a storm implies the opposite side; here reused generically
    as "auto-atencion en la resolucion interna mas gruesa", PASO 5.2).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm = _norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)
        attn = torch.softmax((q @ k) * self.scale, dim=-1)
        out = (attn @ v).permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj(out)


def coordinate_channels(batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
    """Return a (batch, 2, H, W) normalised lat/lon grid, as in the legacy model."""
    lat = torch.linspace(1.0, -1.0, height, device=device, dtype=dtype).view(1, 1, height, 1)
    lon = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype).view(1, 1, 1, width)
    lat = lat.expand(batch, 1, height, width)
    lon = lon.expand(batch, 1, height, width)
    return torch.cat([lat, lon], dim=1)
