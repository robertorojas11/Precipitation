"""Stage 1 regression network for Candidate B (Avila 2026 / Mardani et al.
residual-diffusion recipe -- PASO 5.2 of
docs/decisions/0002-implementacion-v2.md).

Avila (2026) §3, Eq. (1): the high-resolution target is decomposed into a
predictable conditional-mean part and a stochastic residual,

    y = E[y|x] + (y - E[y|x])

Stage 1 (this module) predicts ``E[y|x]``; stage 2
(``src/training/diffusion_residual.py``'s ``ResidualDiffusionModel``) learns
the residual's distribution. Avila (2026) §3.1: "a convolutional network
with an encoder-decoder structure and skip connections, of the U-Net type,
following the Ablated Diffusion Model (ADM) design of Dhariwal and Nichol
[3] ... applies self attention at the coarsest internal resolution ...
base width of 64 feature channels". Trained under MSE (§3.1): "the function
that minimises it is exactly the conditional mean, which is precisely the
quantity the second stage assumes has been removed when it models the
residual". Reference optimizer settings (Avila 2026 Table 5, to be tuned to
this project's data scale, not copied blindly): AdamW, lr 2e-4, weight
decay 1e-5.

Unlike Avila's wind-vector target (which may be negative), this project's
target is precipitation, so the head applies Softplus by default
(``nonnegative_output=True``) -- configurable in case a future residual
target (e.g. a signed log-anomaly) needs an unconstrained output.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from src.models.blocks import DownBlock2D, ResidualBlock2D, SelfAttention2D, coordinate_channels


@dataclass(frozen=True)
class RegressionStageConfig:
    in_channels: int = 4          # e.g. ERA5 cp+lsp (interpolated) + invariants
    out_channels: int = 1         # precipitation mean field
    base_width: int = 64          # Avila 2026 Table 5
    encoder_depths: tuple[int, ...] = (1, 1, 1)  # residual blocks per encoder stage
    attention_at_coarsest: bool = True
    add_coordinates: bool = True
    nonnegative_output: bool = True

    @property
    def input_channels_total(self) -> int:
        return self.in_channels + (2 if self.add_coordinates else 0)


class _UpBlock2D(nn.Module):
    """Bilinear upsample + concat skip + residual block (U-Net decoder stage)."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.block = ResidualBlock2D(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(torch.cat([x, skip], dim=1))
        return self.block(x)


class RegressionUNet(nn.Module):
    """Encoder-decoder U-Net predicting the conditional mean field.

    ``forward(x)`` returns ``(B, out_channels, H, W)`` at the same spatial
    size as the input (all conditioning is expected pre-interpolated to the
    target/HR grid, matching the rest of this project's models).
    """

    def __init__(self, config: RegressionStageConfig | None = None):
        super().__init__()
        self.config = config or RegressionStageConfig()
        c = self.config
        width = c.base_width

        self.stem = nn.Conv2d(c.input_channels_total, width, 3, padding=1, padding_mode="reflect")

        widths = [width * (2 ** i) for i in range(len(c.encoder_depths) + 1)]
        self.down_stages = nn.ModuleList()
        for stage, depth in enumerate(c.encoder_depths):
            in_w, out_w = widths[stage], widths[stage + 1]
            blocks = [DownBlock2D(in_w, out_w)]
            blocks += [ResidualBlock2D(out_w, out_w) for _ in range(depth - 1)]
            self.down_stages.append(nn.ModuleList(blocks))

        bottleneck_width = widths[-1]
        self.bottleneck = ResidualBlock2D(bottleneck_width, bottleneck_width)
        self.attention = (
            SelfAttention2D(bottleneck_width) if c.attention_at_coarsest else nn.Identity()
        )

        self.up_stages = nn.ModuleList()
        for stage in reversed(range(len(c.encoder_depths))):
            in_w, skip_w, out_w = widths[stage + 1], widths[stage], widths[stage]
            self.up_stages.append(_UpBlock2D(in_w, skip_w, out_w))

        self.head = nn.Conv2d(width, c.out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.add_coordinates:
            b, _, h, w = x.shape
            x = torch.cat([x, coordinate_channels(b, h, w, x.device, x.dtype)], dim=1)

        skips = []
        h = self.stem(x)
        skips.append(h)
        for stage in self.down_stages:
            for block in stage:
                h = block(h)
            skips.append(h)
        skips.pop()  # last skip is the bottleneck input itself, not used as a skip

        h = self.attention(self.bottleneck(h))
        for up, skip in zip(self.up_stages, reversed(skips)):
            h = up(h, skip)

        out = self.head(h)
        return F.softplus(out) if self.config.nonnegative_output else out
