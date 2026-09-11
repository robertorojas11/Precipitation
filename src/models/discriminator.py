"""Conditional discriminator for the P1-inspired precipitation GAN
(Candidate A, PASO 5.1 of docs/decisions/0002-implementacion-v2.md).

Maps to spateGAN-ERA5 (P1), "Model description" and Fig. 5:

* Inputs are the high-resolution field (real observation or generator output)
  **and** the coarse-resolution ERA5 context provided to the generator
  (matrix M141). Its objective is binary: real vs. generated (M146, P1 Eq. 1).
* **Anti-quantization noise (M142).** Gaussian noise with mean 1, std 0.05 is
  multiplied into the inputs before any convolution -- "to prevent the model
  from learning to distinguish rainfields based on quantization characteristics".
* **Separate HR / LR paths concatenated late (M143).** The HR field and the LR
  context are processed by independent residual stacks and only merged near the
  end "to encourage a comparison based on latent features extracted on multiple
  resolutions".
* Feature widths follow P1's ratios (M145): HR path 128, 128, 128, 64; LR path
  64, 32; after concatenation a block reduces to 64, then a final 1x1 conv to a
  single logit.

The discriminator returns a raw logit; the BCE is applied in
``src/training/losses_gan.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.blocks import DownBlock2D, ResidualBlock2D


@dataclass(frozen=True)
class DiscriminatorConfig:
    hr_channels: int = 1          # precipitation field
    lr_channels: int = 2          # ERA5 cp + lsp context (on the HR grid)
    invariant_channels: int = 2   # land-sea mask + orography, shared context
    hr_widths: tuple[int, ...] = (128, 128, 128, 64)
    lr_widths: tuple[int, ...] = (64, 32)
    noise_mean: float = 1.0       # P1 M142
    noise_std: float = 0.05       # P1 M142


class ConditionalDiscriminator(nn.Module):
    def __init__(self, config: DiscriminatorConfig | None = None):
        super().__init__()
        self.config = config or DiscriminatorConfig()
        c = self.config

        hr_in = c.hr_channels + c.invariant_channels
        self.hr_stem = nn.Conv2d(hr_in, c.hr_widths[0], 3, padding=1)
        self.hr_path = nn.ModuleList()
        prev = c.hr_widths[0]
        for width in c.hr_widths:
            self.hr_path.append(DownBlock2D(prev, width))
            prev = width
        hr_out = prev

        lr_in = c.lr_channels + c.invariant_channels
        self.lr_stem = nn.Conv2d(lr_in, c.lr_widths[0], 3, padding=1)
        self.lr_path = nn.ModuleList()
        prev = c.lr_widths[0]
        for width in c.lr_widths:
            self.lr_path.append(DownBlock2D(prev, width))
            prev = width
        lr_out = prev

        self.merge = ResidualBlock2D(hr_out + lr_out, 64)
        self.classifier = nn.Conv2d(64, 1, 1)

    def _perturb(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        noise = torch.randn_like(x) * self.config.noise_std + self.config.noise_mean
        return x * noise

    def forward(
        self,
        hr_field: torch.Tensor,
        lr_context: torch.Tensor,
        invariants: torch.Tensor,
    ) -> torch.Tensor:
        """Return a per-sample logit (B, 1, h', w').

        ``hr_field``: (B, 1, H, W); ``lr_context``: (B, lr_channels, H, W)
        (ERA5 already interpolated to the HR grid); ``invariants``: (B, d, H, W).
        A patch-style logit map is returned; callers reduce with ``.mean()``.
        """
        hr = self._perturb(torch.cat([hr_field, invariants], dim=1))
        hr = self.hr_stem(hr)
        for block in self.hr_path:
            hr = block(hr)

        lr = self._perturb(torch.cat([lr_context, invariants], dim=1))
        lr = self.lr_stem(lr)
        for block in self.lr_path:
            lr = block(lr)

        if lr.shape[-2:] != hr.shape[-2:]:
            lr = nn.functional.interpolate(lr, size=hr.shape[-2:], mode="bilinear",
                                            align_corners=False)
        merged = self.merge(torch.cat([hr, lr], dim=1))
        return self.classifier(merged)
