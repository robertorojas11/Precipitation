"""Conditional generator for probabilistic precipitation downscaling
(Candidate A, P1/spateGAN-ERA5-inspired -- PASO 5.1 of
docs/decisions/0002-implementacion-v2.md).

Design, and how it maps to spateGAN-ERA5 (P1):

* **Architecture family.** P1's generator (P1 "Model description", Fig. 5) is a
  four-stage network of 3D-convolutional residual blocks (Res3D) that increases
  spatial resolution 12x and temporal resolution 6x, ending in three Res3D
  blocks and a **Softplus** activation to keep the output non-negative
  (matrix M131-M135). We keep the residual-block backbone and the Softplus tail
  (M135) but use **2D** convolutions.

* **Why 2D (temporal-dimension decision).** The task is hourly ERA5 ->
  hourly IMERG (decision 0002, §1.1/1.4): there is no temporal
  super-resolution, so a 3D generator over a long time axis is not needed. A
  short context of ``context_hours`` consecutive hours (default 1) is
  supported and folded into the channel dimension, which preserves the "the
  model sees temporal context" property of P1 without a Res3D stack. If
  sub-hourly output is ever pursued, swap ``ResidualBlock2D`` for a Res3D
  variant and add a time axis.

* **Spatial factor.** Following P2 (RainShift), the low-resolution ERA5 inputs
  are bilinearly interpolated to the IMERG grid *before* the network (matrix
  M220), so the generator is fully convolutional at high resolution and grid /
  factor agnostic (P2 uses N = 2.5; matrix M205).

* **Ensemble (M136, M160).** Members differ through (a) an explicit latent-noise
  map concatenated to the conditioning and (b) dropout left active at inference.
  ``sample(condition, n_members, seed)`` draws ``n_members`` fields, each with a
  frozen per-member RNG seed -- "we freeze the dropout seed for each produced
  ensemble member" (P1 "Model description").

The generator returns precipitation in the target's physical unit (mm/hr).
The eta-learning regularizer (``src/training/eta_learning.py``) is
model-agnostic and attaches to this generator's output without any
architectural change (PASO 1.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from src.models.blocks import ResidualBlock2D, coordinate_channels


def _rng_devices(device: torch.device) -> list:
    return [device] if device.type == "cuda" else []


@dataclass(frozen=True)
class GeneratorConfig:
    lr_channels: int = 2          # ERA5 cp + lsp (P1 M101)
    invariant_channels: int = 2   # land-sea mask + orography (P2 M202/M206)
    context_hours: int = 1        # hours of ERA5 context, folded into channels
    noise_channels: int = 4       # latent map for ensemble diversity
    base_width: int = 64          # P1 uses 96; 64 is the small-GPU default here
    depth: int = 6                # residual blocks in the trunk
    dropout: float = 0.2          # P1 M136: p = 0.2
    add_coordinates: bool = True

    @property
    def in_channels(self) -> int:
        return (
            self.lr_channels * self.context_hours
            + self.invariant_channels
            + self.noise_channels
            + (2 if self.add_coordinates else 0)
        )


class ConditionalGenerator(nn.Module):
    def __init__(self, config: GeneratorConfig | None = None):
        super().__init__()
        self.config = config or GeneratorConfig()
        c = self.config
        width = c.base_width

        self.stem = nn.Conv2d(c.in_channels, width, 3, padding=1, padding_mode="reflect")
        # Dropout is applied inside residual blocks at three depths (low / mid /
        # high level features), matching P1's "three different generator depths".
        drop_at = {0, c.depth // 2, c.depth - 1}
        self.trunk = nn.ModuleList(
            ResidualBlock2D(width, width, dropout=c.dropout if i in drop_at else 0.0)
            for i in range(c.depth)
        )
        self.head = nn.Sequential(
            ResidualBlock2D(width, width),
            nn.Conv2d(width, 1, 1),
        )

    # -- conditioning -------------------------------------------------------
    def _assemble_input(
        self,
        lr_precip: torch.Tensor,
        invariants: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        parts = [lr_precip, invariants, noise]
        if self.config.add_coordinates:
            b, _, h, w = lr_precip.shape
            parts.append(coordinate_channels(b, h, w, lr_precip.device, lr_precip.dtype))
        return torch.cat(parts, dim=1)

    # -- forward ----------------------------------------------------------
    def forward(
        self,
        lr_precip: torch.Tensor,
        invariants: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One stochastic sample. ``lr_precip`` is already on the HR grid.

        ``lr_precip``: (B, lr_channels * context_hours, H, W)
        ``invariants``: (B, invariant_channels, H, W)
        returns precipitation (B, 1, H, W) in mm/hr, non-negative (Softplus).
        Randomness (latent noise + dropout) is drawn from the ambient torch RNG;
        ``sample()`` seeds it per ensemble member.
        """
        b, _, h, w = lr_precip.shape
        if noise is None:
            noise = torch.randn(
                b, self.config.noise_channels, h, w,
                device=lr_precip.device, dtype=lr_precip.dtype,
            )
        x = self.stem(self._assemble_input(lr_precip, invariants, noise))
        for block in self.trunk:
            x = block(x)
        return F.softplus(self.head(x))

    @torch.no_grad()
    def sample(
        self,
        lr_precip: torch.Tensor,
        invariants: torch.Tensor,
        n_members: int = 3,
        *,
        seed: int | None = None,
        keep_dropout: bool = True,
    ) -> torch.Tensor:
        """Return an ensemble of shape (n_members, B, 1, H, W).

        Each member uses a frozen RNG seed (``seed + member`` when ``seed`` is
        given) so the same input reproduces the same ensemble -- P1 M160.
        Dropout is kept in train mode when ``keep_dropout`` so it perturbs the
        sample (P1 M136); every other layer (norm statistics) is eval mode.
        """
        was_training = self.training
        self.eval()
        if keep_dropout:
            for module in self.modules():
                if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                    module.train()
        members = []
        for m in range(n_members):
            if seed is not None:
                # Freeze the RNG for this member so latent noise AND dropout are
                # reproducible (P1 M160: "freeze the dropout seed for each ...
                # ensemble member"). fork_rng isolates it from the caller.
                with torch.random.fork_rng(devices=_rng_devices(lr_precip.device)):
                    torch.manual_seed(int(seed) + m)
                    members.append(self.forward(lr_precip, invariants))
            else:
                members.append(self.forward(lr_precip, invariants))
        if was_training:
            self.train()
        return torch.stack(members, dim=0)
