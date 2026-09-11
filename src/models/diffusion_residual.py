r"""Stage 2 residual diffusion model for Candidate B, on the EDM framework
of Karras et al. 2022 (PASO 5.2 of docs/decisions/0002-implementacion-v2.md,
following Avila 2026 §3.2/3.3, which documents the exact equations used
here).

**Preconditioning (Avila 2026 Eq. 3-4, EDM).** The raw network ``F`` is
wrapped into a denoiser ``D``:

    D(z; sigma) = c_skip(sigma) * z + c_out(sigma) * F(c_in(sigma) * z, c_noise(sigma))   (Eq. 3)

    c_skip = sigma_data^2 / (sigma^2 + sigma_data^2)
    c_out  = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
    c_in   = 1 / sqrt(sigma_data^2 + sigma^2)
    c_noise = (1/4) * ln(sigma)                                                          (Eq. 4)

``z`` is the noisy standardized residual; ``sigma`` is the noise level (the
std of the Gaussian noise currently added). ``F``'s conditioning input is
the concatenation, along the channel axis, of ``c_in(sigma) * z``, the
upsampled conditioning channels, and the stage-1 mean field (Avila 2026
§3.2: "sixteen input channels and two output channels" in his setup; here
the channel count is whatever the caller's conditioning provides).

**Training loss (Avila 2026 Eq. 5).**

    L = E_{sigma,n}[ lambda(sigma) * || D(r' + sigma*n; sigma) - r' ||^2 ]
    lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2

``sigma`` is drawn per example from ``ln(sigma) ~ N(P_mean, P_std^2)``, with
``P_mean=-1.2``, ``P_std=1.2`` (Karras et al.'s recommended values, reused
unless local data suggests otherwise -- Avila 2026 Table 5).

**Sampling: 2nd-order Heun + stochastic churn (Avila 2026 Eq. 6-7).** The
noise schedule is

    sigma_i = (sigma_max^(1/rho) + i/(N-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho   (Eq. 6)

and churn perturbs each step before the deterministic Heun update:

    gamma = min(churn/N, sqrt(2) - 1)
    s_hat = s * (1 + gamma)
    z <- z + sqrt(s_hat^2 - s^2) * S_noise * epsilon                                        (Eq. 7)

Documented explicitly, per PASO 5.2: **``S_churn`` is the cheapest lever for
improving extreme-peak recovery without retraining.** Avila 2026 §6.1 found
that raising churn (to ~7 at 18 steps) monotonically improved peak recovery
on strong storms before saturating, and is conservative (does not
overshoot) on moderate storms. Tune ``S_churn`` before spending on
architecture changes or retraining.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from src.models.regression_stage import RegressionStageConfig, RegressionUNet


@dataclass(frozen=True)
class EDMConfig:
    sigma_data: float = 0.5   # Avila 2026 Eq. 2/Table 5
    p_mean: float = -1.2      # Karras et al., reused by Avila 2026 Table 5
    p_std: float = 1.2


class EDMPrecond(nn.Module):
    """Wraps a raw network as an EDM denoiser (Eq. 3-4)."""

    def __init__(self, raw_network: nn.Module, config: EDMConfig | None = None):
        super().__init__()
        self.raw_network = raw_network
        self.config = config or EDMConfig()

    def _coefficients(self, sigma: torch.Tensor) -> tuple[torch.Tensor, ...]:
        sigma_data = self.config.sigma_data
        c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
        c_out = sigma * sigma_data / torch.sqrt(sigma ** 2 + sigma_data ** 2)
        c_in = 1.0 / torch.sqrt(sigma_data ** 2 + sigma ** 2)
        c_noise = 0.25 * torch.log(sigma)
        return c_skip, c_out, c_in, c_noise

    def forward(
        self,
        noisy_residual: torch.Tensor,
        sigma: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """``noisy_residual``: (B, C, H, W). ``sigma``: (B,) or scalar tensor.
        ``conditioning``: (B, C_cond, H, W), concatenated with the scaled
        noisy input before the raw network (mean field + ERA5 channels)."""
        sigma = sigma.reshape(-1, 1, 1, 1).to(noisy_residual.dtype)
        c_skip, c_out, c_in, c_noise = self._coefficients(sigma)
        network_input = torch.cat([c_in * noisy_residual, conditioning], dim=1)
        raw_output = self.raw_network(network_input, c_noise.reshape(-1))
        return c_skip * noisy_residual + c_out * raw_output


def edm_loss(
    denoiser: EDMPrecond,
    clean_residual: torch.Tensor,
    conditioning: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Avila 2026 Eq. (5): per-example noise level, weighted denoising MSE."""
    config = denoiser.config
    batch = clean_residual.shape[0]
    device, dtype = clean_residual.device, clean_residual.dtype
    log_sigma = config.p_mean + config.p_std * torch.randn(
        batch, device=device, dtype=dtype, generator=generator
    )
    sigma = torch.exp(log_sigma)
    noise = torch.randn(clean_residual.shape, device=device, dtype=dtype, generator=generator)
    sigma_b = sigma.reshape(-1, 1, 1, 1)
    noisy_residual = clean_residual + sigma_b * noise

    weight = (sigma_b ** 2 + config.sigma_data ** 2) / (sigma_b * config.sigma_data) ** 2
    prediction = denoiser(noisy_residual, sigma, conditioning)
    return (weight * (prediction - clean_residual) ** 2).mean()


def karras_sigma_schedule(
    num_steps: int, sigma_min: float, sigma_max: float, rho: float, *, device=None, dtype=None
) -> torch.Tensor:
    """Avila 2026 Eq. (6). Returns ``num_steps`` sigmas, decreasing to 0 at the end."""
    i = torch.arange(num_steps, device=device, dtype=dtype or torch.float64)
    inv_rho = 1.0 / rho
    sigmas = (
        sigma_max ** inv_rho + i / (num_steps - 1) * (sigma_min ** inv_rho - sigma_max ** inv_rho)
    ) ** rho
    return torch.cat([sigmas, torch.zeros(1, device=device, dtype=sigmas.dtype)])


@dataclass(frozen=True)
class HeunSamplerConfig:
    num_steps: int = 18
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    s_churn: float = 0.0        # 0 = deterministic Heun; Avila 2026 uses 7
    s_noise: float = 1.007
    s_tmin: float = 0.0
    s_tmax: float = float("inf")


@torch.no_grad()
def heun_sample(
    denoiser: EDMPrecond,
    conditioning: torch.Tensor,
    shape: tuple[int, ...],
    config: HeunSamplerConfig | None = None,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """2nd-order Heun sampler with stochastic churn (Avila 2026 Eq. 6-7).

    ``shape`` is the residual tensor's shape, e.g. ``(B, C, H, W)``; returns
    a standardized-residual sample of that shape.
    """
    config = config or HeunSamplerConfig()
    device, dtype = conditioning.device, conditioning.dtype
    sigmas = karras_sigma_schedule(
        config.num_steps, config.sigma_min, config.sigma_max, config.rho,
        device=device, dtype=torch.float64,
    ).to(dtype)

    x = torch.randn(shape, device=device, dtype=dtype, generator=generator) * sigmas[0]
    num_steps = config.num_steps
    for i in range(num_steps):
        s_cur, s_next = sigmas[i], sigmas[i + 1]
        gamma = (
            min(config.s_churn / num_steps, math.sqrt(2.0) - 1.0)
            if config.s_tmin <= float(s_cur) <= config.s_tmax
            else 0.0
        )
        s_hat = s_cur * (1.0 + gamma)
        if gamma > 0:
            noise = torch.randn(shape, device=device, dtype=dtype, generator=generator)
            x = x + (s_hat ** 2 - s_cur ** 2).clamp_min(0.0).sqrt() * config.s_noise * noise

        sigma_hat_batch = torch.full((shape[0],), float(s_hat), device=device, dtype=dtype)
        denoised = denoiser(x, sigma_hat_batch, conditioning)
        d_cur = (x - denoised) / s_hat.clamp_min(1e-12)
        x_next = x + (s_next - s_hat) * d_cur

        if s_next > 0:
            sigma_next_batch = torch.full((shape[0],), float(s_next), device=device, dtype=dtype)
            denoised_next = denoiser(x_next, sigma_next_batch, conditioning)
            d_next = (x_next - denoised_next) / s_next.clamp_min(1e-12)
            x_next = x + (s_next - s_hat) * 0.5 * (d_cur + d_next)

        x = x_next
    return x


class SimpleNoiseConditionedUNet(nn.Module):
    """Minimal raw network ``F`` for the EDM denoiser: a ``RegressionUNet``
    with the noise-level embedding added as extra broadcast input channels
    (a plain, testable stand-in for Avila 2026's FiLM-based noise embedding
    -- swap in a FiLM/AdaGN block for production use without changing the
    ``EDMPrecond`` contract).
    """

    def __init__(self, config: RegressionStageConfig):
        super().__init__()
        self.unet = RegressionUNet(
            RegressionStageConfig(
                in_channels=config.in_channels + 1,  # +1 for the noise embedding channel
                out_channels=config.out_channels,
                base_width=config.base_width,
                encoder_depths=config.encoder_depths,
                attention_at_coarsest=config.attention_at_coarsest,
                add_coordinates=config.add_coordinates,
                nonnegative_output=False,  # residual can be signed
            )
        )

    def forward(self, x: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        noise_channel = c_noise.reshape(b, 1, 1, 1).expand(b, 1, h, w)
        return self.unet(torch.cat([x, noise_channel], dim=1))
