r"""Adversarial objective for the P1-inspired precipitation GAN (Candidate A,
PASO 5.1 of docs/decisions/0002-implementacion-v2.md).

All equations are quoted verbatim from **P1** (Glawion et al. 2025,
*spateGAN-ERA5*), section "Objective function", and implemented on **exactly
the valid target pixels** -- invalid observations never enter a loss (repo
design rule, ``agents.md``; consistent with P1 M107 "grids ... not covered
... marked with NaNs").

Notation: ``X`` = ERA5 coarse context, ``Y`` = high-resolution observation,
``\hat{Y}`` / ``\hat{Y}_i`` = generator sample(s), ``D`` = discriminator,
``G`` = generator.

------------------------------------------------------------------------
P1 Eq. (1)   L_D  = E_{X,Y}[ log D(X, Y) ] - E_{X,\hat{Y}}[ log(1 - D(X, \hat{Y})) ]
P1 Eq. (2)   L_GAN(G) = E_X[ log D(X, G(X)) ]
P1 Eq. (3)   L_L1(G)  = | Y - (1/3) \sum_{i=1}^{3} \hat{Y}_i |      (ensemble L1)
P1 Eq. (4)   L_G  = L_GAN(G) + L_L1(G)                              (no weighting)
------------------------------------------------------------------------

P1 minimises a stepwise adversarial objective (refs 54, 55). Eq. (1)/(2) are
written as expectations of ``log D``; the *minimisation* forms below are the
standard binary-cross-entropy equivalents:

* minimising ``L_D`` w.r.t. D  == BCE(D(X,Y), 1) + BCE(D(X,\hat{Y}), 0)
* the generator *maximises* ``L_GAN(G)``; the non-saturating minimisation form
  is ``BCE(D(X, G(X)), 1)``.

P1 Eq. (3) explicitly averages **3** members ("comparing high-resolution
targets to the ensemble mean prediction of 3 members") -- matrix M152 flags
this as "easy to confuse with a 1-member L1". We keep N configurable with a
floor of 3.

The eta-learning W1 regularizer (``src/training/eta_learning.py``) is added
to ``generator_loss``'s total by the training loop, not inside this module
-- it is model-agnostic (PASO 1.2) and applies identically to Candidate B
and the deterministic baseline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
) -> torch.Tensor:
    """P1 Eq. (1): binary-cross-entropy discriminator loss (matrix M146, M150).

    ``real_logits`` = D(X, Y), ``fake_logits`` = D(X, \\hat{Y}) with \\hat{Y}
    detached from the generator graph. Minimising this equals minimising
    ``-E[log D(X,Y)] - E[log(1 - D(X, \\hat{Y}))]``.
    """
    real = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    return real + fake


def generator_adversarial_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """P1 Eq. (2): adversarial term ``L_GAN(G)`` (non-saturating minimisation form).

    The generator maximises ``E_X[log D(X, G(X))]``; equivalently it minimises
    ``BCE(D(X, G(X)), 1)``.
    """
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


def ensemble_l1_loss(
    members: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """P1 Eq. (3): ``L_L1(G) = | Y - mean_i \\hat{Y}_i |`` over N >= 3 members.

    ``members``: (N, B, 1, H, W) with N >= 3 (P1 uses exactly 3). ``target``:
    (B, 1, H, W). The L1 is taken between the observation and the **ensemble
    mean**, then averaged over valid pixels only.
    """
    if members.dim() != target.dim() + 1:
        raise ValueError("members must be (N, *target.shape)")
    n = members.shape[0]
    if n < 3:
        raise ValueError(f"P1 Eq. (3) averages >= 3 members; got N={n}")
    ensemble_mean = members.mean(dim=0)
    return _masked_mean(torch.abs(target - ensemble_mean), mask)


def generator_loss(
    fake_logits: torch.Tensor,
    members: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    l1_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """P1 Eq. (4): ``L_G = L_GAN(G) + L_L1(G)``.

    P1 uses no explicit weighting between the two terms; ``l1_weight`` defaults
    to 1.0 to match the paper and is exposed only for ablation.
    """
    adversarial = generator_adversarial_loss(fake_logits)
    l1 = ensemble_l1_loss(members, target, mask)
    total = adversarial + l1_weight * l1
    return total, {
        "generator_total": total.detach(),
        "generator_adversarial": adversarial.detach(),
        "generator_ensemble_l1": l1.detach(),
    }
