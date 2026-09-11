r"""Extreme Event Aware (eta-) Learning regularizer (Chang & Sapsis 2026),
PASO 5.3 of docs/decisions/0002-implementacion-v2.md.

Model-agnostic: attaches to any of Candidate A (GAN), Candidate B
(regression + diffusion), or the deterministic baseline, by adding a term to
the existing loss (PASO 1.2). The observable used throughout this project is
the **spatial maximum of the generated precipitation field**
(:func:`spatial_maximum_observable`), matching "the observable for the
coarser simulations ... the spatial maximum of a precipitation field, a
natural candidate for studying extremes" (paper, "A real-world precipitation
downscaling challenge").

Core optimization objective (paper Eq. 11, approximating ``y`` directly --
our case, since the observable is computed on the model's *output*, not on
an intermediate state ``u``):

    y_eta := argmin_phi  (1/n) sum_i l(phi, x_i, y_i)  +  lambda * W1(phi#mu, nu0)

**W1 via its 1-D quantile representation (Eq. 13):**

    W1(nu1, nu2) = integral_0^1 | F^-1_nu1(q) - F^-1_nu2(q) | dq

approximated by Monte Carlo over a quantile-level set Q (Eq. 14):

    W1(...) ~= (1/n_q) * sum_i | F^-1_{model}(q_i) - F^-1_{nu0}(q_i) |

**Tail-biased quantile sampling.** Instead of sampling Q uniformly on
[0, 1], sample it uniformly on [tau, 1] (tau typically 0.95-0.975) so
"the W1 term concentrates its signal on the tail" (paper, "Estimating W1 to
emphasize tails"). This is the term actually used in the training objective
(Eq. 15):

    min_phi  (1/n) sum_i l(phi,x_i,y_i)  +  (lambda/n_q) sum_i |F^-1_{model}(q_i) - F^-1_{nu0}(q_i)|,   {q_i} ~ [tau, 1]

**ERM pre-training (critical stabilization, "Gradient conflict and ERM
pre-training").** Before the eta term is ever computed against the model's
own output distribution, the model is trained with the supervised loss
*alone* -- an untrained model's output quantiles are not a reliable index of
which *inputs* are "tail-like", so activating the W1 term too early causes
gradient conflict. :class:`EtaLearningSchedule` tracks this as a step count.

**Inference-Informed Continual Training (IICT), Algorithm 1/2.** The set of
auxiliary-input indices used to evaluate the tail loss is refreshed every
``omega`` iterations by running inference on the full auxiliary set,
sorting the induced observable values, and picking the input whose
observable realizes each target quantile (nearest-rank). See
:func:`update_tail_index`.

**Choice of lambda ("Choice of lambda").** ``lambda`` is a plain
hyperparameter by default, but may be auto-balanced by gradient norm so the
scaled W1 gradient has a comparable magnitude to the ERM gradient early in
training:

    lambda = || grad_theta L_ERM ||_2 / ( || grad_theta L_W1 ||_2 + eps )

See :func:`balance_lambda`. This only rescales the W1 gradient's magnitude;
ERM pre-training and periodic tail-index refresh remain the primary
stabilization mechanisms (paper: "lambda only rescales the W1 gradient; it
does not change its direction").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


def spatial_maximum_observable(field: torch.Tensor) -> torch.Tensor:
    """``g(u) = max(u)`` over spatial dims, per sample.

    ``field``: (..., H, W) or (..., C, H, W) -- the trailing two dims are
    treated as the spatial domain. Returns a tensor with those two
    dimensions reduced away.
    """
    return field.flatten(start_dim=-2).max(dim=-1).values


def tail_quantile_levels(
    n_q: int, tau: float = 0.95, *, generator: torch.Generator | None = None,
    device=None, dtype=None,
) -> torch.Tensor:
    """``n_q`` quantile levels sampled uniformly on ``[tau, 1]`` (tail-biased).

    ``tau`` is "typically chosen around 0.95 or 0.975" (paper, "Estimating
    W1 to emphasize tails").
    """
    if not 0.0 <= tau < 1.0:
        raise ValueError(f"tau must be in [0, 1), got {tau}")
    u = torch.rand(n_q, generator=generator, device=device, dtype=dtype)
    return tau + (1.0 - tau) * u


def _empirical_quantiles(samples: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Inverse-CDF (quantile function) of an empirical 1-D sample set at
    levels ``q``, via linear interpolation (``torch.quantile``)."""
    samples = samples.reshape(-1).to(torch.float64)
    return torch.quantile(samples, q.to(torch.float64)).to(samples.dtype)


def wasserstein1_1d(
    samples_a: torch.Tensor,
    samples_b: torch.Tensor,
    quantile_levels: torch.Tensor | None = None,
    *,
    n_q: int = 200,
) -> torch.Tensor:
    """General 1-D W1 distance via the quantile representation (Eq. 13),
    approximated by Monte Carlo quadrature (Eq. 14) over ``quantile_levels``
    (uniform on ``[0, 1]`` with ``n_q`` points if not given). Use
    :func:`eta_tail_loss` instead when the tail-biased sampling of PASO 5.3
    is wanted.
    """
    if quantile_levels is None:
        quantile_levels = torch.linspace(0.0, 1.0, n_q + 2, device=samples_a.device)[1:-1]
    qa = _empirical_quantiles(samples_a, quantile_levels)
    qb = _empirical_quantiles(samples_b, quantile_levels)
    return torch.abs(qa - qb).mean()


def eta_tail_loss(
    model_observable_samples: torch.Tensor,
    reference_observable_samples: torch.Tensor,
    *,
    tau: float = 0.95,
    n_q: int = 100,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """The tail-focused W1 term of Eq. (15): quantile levels drawn from
    ``[tau, 1]``, quantiles of the model's induced observable distribution
    compared against quantiles of the reference distribution ``nu0``
    (represented here by a sample pool -- either the empirical distribution
    for a data-rich region, or a batch drawn from a hypothesized GEVD for a
    data-scarce region, see docs/ETA_LEARNING_NOTES.md).
    """
    q = tail_quantile_levels(
        n_q, tau, generator=generator,
        device=model_observable_samples.device, dtype=model_observable_samples.dtype,
    )
    model_q = _empirical_quantiles(model_observable_samples, q)
    reference_q = _empirical_quantiles(reference_observable_samples, q)
    return torch.abs(model_q - reference_q).mean()


def balance_lambda(
    erm_grad_norm: float, w1_grad_norm: float, *, eps: float = 1e-8
) -> float:
    """Gradient-norm balancing rule ("Choice of lambda"):

        lambda = || grad_theta L_ERM ||_2 / ( || grad_theta L_W1 ||_2 + eps )

    Only rescales the W1 gradient's magnitude to match the ERM gradient; it
    does not change its direction.
    """
    return float(erm_grad_norm) / (float(w1_grad_norm) + eps)


def update_tail_index(
    observable_values: torch.Tensor, quantile_levels: torch.Tensor
) -> torch.Tensor:
    """Algorithm 2 (Update_Tail_Index): given the observable evaluated on
    every input of the auxiliary set (``observable_values``, shape
    ``(n_x,)``), return the auxiliary-set index whose observable value
    realizes each target quantile in ``quantile_levels`` (nearest-rank).

    This is what lets a training loop restrict the *gradient-carrying*
    forward/backward pass to only ``n_q << n_x`` inputs, while still basing
    the choice of *which* inputs matter on inference over the full pool
    (paper, "Memory challenges and tail-set selection for W1 estimation").
    """
    n = observable_values.numel()
    order = torch.argsort(observable_values)
    ranks = torch.clamp((quantile_levels * n).ceil().long() - 1, 0, n - 1)
    return order[ranks]


@dataclass
class EtaLearningSchedule:
    """Tracks the two procedural stabilizers from "Effective training
    strategies": ERM pre-training (:attr:`erm_pretrain_steps`) and periodic
    tail-index refresh (:attr:`refresh_every`, paper's ``omega``).
    """

    erm_pretrain_steps: int = 0
    refresh_every: int = 1
    step: int = field(default=0, init=False)
    last_refresh_step: int | None = field(default=None, init=False)

    def is_eta_active(self) -> bool:
        return self.step >= self.erm_pretrain_steps

    def should_refresh_tail_index(self) -> bool:
        if not self.is_eta_active():
            return False
        if self.last_refresh_step is None:
            return True
        return (self.step - self.last_refresh_step) >= self.refresh_every

    def mark_refreshed(self) -> None:
        self.last_refresh_step = self.step

    def advance(self) -> None:
        self.step += 1
