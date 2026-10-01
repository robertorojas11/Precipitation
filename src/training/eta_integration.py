"""Wires ``src/training/eta_learning.py``'s Extreme-Event-Aware (eta-)
Learning regularizer into an actual training loop for the first time
(``docs/architecture_v3.md`` previously tracked this as "schedule tracker;
not yet wired into an actual training loop -- no trainable dataset exists
yet"; stage 6 is what finally provides that dataset). Model-agnostic by
construction (PASO 1.2 of docs/decisions/0002-implementacion-v2.md): the
same :class:`EtaRegularizer` is reused by ``train_baseline.py``,
``train_gan.py`` (on the generator), and ``train_diffusion.py`` (on stage
1's regression mean only -- see docs/stages_v2/06-train.md's "Known
limitations" for why stage 2 is not wired).

Two auxiliary pools, built once per run from the training split itself:

- ``reference_pool`` (the paper's ``nu0``): spatial maxima of **real**
  target fields, i.e. the empirical "how extreme does this region's
  observed rain actually get" distribution
  (docs/ETA_LEARNING_NOTES.md's "empirical nu0" case for a data-rich
  training region; the hypothesized-GEVD case for a data-scarce zero-shot
  region is designed there but not wired here).
- ``auxiliary_inputs``: real ``(inputs, invariants)`` pairs Algorithm 1/2
  (IICT) periodically runs inference over, under ``torch.no_grad()``, to
  find which inputs *currently* induce the model's most extreme outputs --
  the gradient-carrying eta forward pass is then restricted to just the
  ``n_q`` selected examples (``update_tail_index``), the memory-saving
  restriction the paper's Algorithm 2 is for.

Using the sanitized (NaN -> 0) target field for the spatial-maximum
observable is deliberately **not** an approximation: precipitation is
non-negative, so a zero-filled invalid pixel can never exceed a genuine
valid-pixel value and can never become the max unless every valid pixel in
that field is itself exactly zero (in which case the max is correctly 0
either way). No masking is needed before taking the max.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from src.training.eta_learning import (
    EtaLearningSchedule,
    eta_tail_loss,
    spatial_maximum_observable,
    tail_quantile_levels,
    update_tail_index,
)

ForwardFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _per_sample_observable(field: torch.Tensor) -> torch.Tensor:
    """``spatial_maximum_observable`` only reduces the trailing two (H, W)
    dims by design (its docstring: "returns a tensor with those two
    dimensions reduced away"), so a (B, 1, H, W) precipitation field comes
    back as (B, 1), not (B,). ``update_tail_index`` needs a genuinely 1-D
    array to ``argsort`` over a proper batch ranking -- argsort on a (B, 1)
    tensor sorts each length-1 row independently (a no-op) instead of
    ranking across the batch, silently producing a degenerate index. Every
    model here has exactly one output channel, so this is always safe.
    """
    return spatial_maximum_observable(field).reshape(field.shape[0])


@dataclass(frozen=True)
class EtaConfig:
    lambda_: float = 0.0  # 0 disables eta-learning entirely (ERM-only training)
    tau: float = 0.95
    n_q: int = 100
    pretrain_steps: int = 0
    refresh_every: int = 200
    reference_pool_size: int = 512
    auxiliary_pool_size: int = 512

    @property
    def enabled(self) -> bool:
        return self.lambda_ > 0.0


class EtaRegularizer:
    def __init__(self, config: EtaConfig):
        self.config = config
        self.schedule = EtaLearningSchedule(
            erm_pretrain_steps=config.pretrain_steps, refresh_every=config.refresh_every,
        )
        self.reference_pool: torch.Tensor | None = None
        self.auxiliary_inputs: dict[str, torch.Tensor] | None = None
        self._tail_index: torch.Tensor | None = None

    def set_reference_pool(self, target_spatial_maxima: torch.Tensor) -> None:
        self.reference_pool = target_spatial_maxima

    def set_auxiliary_inputs(self, inputs: torch.Tensor, invariants: torch.Tensor) -> None:
        self.auxiliary_inputs = {"inputs": inputs, "invariants": invariants}

    def maybe_refresh_tail_index(self, forward_fn: ForwardFn) -> None:
        """Algorithm 1's inference step + Algorithm 2's index selection,
        run only every ``refresh_every`` steps (and only once eta-learning
        is active, i.e. ERM pre-training has finished)."""
        if not self.config.enabled or not self.schedule.should_refresh_tail_index():
            return
        if self.auxiliary_inputs is None:
            return
        with torch.no_grad():
            prediction = forward_fn(self.auxiliary_inputs["inputs"], self.auxiliary_inputs["invariants"])
            observable = _per_sample_observable(prediction)
        q = tail_quantile_levels(
            self.config.n_q, self.config.tau, device=observable.device, dtype=observable.dtype,
        )
        self._tail_index = update_tail_index(observable, q)
        self.schedule.mark_refreshed()

    def term(self, forward_fn: ForwardFn) -> torch.Tensor | None:
        """The gradient-carrying W1 tail term (Eq. 15), or ``None`` when
        eta-learning is disabled, still in its ERM pre-training window, or
        no tail index has been computed yet."""
        if not self.config.enabled or not self.schedule.is_eta_active():
            return None
        if self._tail_index is None or self.auxiliary_inputs is None or self.reference_pool is None:
            return None
        tail_inputs = self.auxiliary_inputs["inputs"][self._tail_index]
        tail_invariants = self.auxiliary_inputs["invariants"][self._tail_index]
        prediction = forward_fn(tail_inputs, tail_invariants)
        model_observable = _per_sample_observable(prediction)
        return eta_tail_loss(model_observable, self.reference_pool, tau=self.config.tau, n_q=self.config.n_q)

    def step(self) -> None:
        self.schedule.advance()


def build_eta_pools(
    dataset, config: EtaConfig, *, device, denormalize_target_fn, dataloader_generator=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Loads a one-off random subset of ``dataset`` to build both auxiliary
    pools (see module docstring): the ``nu0`` reference pool of real target
    spatial maxima, and the pool of real ``(inputs, invariants)`` pairs used
    for periodic tail-index refresh. ``denormalize_target_fn`` converts a
    normalized target batch into the physical-unit space the observable
    should be computed in (mm/hr for every candidate here) -- passed in
    rather than imported, so this helper stays agnostic to which candidate
    calls it.
    """
    from torch.utils.data import DataLoader

    pool_size = min(max(config.reference_pool_size, config.auxiliary_pool_size), len(dataset))
    loader = DataLoader(dataset, batch_size=min(32, pool_size), shuffle=True, generator=dataloader_generator)
    inputs_chunks, invariants_chunks, target_chunks = [], [], []
    collected = 0
    for batch in loader:
        inputs_chunks.append(batch["inputs"])
        invariants_chunks.append(batch["invariants"])
        target_chunks.append(denormalize_target_fn(batch["target"]))
        collected += batch["inputs"].shape[0]
        if collected >= pool_size:
            break
    inputs = torch.cat(inputs_chunks, dim=0)[:pool_size].to(device)
    invariants = torch.cat(invariants_chunks, dim=0)[:pool_size].to(device)
    target_physical = torch.cat(target_chunks, dim=0)[:pool_size].to(device)
    reference_pool = _per_sample_observable(target_physical)
    return reference_pool, inputs[: config.auxiliary_pool_size], invariants[: config.auxiliary_pool_size]
