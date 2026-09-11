r"""Shapley-value attribution over model input channels (PASO 6 of
docs/decisions/0002-implementacion-v2.md), following Avila 2026 §6.2:
"we compute Shapley values ... over the twelve conditioning channels. A
Shapley value distributes the model's output among its inputs by asking,
for every possible subset of channels, how much the output changes when
one further channel is added, and then averaging that contribution over all
subsets ... it is the unique attribution that is fair in the sense of
dividing the whole output among the inputs while giving equal credit to
channels that play interchangeable roles. Because there are only twelve
channels here, all 2^12 = 4096 subsets can be enumerated exactly, so the
values are not sampling approximations. An uninformative baseline replaces
channels outside a given subset."

This validates empirically which input channels a trained model actually
relies on for the observable of interest ("precipitacion maxima predicha",
mirroring Avila's "predicted peak wind"), instead of assuming importance
from physical argument alone.

**Exact vs. sampling.** The classic Shapley value for player (channel) ``i``
is

    phi_i = sum_{S subset of N\{i}} [|S|! (n-|S|-1)!] / n! * (v(S u {i}) - v(S))

For this project's channel counts (a handful of ERA5/terrain channels,
comparable to or smaller than Avila's twelve), :func:`exact_shapley_values`
enumerates every subset exactly, matching the paper's methodology. If the
number of channels makes ``2**n`` enumeration impractical,
:func:`sampling_shapley_values` falls back to the standard unbiased
permutation-sampling Monte Carlo estimator (Castro et al. 2009) -- the same
role ``shap.KernelExplainer``/``DeepExplainer`` play in the paper, without
adding a dependency for a computation this module already implements
directly. :func:`shapley_values` dispatches between the two automatically.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial
import random
from typing import Callable, Iterable, Sequence

ValueFunction = Callable[[frozenset[int]], float]

DEFAULT_MAX_EXACT_CHANNELS = 14


def _cached_value_fn(value_fn: ValueFunction) -> ValueFunction:
    cache: dict[frozenset[int], float] = {}

    def wrapped(subset: frozenset[int]) -> float:
        if subset not in cache:
            cache[subset] = value_fn(subset)
        return cache[subset]

    return wrapped


def exact_shapley_values(value_fn: ValueFunction, n_channels: int) -> list[float]:
    """Enumerate all ``2**n_channels`` subsets exactly (Avila 2026 §6.2)."""
    n = n_channels
    v = _cached_value_fn(value_fn)
    phi = [0.0] * n
    all_indices = list(range(n))
    for i in all_indices:
        others = [j for j in all_indices if j != i]
        for s in range(len(others) + 1):
            weight = factorial(s) * factorial(n - s - 1) / factorial(n)
            for subset in combinations(others, s):
                subset_frozen = frozenset(subset)
                marginal = v(subset_frozen | {i}) - v(subset_frozen)
                phi[i] += weight * marginal
    return phi


def sampling_shapley_values(
    value_fn: ValueFunction,
    n_channels: int,
    *,
    n_samples: int = 2000,
    rng: random.Random | None = None,
) -> list[float]:
    """Unbiased permutation-sampling Monte Carlo Shapley estimator, used when
    ``2**n_channels`` exact enumeration is impractical."""
    n = n_channels
    v = _cached_value_fn(value_fn)
    rng = rng or random.Random()
    phi = [0.0] * n
    empty_value = v(frozenset())
    indices = list(range(n))
    for _ in range(n_samples):
        order = indices[:]
        rng.shuffle(order)
        prev_subset: frozenset[int] = frozenset()
        prev_value = empty_value
        for idx in order:
            subset = prev_subset | {idx}
            value = v(subset)
            phi[idx] += value - prev_value
            prev_subset, prev_value = subset, value
    return [total / n_samples for total in phi]


def shapley_values(
    value_fn: ValueFunction,
    n_channels: int,
    *,
    max_exact_channels: int = DEFAULT_MAX_EXACT_CHANNELS,
    n_samples: int = 2000,
    rng: random.Random | None = None,
) -> list[float]:
    """Exact enumeration when ``n_channels <= max_exact_channels`` (Avila
    2026's own regime, 12 channels), otherwise permutation sampling."""
    if n_channels <= max_exact_channels:
        return exact_shapley_values(value_fn, n_channels)
    return sampling_shapley_values(value_fn, n_channels, n_samples=n_samples, rng=rng)


def masked_value_function(
    predict_fn: Callable[[Sequence], float],
    sample: Sequence,
    baseline: Sequence,
) -> ValueFunction:
    """Build a Shapley ``value_fn`` from a model prediction callable.

    ``predict_fn`` receives one array-like of channels (e.g. a stacked
    ``(C, H, W)`` tensor) and returns the scalar observable of interest
    (e.g. predicted peak precipitation, mirroring
    ``src.training.eta_learning.spatial_maximum_observable``).
    ``baseline`` replaces every channel *not* in the queried subset --
    "an uninformative baseline replaces channels outside a given subset"
    (Avila 2026 §6.2); a per-channel mean or zero field are typical choices.
    """
    if len(sample) != len(baseline):
        raise ValueError("sample and baseline must have the same channel count")
    n = len(sample)

    def value_fn(subset: frozenset[int]) -> float:
        composed = [sample[i] if i in subset else baseline[i] for i in range(n)]
        return float(predict_fn(composed))

    return value_fn


def channel_attribution(
    predict_fn: Callable[[Sequence], float],
    sample: Sequence,
    baseline: Sequence,
    channel_names: Iterable[str],
    *,
    max_exact_channels: int = DEFAULT_MAX_EXACT_CHANNELS,
    n_samples: int = 2000,
    rng: random.Random | None = None,
) -> dict:
    """Convenience wrapper: returns a dict with per-channel Shapley values,
    the base value ``v(empty set)``, and the full-input prediction ``v(all)``
    -- the three quantities a SHAP-waterfall-style plot needs (Avila 2026
    Fig. 4: "the bars sum to the difference between the baseline and the
    final prediction").
    """
    names = list(channel_names)
    n = len(names)
    value_fn = masked_value_function(predict_fn, sample, baseline)
    values = shapley_values(
        value_fn, n, max_exact_channels=max_exact_channels, n_samples=n_samples, rng=rng
    )
    base_value = value_fn(frozenset())
    full_value = value_fn(frozenset(range(n)))
    return {
        "channel_names": names,
        "shapley_values": values,
        "base_value": base_value,
        "prediction": full_value,
        # Additivity check: for an exact computation this should equal
        # `full_value - base_value` up to floating point error.
        "sum_of_shapley_values": sum(values),
    }
