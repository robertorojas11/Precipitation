# η-learning notes: what `ν0` is, where it comes from, and how it can mislead

This document exists so that nobody in the future treats the output of the
η-learning regularizer as verified ground truth without this caveat. Read
this before using `src/training/eta_learning.py` on a real training run, and
before interpreting any extreme value it produces.

Source paper: Chang & Sapsis 2026, *Extreme Event Aware (η-) Learning*,
Nature Communications (`docs/audit/papers/chang_sapsis2026_eta-learning.pdf`).
Implementation: `src/training/eta_learning.py`. Methodology decision:
`docs/decisions/0002-implementacion-v2.md` §1.2-1.3.

## What `ν0` is

`ν0` is a probability measure on the space of the chosen **observable**
(here, the spatial maximum of the generated precipitation field,
`src/training/eta_learning.py::spatial_maximum_observable`). It is *not* a
ground-truth precipitation field, and it is *not* a distribution over
pixel-level values -- it is a distribution over a single scalar summary of
each generated field. The η-learning training objective adds a term that
pulls the model's induced distribution of that scalar toward `ν0`, using a
Wasserstein-1 distance restricted to the tail (quantiles above `τ`, see the
module docstring for the exact equations).

`ν0` is the "prior knowledge" the paper's whole framework is built around:
"we leverage a priori available statistical information for an observable
indicative of extremeness ... often available through asymptotic analysis
... or a dataset of unlabeled data." Without a `ν0`, there is nothing for
the η term to regularize toward, and the framework reduces to plain ERM
(supervised learning alone) — which the paper's Theorem 4 shows is
provably inadequate in the extreme-event region when training data is
scarce there.

## Where `ν0` comes from, per region (docs/decisions/0002 §1.3)

| Region | `ν0` source | Status |
|---|---|---|
| `mexico_pacifico` | Empirical: `ν0 = y_#μ`, estimated directly from the historical IMERG/Oya record for that region, once acquired. | Planned — no acquired data yet (`docs/DATA_SOURCES.md`). |
| `mexico_atlantico_golfo` | Empirical, same as above. | Planned — no acquired data yet. |
| `puerto_rico_caribe` | **Hypothesized generalized extreme value distribution (GEVD)**, parameterized from published regional hurricane/tropical-storm climatology literature, *not* from this project's own historical IMERG/Oya record. | Planned — the region itself is new (`docs/DATA_SOURCES.md`); no cascade has been built for it yet, so there is no empirical basis to fall back on even once acquisition starts. |

**Operational threshold for "sufficient history" to go empirical** (a
project decision, not a value from the paper, `docs/decisions/0002
-implementacion-v2.md` OQ-4): provisionally, **≥10 years of valid hourly
IMERG/Oya coverage** for a region. Revisit this once real coverage
statistics are known.

## Why the hypothesized case is a hypothesis, not a fact

The paper's own worked example ("η-Downscaling with hypothesized
heavier-tailed distributions") is explicit that this substitution is a
**modeling choice under an assumed adverse-tail hypothesis**, not a
verified fact: "This experiment does not validate the hypothesized `ν0` or
the amplified precipitation values produced under that hypothesis... The
fields illustrate how the learned downscaling map responds when an assumed
heavier tail is imposed, rather than providing evidence that the amplified
extremes will occur or that their magnitudes are physically validated."

**Quantified cost of a badly specified `ν0`** (paper's own sensitivity
analysis, SI Table 2 / Fig. 6, cited in
`docs/decisions/0002-implementacion-v2.md` §1.3): perturbing the reference
tail quantile by a bias parameter `α` from −1 to 2 (moving the assumed tail
in the heavier direction) increased full-grid RMSE from 2.926 to 3.527 — a
**+20.55% relative increase** — while SSIM stayed almost unchanged
(0.941→0.947). In other words: a badly specified `ν0` mainly shows up as a
**pointwise-fidelity cost** (RMSE), not necessarily as a visible change in
spatial structure (SSIM) — a model can look structurally reasonable while
being quantitatively wrong at the tail because of a bad `ν0`.

## The concrete warning for Puerto Rico/Caribe

Because this region's `ν0` is a hypothesized GEVD built from *external*
climatology literature rather than this project's own data:

1. **Never present an extreme value generated under this `ν0` as an
   observed or verified event.** It is, at best, "a statistically
   consistent, physically plausible" scenario conditioned on an assumed
   tail law (paper's own language) — not a forecast, not a nowcast, and not
   equivalent to an IMERG/Oya-observed extreme.
2. **The specific GEVD parameters used must be documented explicitly**
   wherever they are chosen (a future commit updating this file), including
   the literature source and why that source was judged applicable to this
   region's tropical-cyclone climatology. Do not silently pick parameters
   without recording the source.
3. **Any downstream product derived from this region's η-trained model**
   (quantile maps, exceedance probabilities in
   `src/evaluation/pixel_distributions.py`) must carry a visible caveat that
   its tail behavior is conditioned on a hypothesized reference law, not on
   observed regional history.
4. **Re-evaluate the choice of `ν0` for this region as soon as enough
   IMERG/Oya history accumulates** to check the hypothesized GEVD against
   real (if still limited) data, rather than leaving the hypothesis
   unexamined indefinitely.

## Training procedure reminders (why the code is structured this way)

- **ERM pre-training is mandatory, not optional.** Activating the η term
  before the model has learned the basic input-output relationship causes
  gradient conflict: the model's own (unreliable, untrained) output
  distribution is used to pick which inputs count as "tail" samples for the
  W1 loss, and an untrained model's ranking of its own outputs is not a
  reliable proxy for which physical inputs are actually extreme.
  `EtaLearningSchedule.erm_pretrain_steps` enforces this as a step count.
- **The tail-index refresh (IICT) must recur periodically, not once.** The
  set of "which inputs look extreme under the current model" changes as
  training proceeds; `EtaLearningSchedule.refresh_every` (paper's `ω`) and
  `update_tail_index()` implement the periodic refresh.
- **`λ` only rescales gradient magnitude, not direction.** If training is
  unstable, first check ERM pre-training length and refresh cadence before
  reaching for `λ` — the paper is explicit that magnitude balancing alone
  cannot fix a directional conflict caused by unreliable tail-set selection.
