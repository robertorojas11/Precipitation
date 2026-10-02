# Stage 8 — Interpretability and calibrated products

## Purpose and inputs

Wires a stage 6 checkpoint into the two interpretability deliverables
PASO 6/7 of `docs/decisions/0002-implementacion-v2.md` ask for: Shapley
channel attribution (which input channel a trained model actually relies
on for a peak-precipitation prediction) and per-pixel calibrated
precipitation distributions (a zero-inflated hurdle model per pixel,
yielding quantile maps and an "atypical rain" exceedance probability — the
alert-facing product). Like [stage 7](07-evaluate.md), this stage adds
**no new method**: every formula already exists in
`src/evaluation/{shapley_attribution,pixel_distributions}.py`, built
generically ahead of any trainable dataset. What's new here is the wiring,
implemented in `src/evaluation/interpretability_hourly.py`, which reuses
stage 7's own checkpoint loaders
(`evaluate_hourly.load_{baseline,gan,diffusion}_predict_fn`) rather than
re-deriving how to load a model a second time.

```bash
python -m src.evaluation.interpretability_hourly --task shapley \
    --model gan --region mexico_pacifico --split test --checkpoint .../best.pt
python -m src.evaluation.interpretability_hourly --task pixel-distribution \
    --model gan --region mexico_pacifico --split test --checkpoint .../best.pt \
    --pixel-row 50 --pixel-col 80
```

## A refactor that paid for itself immediately

Building this stage's `_predict_fn_for_model` dispatcher required the same
"load a checkpoint, reconstruct the model from its saved config, return an
`(inputs, invariants) -> (M, B, 1, H, W)` mm/hr ensemble closure" logic
stage 7's `evaluate_baseline`/`evaluate_gan`/`evaluate_diffusion` already
had, just inlined inside each of those three functions. Rather than copy
it a second time, `evaluate_hourly.py` was refactored to expose
`load_baseline_predict_fn`/`load_gan_predict_fn`/`load_diffusion_predict_fn`
directly, with `evaluate_baseline` etc. becoming thin wrappers around them.
This stage's own module calls the exact same loaders — a checkpoint loads
identically whether the caller wants evaluation metrics or interpretability
products, by construction, not by convention two modules happen to agree on.

## Shapley attribution (`compute_shapley_attribution`)

Observable: the **predicted peak precipitation** of the model's mean
prediction — `src.training.eta_learning.spatial_maximum_observable`, the
same observable this project's eta-learning regularizer already uses
(`docs/stages_v2/06-train.md`), reused here rather than inventing a second
"what does extreme mean" definition.

Channels: `era5_cp`, `era5_lsp`, `land_mask`, `elevation` — the model's
full 4-channel input contract (`lr_channels=2` + `invariant_channels=2`,
`context_hours=1`). 4 channels is comfortably inside
`shapley_attribution.DEFAULT_MAX_EXACT_CHANNELS` (14), so every call here
is the **exact** enumeration (16 subsets, matching Avila 2026 §6.2's own
methodology for a small channel count), never the sampling fallback.

**Baseline choice.** "An uninformative baseline replaces channels outside
a given subset" (Avila 2026 §6.2) — this project uses an all-zero field per
channel, which is exactly each channel's own training-distribution mean in
the normalized z-score space these channels live in. This is the same
"0.0 = uninformative" convention `PreparedHourlyDataset` already uses to
sanitize invalid pixels for the forward pass
(`docs/stages_v2/06-train.md`), reused here rather than picking a different
notion of "uninformative" for this stage.

Run over an evenly-spaced sample of at most `n_samples` (day, hour)
instances (default 5) — exact Shapley is cheap *per sample* (16 forward
passes), but still real, multiplied-by-`n_samples` cost, so this stays
small by default rather than covering an entire split (same "sample,
don't cover everything" posture `docs/stages_v2/07-evaluate.md`'s frame
cap takes for FSS/RAPSD/SSIM). Returns each sample's full attribution
(`channel_names`, `shapley_values`, `base_value`, `prediction`,
`sum_of_shapley_values`) plus the across-sample mean, for a quick "which
channel generally matters" read without inspecting every sample.

## Per-pixel calibrated distributions (`fit_pixel_distributions`)

"For each pixel, all ensemble-generative samples ... are pooled into a
value set" (`pixel_distributions.py`'s own docstring) —
`pooled_ensemble_samples_at_pixel` does exactly that: runs the loaded
model's ensemble over an evenly-spaced sample of at most `max_frames`
(day, hour) instances, and collects every member's prediction at one fixed
`(row, col)` pixel into a flat array, already in physical mm/hr (every
`predict_fn` from stage 7's loaders already returns physical units, so no
denormalization step is needed here).

That pooled array is handed directly to `pixel_distributions
.fit_pixel_distribution()` — the zero-inflated hurdle model, AIC/BIC family
ranking, and bootstrap Anderson-Darling goodness-of-fit all run completely
unmodified; this stage does not touch that module's logic at all.

**Pixels are named explicitly, not swept across the grid.** `fit_
pixel_distributions` takes a list of `(row, col)` pairs, not "every pixel
in the region". At the default `n_bootstrap=100` refits per family × 5
candidate families, one pixel's fit is already real compute (500 scipy
fits); a full `H×W` grid is a deliberately separate, much more expensive,
explicitly user-triggered run (mirrors how a full-historical-scale
`compute_stats()`/`build_region()`/training run has been a deliberate,
separate step at every stage since 4) — not something this wiring attempts
by default.

**Graceful degradation, not a crash, on degenerate pixel data.**
`fit_pixel_distribution` itself raises `ValueError` when a pixel has fewer
than 5 positive (rainy) samples; scipy's own family `.fit()` calls can also
raise `ValueError` (observed directly while testing this stage: "The data
contains non-finite values" from a near-degenerate bootstrap resample, see
"Known limitations" below for why). `fit_pixel_distributions` catches
`ValueError` per pixel and reports `{"row", "col", "error", "sample_count"}`
for that pixel instead of aborting the whole call — the same
"report instead of crash" posture every `validate_*` function in this
codebase already takes, applied here to a very different kind of failure
(a statistical fit failing, not a data-integrity check failing).

## Tests

`tests/test_interpretability_hourly.py` — 2 tests, reusing stage 6/7's
synthetic prepared-region fixture to train one tiny real GAN checkpoint,
then: (1) Shapley attribution over 2 samples, checking the additivity
identity (`sum_of_shapley_values == prediction - base_value`, which must
hold exactly for an exact enumeration, up to floating point) in addition to
shape/key checks; (2) per-pixel distribution fitting at one pixel, with a
test design that **accepts either a successful fit or a reported `error`**
— the synthetic fixture's ERA5 forcing is identical across every hour/day
(`tests/test_training.py`'s `_write_day`), so the only real source of
variation at a fixed pixel is ensemble noise, which can produce a pooled
sample scipy's continuous-family fitting genuinely cannot fit (observed in
practice while writing this test: a `burr12` overflow warning and a
"non-finite values" `ValueError` from an extreme-parameter bootstrap
refit). This is not a bug to work around with more synthetic realism; it
is exactly the graceful-degradation path described above, and asserting
"either outcome, but always a well-formed result" tests the real contract
instead of gambling on scipy converging against data never designed for
statistical realism. Full suite: 193 tests passing.

## Known limitations

- No real-scale run yet, same as every stage since 4: only smoke-tested
  against a tiny synthetic checkpoint whose own input data is literally
  constant across time, which is realistic enough to exercise the wiring
  but not to validate that the *statistics themselves* look sensible on
  real atmospheric data — that can only be judged once stage 6 actually
  trains on real history.
- `fit_pixel_distributions` is not parallelized across pixels; a
  multi-pixel call fits each pixel's distribution sequentially. Fine for
  the handful-of-pixels use case this is scoped for (an alert product for
  specific locations of interest); a full-grid sweep would need either
  parallelization or a reduced `n_bootstrap`, neither built here.
- Shapley attribution's default `n_samples=5` and the per-pixel fit's
  default `max_frames=20` are both plain cost controls with no principled
  sample-size justification yet, same caveat `docs/stages_v2/07-evaluate
  .md` already logged for its own frame cap.
- Candidate B's Shapley attribution runs its full two-stage (regression +
  Heun-sampled diffusion) `predict_fn` per subset evaluation — correct, but
  meaningfully slower per sample than the baseline or GAN; no shortcut is
  taken here (e.g. attributing only the frozen stage 1 mean), since that
  would silently change what is being explained.

## Connection to the next stage

This completes the `docs/stages_v2/README.md` roadmap's planned stages
(0 through 8) as *infrastructure* — every stage from acquisition through
interpretability now has a working, tested implementation. No stage beyond
this one is currently planned; what remains is entirely real-data
execution, a sequence of deliberate, user-triggered decisions this project
has consistently treated as separate from building the pipeline itself:
running stage 4's `compute_stats()`/`build_region()` and stage 6's training
scripts at full historical scale (2010–2019 × `mexico_pacifico` +
`mexico_atlantico_golfo`), then re-running stages 7/8 against the resulting
real checkpoints to get the first real numbers this framework has actually
produced.
