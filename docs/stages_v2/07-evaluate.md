# Stage 7 — Evaluation (CRPS, FSS, percentile-MAE, RMSE/SSIM, comparison)

## Purpose and inputs

Wires a stage 6 checkpoint plus a prepared split into the metrics
`docs/decisions/0002-implementacion-v2.md` §8 asks for, and into the
cross-model comparison table. This stage adds **no new metric formulas** —
every one already existed, built generically, ahead of any trainable
dataset (`docs/architecture_v3.md`'s B.5 rows: "DONE ... no real training
run yet to feed it"). What was missing, and what
`src/evaluation/evaluate_hourly.py` adds, is the part that turns a
checkpoint + a prepared split into the actual numbers those functions
expect: model loading, ensemble generation, masking, and pooling across an
entire split.

```bash
python -m src.evaluation.evaluate_hourly --model baseline  --region mexico_pacifico --split test \
    --checkpoint outputs/v2_generative/mexico_pacifico/deterministic_resnet/<run>/best.pt
python -m src.evaluation.evaluate_hourly --model gan       --region mexico_pacifico --split test \
    --checkpoint outputs/v2_generative/mexico_pacifico/gan/<run>/best.pt
python -m src.evaluation.evaluate_hourly --model diffusion --region mexico_pacifico --split test \
    --stage1-checkpoint .../diffusion_stage1_regression/<run>/best.pt \
    --stage2-checkpoint .../diffusion_stage2_residual/<run>/best.pt
python -m src.evaluation.evaluate_hourly --model bilinear  --region puerto_rico_caribe --split test
```

Each call prints a JSON report and, with `--output`, writes it atomically to
disk (`src.evaluation.evaluate_hourly.write_report`). `to_model_run_result()`
flattens a report into the `(model, eta_enabled)` shape
`src.evaluation.model_comparison` aggregates, so several reports can be
combined into one comparison table (`build_comparison_table`,
`eta_learning_contribution`).

## A found-and-fixed gap in stage 6: checkpoints didn't save their own architecture

Writing the first real `evaluate_*` call against a stage 6 checkpoint failed
immediately with a PyTorch `size mismatch` / `Missing key(s)` error:
`train_baseline.py` and `train_gan.py` saved only `model_state` /
`generator_state`+`discriminator_state` in their checkpoints, never the
`base_width`/`depth` (or generator/discriminator config) actually used to
build that network. `train_diffusion.py`'s stage 1 already did this
correctly (`model_config`); its stage 2 had the same gap as the other two.
Reconstructing the architecture from *defaults* instead of from what was
actually trained works only by coincidence (when the defaults happen to
match) and silently loads the wrong-shaped network otherwise — exactly the
failure mode these tests hit using a deliberately tiny (`base_width=8`)
test architecture against a default-`base_width=64` reconstruction.

Fixed in `src/training/{train_baseline,train_gan,train_diffusion}.py`: every
checkpoint now also saves `model_config` / `generator_config`+
`discriminator_config` / `denoiser_config` (an `asdict()` of the exact
config dataclass used to build that run's network), and
`evaluate_hourly.py`'s loaders rebuild each model from that saved config
rather than from a fresh default-constructed one. This is the same
"checkpoint must be self-describing" convention
`train_diffusion.py`'s stage 1 already modeled correctly — the other two
scripts just hadn't needed it yet, since stage 6's own tests never
round-tripped a checkpoint through a *different* process the way an
evaluation run does.

## Ensemble convention

Every model is evaluated as an `(M, B, 1, H, W)` ensemble of **physical**
mm/hr predictions — the one shared contract `run_evaluation()`'s core loop
depends on, regardless of which representation a candidate trains in
internally (Candidate B's own training loop works in normalized log-space;
its `evaluate_diffusion()` closure denormalizes before returning, so the
shared loop never has to know the difference):

- **Baseline** (`evaluate_baseline`): `M=1`, the model's own deterministic
  output. CRPS correctly reduces to MAE for `M=1` (P1 M181/P2 M247,
  verified by `crps_ensemble`'s own tests, not re-verified here).
- **Bilinear baseline** (`evaluate_bilinear_baseline`): also `M=1`, no
  model at all.
- **GAN** (`evaluate_gan`): reuses `ConditionalGenerator.sample()` directly
  (frozen per-member dropout seed, P1 M160) — no new ensemble logic needed,
  the model already provides exactly the right method.
- **Candidate B** (`evaluate_diffusion`): draws `M` independent Heun-sampled
  residuals (each with its own seeded `torch.Generator`) and adds each to
  the one frozen stage 1 mean, since only the residual is stochastic.

## The bilinear baseline is read from stage 2, not recomputed

P2 M225 / P1 M199's lower-bound baseline is "ERA5 total precipitation
bilinearly interpolated to the target grid". Stage 2
(`build_hourly_dataset.py`) already reprojects ERA5 cp/lsp onto the
canonical HR grid with `Resampling.bilinear` — confirmed by reading that
module, not assumed — so the **build-stage** NPZ's `inputs` field, summed
over its two channels, already *is* this baseline. `evaluate_bilinear_
baseline()` reads straight from `dataset_index_hourly_<region>.csv` and the
build-stage NPZs (`_read_accepted_rows`), not from the prepared split, and
calls neither a model nor `src.evaluation.baselines.bilinear_interpolation`
(that function exists for a caller that only has a genuinely low-resolution
source array; this project's pipeline already produces the interpolated
field it would compute, one stage earlier). The *prepared* (stage 4)
tensors cannot substitute for this: M216's IMERG-range clip is lossy, so
a baseline computed from the clipped/normalized tensors would not be the
same "raw ERA5, no correction at all" lower bound the metric is meant to
contextualize everything else against.

## Pooled vs. per-frame metrics

- **Pooled over the whole split** (every valid pixel, concatenated into
  flat arrays): `deterministic_metrics.pooled_scores` (R2/RMSE/MAE/BIAS)
  and `probabilistic_metrics.percentile_mae` — both need the *global*
  distribution (a percentile, or R2's own global variance, cannot be
  correctly averaged from per-batch values). CRPS is the one exception:
  since it is already a plain per-pixel mean, a valid-pixel-weighted
  running average of each batch's CRPS is mathematically identical to
  pooling every pixel first — so CRPS is accumulated as a running sum
  instead of retaining a flat array, avoiding one potentially-large
  allocation.
- **Per-frame, on an evenly-spaced sample of at most `MAX_SPATIAL_FRAMES`
  (20) frames**: FSS (`mean_fss`), RAPSD (`rapsd`), full-field RMSE/SSIM
  (`full_field_rmse_ssim`), and the rank histogram (`rank_histogram`,
  summed across sampled frames — a standard way to pool a rank histogram
  over many verification instances). These are inherently 2D, so there is
  no "pool across the whole split" version of them to begin with; capping
  the frame count is purely a cost control, documented as a known
  limitation below, not a metric-formula change.

RAPSD additionally needs a full, finite rectangular grid for the FFT — no
masked-FFT exists in this codebase or in pysteps' own API used as a
fallback — so invalid pixels are zero-filled (`np.where(mask, field, 0.0)`)
before the transform, a standard practical simplification for spectral
analysis of a field with missing data, not a precision-engineered one; the
two spectra (predicted vs. observed) are compared with a single scalar,
`rapsd_log_distance` (mean absolute log-spectrum difference), rather than
returning full spectra in every JSON report.

`IMERG_PIXEL_KM` (FSS's spatial-scale unit) is a nominal equatorial
`0.1 deg * 111 km/deg` conversion, not latitude-corrected per region — FSS's
own scale buckets (P1 M184, tens to hundreds of km) are coarse enough that
this approximation does not change which bucket a scale falls into for any
of this project's three regions, but it is an approximation, not an exact
per-pixel area.

## Tests

`tests/test_evaluate_hourly.py` — 4 tests, reusing stage 6's own synthetic
prepared-region fixture (`tests.test_training._WithPreparedSyntheticRegion`)
to train tiny *real* checkpoints (1 epoch each, `base_width=8`) for the
baseline, the GAN, and Candidate B's stage 1 + stage 2, then evaluates each
one and asserts the report has every expected key with finite, non-empty
pooled metrics; plus one test for the bilinear baseline (no training
needed). One test also exercises `write_report` + `to_model_run_result` +
`build_comparison_table` together. These tests do not re-verify any metric
formula (each metric module has its own dedicated test file already); they
verify the *wiring* — that a checkpoint loads with the right shapes, that
ensembles have the right axis, that masks reach every metric call. Full
suite: 191 tests passing.

## Known limitations

- Pooled metrics hold the whole split's valid pixels in memory as flat
  arrays (see "Pooled vs. per-frame metrics" above) — fine at this stage's
  test scale, a real cost at full historical scale (not yet run, same gap
  noted in every stage since 4). A streaming/chunked percentile and R2
  computation would be the fix, not built here to avoid optimizing before
  real throughput is measured (same stance taken in `docs/stages_v2/
  06-train.md`'s dataset I/O limitation).
- The 20-frame cap on FSS/RAPSD/SSIM/rank-histogram is a cost control with
  no principled sample-size justification yet; revisit once a real
  full-scale evaluation run's wall-clock cost is actually measured.
- `rainfarm` (P1 M195) remains unimplemented (`src/evaluation/baselines.py`
  raises `NotImplementedError` on purpose) — only the bilinear baseline is
  wired here.
- Shapley attribution and per-pixel calibrated distributions
  (`src/evaluation/{shapley_attribution,pixel_distributions}.py`) are
  **not** part of this stage — `docs/stages_v2/README.md` scopes them to
  stage 8 ("Interpretability and calibrated products"), a separate
  PASO from evaluation metrics.
- No real-scale run yet, consistent with every stage since 4: only
  smoke-tested against tiny synthetic checkpoints. In particular, the
  geographic-generalization comparison (`puerto_rico_caribe`, zero-shot,
  never trained on) has not been run against any real trained model —
  the infrastructure to run it (`--region puerto_rico_caribe`) exists and
  needs no special-casing (a prepared split is a prepared split regardless
  of training role), but no real checkpoint has been evaluated against it.
- No orchestrator yet ties stage 6's checkpoint output directly to this
  stage's `--checkpoint` input (same "no orchestrator yet" gap noted
  throughout `docs/stages_v2/`); a run's path must be copied by hand.

## Connection to the next stage

Stage 8 (interpretability and calibrated products — PLANNED to be wired,
code already exists per `docs/architecture_v3.md`) would run Shapley
attribution and per-pixel distribution fitting against a trained
checkpoint the same way this stage runs evaluation metrics against one —
reusing the same checkpoint-loading pattern this stage established
(including the architecture-config fix above, which that stage will need
too). Beyond that, the next user-directed decisions are: run stage
4's `compute_stats()`/`build_region()` and stage 6's training scripts at
full historical scale, then re-run this stage's evaluation for real
numbers — none of that is implied by finishing this stage's infrastructure.
