# Stage 6 — Training (deterministic baseline, Candidate A, Candidate B)

## Purpose and inputs

The first stage with an actual trainable PyTorch loop: three separate
training scripts, one per model this project compares (PASO 5 of
`docs/decisions/0002-implementacion-v2.md`), all reading stage 4/5's
normalized, per-region prepared tensors directly (`norm_stats.json` +
`manifest_prepared_<region>.json`; recommended precondition, enforced by
`src/data_preprocessing/pipeline_hourly.py` for stages 1-5 but not
automatically before this stage, which stays a separate, manual step by
design: [stage 5's](05-validate-prepared.md)
`validate_prepared(region)["accepted"]`). Implementation:
`src/training/{dataset,common,eta_integration,train_baseline,train_gan,
train_diffusion}.py`.

```bash
python -m src.training.train_baseline  --region mexico_pacifico
python -m src.training.train_gan       --region mexico_pacifico
python -m src.training.train_diffusion --stage regression --region mexico_pacifico
python -m src.training.train_diffusion --stage diffusion  --region mexico_pacifico \
    --stage1-checkpoint outputs/v2_generative/mexico_pacifico/diffusion_stage1_regression/<run>/best.pt
```

Each script writes a `run.json` (config + git revision + the `norm_stats.json`
used), a growing `history.json` (one row per epoch), and `last.pt`/`best.pt`
checkpoints, under `outputs/v2_generative/<region>/<model_name>/<run_name>/` —
the same run-provenance shape as `legacy/deterministic_unet/train.py`.

## Training decision: single region per run

`src/training/dataset.py`'s `PreparedHourlyDataset` loads **one** region's
one split. This is not a missing feature: `docs/architecture_v3.md`'s B.4
row already marks the hierarchical region-union mechanism (RainShift's
`A1 ⊂ A2 ⊂ A3 ⊂ A4`) **DEFERRED**, reaffirming decision 0001's
"single-region start". Pooling `mexico_pacifico` and
`mexico_atlantico_golfo` into one batch would additionally require either
padding to a common shape or a region-homogeneous batch sampler (each
region has its own canonical grid shape — 200×160, 150×140, 130×260) —
neither exists yet, so single-region training sidesteps a problem this
project has already decided not to solve in phase 1.

One sample is one `(day, hour)` pair, not a whole day: every model here
folds its (default 1-hour) ERA5 context into the channel dimension rather
than carrying an explicit time axis (see `src/models/generator.py`'s own
design note on this), so each hour of each accepted day is a fully
independent training example. `__len__` is `accepted_days * 24`.

## NaN sanitization is a forward-pass concern only

Stage 4 deliberately leaves invalid pixels as `NaN` after the log transform
and normalization (`docs/stages_v2/README.md`'s shared experimental
contract: "never silently converted to zero rain"). A NaN fed into a
convolution poisons every pixel in its receptive field, not just the
invalid one, so `PreparedHourlyDataset.__getitem__` replaces NaN with `0.0`
(the distribution's own mean in normalized z-score space) **only** in the
tensors handed to a network. The original validity masks (`target_valid_
mask`, `input_valid_mask`, `basin_mask`) are returned unchanged, combined by
`src.training.common.loss_mask()` into the one mask every loss and metric
in this stage is restricted to, and the substitution never reaches a loss.

## Candidate 0 — deterministic ResNet baseline (`train_baseline.py`)

ERM loss: masked L1 (MAE) in physical mm/hr —
`src/models/deterministic_resnet.py`'s own docstring: "this is the
deterministic reference point for which CRPS reduces to MAE" (P1 M181, P2
M247) — not the legacy pipeline's heavier multi-term loss (occurrence /
amount / gradient / multiscale), which was tuned for the daily CHIRPS/Oya
task and isn't part of this project's P1-P4 contract. Optimizer: AdamW,
lr=3e-4, weight_decay=1e-4 (not a paper value — mirrors the legacy
baseline's own default, documented as a starting point to tune, not a
citation). Optional eta-learning via `--eta-lambda` (0 = disabled).

## Candidate A — conditional GAN (`train_gan.py`)

Optimizer hyperparameters here **are** paper values (matrix
`docs/audit/matriz_trazabilidad_metodologica.md` M155/M156/M158, P1
"Training and model selection"): generator AdamW lr=1e-4, betas=(0.0,
0.999); discriminator AdamW lr=2e-4, betas=(0.0, 0.5); batch size 9. Each
step: (1) discriminator update on `discriminator_loss` (P1 Eq. 1) using one
detached generator sample; (2) generator update on `generator_loss` =
`generator_adversarial_loss` (Eq. 2) + `ensemble_l1_loss` over 3 members
(Eq. 3) = Eq. 4.

**Masking the discriminator's real/fake fields — a documented deviation.**
P1's discriminator operates on full continuous HR grids; this project's
grids include out-of-basin/invalid pixels that must never be used as if
they were real information in a loss. A per-pixel mask cannot be applied
only to the real field or only to the fake field without teaching the
discriminator to use the mask pattern itself as a tell, so **both** fields
are multiplied by the identical `loss_mask(batch)` before either reaches
the discriminator (`_apply_mask` in `train_gan.py`). Since the exact same
zero pattern appears in both real and fake at the same spatial locations,
the discriminator cannot learn anything about rain from those pixels — it
can only ever compare the genuinely-valid ones. The ensemble L1 term uses
the same mask directly on valid pixels, same as everywhere else in this
codebase.

## Candidate B — regression + EDM residual diffusion (`train_diffusion.py`)

Two sequential sub-stages, Avila (2026) Eq. (1): `y = E[y|x] + (y - E[y|x])`.
Stage 1 (`train_regression_stage`) must finish and be frozen before stage 2
(`train_diffusion_stage`) has a conditional mean to compute residuals
against — run as two separate CLI invocations, the second taking the
first's `best.pt` via `--stage1-checkpoint`.

**Representation choice, documented explicitly.** Both sub-stages operate
in the *same normalized log-space* as `target_norm` (stage 4's output), not
physical mm/hr, unlike the baseline and the GAN generator (whose Softplus
output heads are architecturally committed to a non-negative physical
representation and could not represent a signed residual).
`RegressionUNet` is built here with `nonnegative_output=False` precisely
because this decomposition needs a *signed* conditional mean and a *signed*
residual — exactly the case `regression_stage.py`'s own docstring
anticipates ("a future residual target (e.g. a signed log-anomaly)").
Stage 1's optimizer: AdamW, lr=2e-4, weight_decay=1e-5 (Avila 2026 Table 5).

**Conditioning layout.** Stage 1 input: `cat([inputs_norm, invariants_norm])`
(4 channels). Stage 2's `EDMPrecond` conditioning tensor:
`cat([inputs_norm, invariants_norm, stage1_mean])` (5 channels); its raw
network (`SimpleNoiseConditionedUNet`) must be built with
`in_channels = 1 (scaled noisy residual) + 5 (conditioning) = 6` — easy to
get wrong, since `EDMPrecond.forward` concatenates the scaled residual
*before* calling the raw network, so the raw network's declared
`in_channels` has to cover that extra channel too, not just the
conditioning tensor's own width (`_build_denoiser`'s comment documents this
exactly after a real test failure caught the off-by-one).

**Masked EDM loss, a documented deviation from
`src.models.diffusion_residual.edm_loss`.** That function's own `.mean()`
reduction is unmasked; training the denoiser to reconstruct a residual
computed from a fabricated zero-filled target at invalid pixels would
silently use invalid data in a loss. `train_diffusion.masked_edm_loss`
reproduces `edm_loss`'s exact per-pixel weighting formula (Avila 2026 Eq.
5) and only changes the final reduction to `masked_mean` — `EDMPrecond`
itself is reused completely unmodified.

Stage 2 validation runs the real Heun+churn sampler
(`src.models.diffusion_residual.heun_sample`) to produce a residual sample,
adds it to the frozen stage 1 mean, and scores the result — the only place
in this stage where the sampler runs during training (an actual cost: 18
steps × ~2 network evaluations each, by default).

## Eta-learning wiring (`eta_integration.py`)

`docs/architecture_v3.md` previously tracked eta-learning as a "schedule
tracker; not yet wired into an actual training loop -- no trainable
dataset exists yet". This stage is what finally provides that dataset, and
`EtaRegularizer` wires `src/training/eta_learning.py` into all three
candidates (model-agnostic by construction, PASO 1.2) — disabled by default
everywhere (`--eta-lambda 0`), opt-in per run.

Two auxiliary pools are built once per run, from the training split itself:

- `reference_pool` (the paper's `nu0`): spatial maxima of **real** target
  fields — the empirical "how extreme does this region's observed rain
  actually get" distribution (`docs/ETA_LEARNING_NOTES.md`'s "empirical
  nu0" case; the hypothesized-GEVD case for a data-scarce zero-shot region
  is designed there but not wired here).
- `auxiliary_inputs`: real `(inputs, invariants)` pairs Algorithm 1/2 (IICT)
  periodically runs inference over, under `torch.no_grad()`, to find which
  inputs *currently* induce the model's most extreme outputs; the
  gradient-carrying eta forward pass is then restricted to just the `n_q`
  selected examples (`update_tail_index`) — the memory-saving restriction
  the paper's Algorithm 2 is for.

**A found-and-fixed bug while wiring this up:**
`eta_learning.spatial_maximum_observable()` only reduces the trailing two
`(H, W)` dimensions by design (its own docstring says so explicitly), so a
`(B, 1, H, W)` precipitation field comes back as `(B, 1)`, not `(B,)`.
`update_tail_index`'s `torch.argsort` needs a genuinely 1-D array to rank
across the whole batch; `argsort` on a `(B, 1)` tensor sorts each length-1
row independently — a silent no-op that produced a degenerate,
all-identical tail index and a 5-D indexing crash a few calls later
(`too many values to unpack (expected 4)` deep inside
`DeterministicResNet.forward`, caught by `TrainBaselineTest
.test_eta_learning_runs_when_enabled`). Fixed with a small
`_per_sample_observable()` wrapper that reshapes to `(B,)` right after
calling the shared `eta_learning` function, used at all three call sites in
`eta_integration.py` — `eta_learning.py` itself was not changed, since its
documented contract ("the trailing two dims are reduced away") is correct;
the bug was in not accounting for the leftover channel dimension at the
call site.

**Using the sanitized (NaN → 0) field for the spatial-maximum observable is
deliberately not an approximation**: precipitation is non-negative, so a
zero-filled invalid pixel can never exceed a genuine valid-pixel value and
can never become the max unless every valid pixel in that field is itself
exactly zero (in which case the max is correctly 0 either way) — no masking
is needed before taking the max, unlike every other loss in this stage.

**Eta-learning is wired onto stage 1 only for Candidate B.** Stage 2's
observable would require running the full Heun sampler inside the training
loop just to get one sample to regularize against — a real compute cost,
documented here as a known limitation rather than silently skipped.

## Tests

`tests/test_training.py` — 7 tests, against a small synthetic region (6.4°
× 6.4° → a 64×64 canonical grid, deliberately **not** tiny: the
discriminator's 4-stage downsampling collapses a too-small grid, e.g.
20×20, to 1×1 and then crashes reflect-padding a size-1 dimension — 64
halves cleanly to 4 and comfortably exceeds every real region's smallest
dimension, `puerto_rico_caribe`'s 130×260). Covers: `PreparedHourlyDataset`
length/shape/dtype/NaN-sanitization and its `ValueError` on an unknown
split; one training epoch each for the baseline, the GAN, and Candidate
B's stage 1 → stage 2 sequence, asserting checkpoints and history are
written and every loss stays finite; eta-learning enabled end-to-end on the
baseline (`eta_lambda=0.1, refresh_every=1`), exercising pool construction,
tail-index refresh, and the regularizer term together. 187 tests pass
across the full suite.

Two real bugs were caught by these tests before any real-data run could
have hit them silently: `PooledMetrics.update()`'s boolean-mask indexing
didn't broadcast a `(B, H, W)` mask against a `(B, 1, H, W)` field (fixed in
`src/training/common.py`), and the eta-learning observable-shape bug above.

## Known limitations

- No real-scale run yet: only smoke-tested on tiny synthetic data, since
  stage 4's `compute_stats()`/`build_region()` haven't been run at full
  historical scale either (`docs/stages_v2/04-prepare.md`'s own "Known
  limitations"). Real training is the next user-directed step.
- No mixed precision / `torch.amp` (unlike `legacy/deterministic_unet
  /train.py`, which uses it) — kept out for a first working version;
  worth adding once real GPU runs are underway and iteration speed matters.
- `PreparedHourlyDataset` re-reads and fully decompresses a day's NPZ on
  every `__getitem__` call, even when 24 consecutive indices request the
  same day — no caching. Acceptable for this stage's tests; a real
  per-region epoch (thousands of days) would benefit from a small
  per-worker LRU cache, not built here to avoid premature optimization
  before real training throughput is actually measured.
- Candidate B's eta-learning gap (stage 2 only) and the GAN's
  discriminator-masking deviation, both above, are deliberate and
  documented, not oversights.
- `src/data_preprocessing/pipeline_hourly.py` orchestrates stages 1-5 (the
  dataset-building side) but deliberately does not extend into training --
  nothing yet enforces stage 5's validation gate automatically before a
  training script runs; check it by hand first.

## Connection to the next stage

Stage 7 (evaluation — PLANNED) consumes each candidate's `best.pt`
checkpoint against `src/evaluation/{probabilistic_metrics,
deterministic_metrics,model_comparison,baselines}.py` (all already built,
per `docs/architecture_v3.md`'s B.5 row, but with "no real training run yet
to feed it") — CRPS, percentile-MAE, FSS, RMSE/SSIM, and the
geographic-generalization comparison on `puerto_rico_caribe` (zero-shot,
never trained on), across the deterministic baseline, Candidate A, and
Candidate B, with and without eta-learning.
