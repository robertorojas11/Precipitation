# Architecture v2-generative

Living document for the migration defined in
`docs/decisions/0002-implementacion-v2.md`. Updated after each PASO lands,
not before (transversal rule). See that decision document for full
rationale; this file tracks *implementation status*.

## Two candidate architectures + shared regularizer

```text
                         ERA5 cp+lsp (hourly, per region)
                                    |
                +-------------------+-------------------+
                |                                       |
     Candidate A: cGAN                      Candidate B: Regression + EDM
     (spateGAN-ERA5-inspired)                residual diffusion (Avila 2026 /
     generator + discriminator,              Mardani et al.)
     ensemble via frozen dropout             stage 1: U-Net regression (mean)
     L_D + L_GAN(G) + L_L1(G) ensemble       stage 2: EDM diffusion (residual)
                |                                       |
                +-------------------+-------------------+
                                    |
                    + eta-learning W1 regularizer (Chang & Sapsis 2026)
                    (both candidates, and optionally the deterministic
                     baseline -- src/training/eta_learning.py)
                                    |
                    Evaluation: CRPS, FSS, percentile-MAE, RMSE/SSIM,
                    Shapley attribution, per-pixel calibrated distributions
```

Third comparison point: `src/models/deterministic_resnet.py` (planned, PASO
5.4), a fresh baseline adapted to the hourly ERA5→IMERG contract -- **not**
the archived `legacy/deterministic_unet/` (daily CHIRPS/Oya, kept for
continuity, see `legacy/README.md`).

## Data cascade design

Two supported shapes; which one a given training run uses is a
configuration choice, not a hard-coded pipeline shape (PASO 3.2):

**Single-step (default for phase 1):** `ERA5 cp+lsp -> model -> IMERG`
(hourly target directly). Simpler, fewer moving parts, matches RainShift's
own task formulation exactly.

**Two-stage cascade (design documented, not required for phase 1):**
`ERA5 cp+lsp -> model_stage1 -> IMERG` (supervised, hourly) followed by
`IMERG -> model_stage2 -> Oya` (refinement to Oya's characteristics/detail
level). Rationale: IMERG is the higher-trust, globally-consistent target;
Oya is a secondary, lower-trust refinement source (limitations in
`docs/DATA_SOURCES.md`). A cascade lets a second, smaller model learn only
the IMERG-to-Oya adjustment instead of re-learning the full ERA5-to-target
mapping twice. Not implemented in phase 1: no training run currently
requires Oya as a final target rather than a continuity/comparison source.

## Region framework

`src/data_preprocessing/regions.py`: `mexico_pacifico`,
`mexico_atlantico_golfo`, `puerto_rico_caribe`. Each is a bounding box
intersected with a tropical-cyclone basin polygon
(`ATLANTICO_SHP_PATH`/`PACIFICO_SHP_PATH`). Used uniformly by acquisition
(`era5_cds.py`, `imerg.py`, region-scoped `export_oya.py`) and, going
forward, by preprocessing/training/evaluation. Adding a region is a registry
entry, not a pipeline rewrite (mirrors RainShift's `A1 ⊂ A2 ⊂ A3 ⊂ A4`
mechanism).

## Traceability table (M101–M266 + P3/P4 additions)

Status values: **DONE** (implemented + tested), **PARTIAL**, **PLANNED**
(designed, not yet implemented), **DEFERRED** (explicitly out of scope for
this phase, per `docs/decisions/0002-implementacion-v2.md`), **N/A** (paper
requirement doesn't apply to this project's scope).

### Framework / data (P2 block, reused from the RainShift skeleton)

| Block | Item | Status | Where |
|---|---|---|---|
| B.1 | Region objects, disjoint train/eval sets | DONE | `src/data_preprocessing/regions.py` |
| B.1 | LR ERA5 channels (cp+lsp) + HR invariants | PARTIAL | `era5_cds.py` acquires cp+lsp; HR invariants (land-sea mask, orography) not yet wired into a per-region prepared dataset |
| B.1 | Target: IMERG V07, hourly | DONE (acquisition) / PLANNED (dataset build) | `src/data_extraction/imerg.py` |
| B.2 | Clip ERA5 by IMERG min/max, log transform, pooled-variance normalization | PLANNED | not yet ported to the new region-scoped dataset builder |
| B.4 | Hierarchical region-union training scenarios (A1–A4) | DEFERRED | mechanism supported by the region registry; not exercised with a multi-region union yet (single-region start, per 0001/0002) |
| B.5 | CRPS (Eq. 1), percentile-MAE (Eq. 2) | PLANNED | `src/evaluation/` (PASO 8) |
| B.6 | Quantile mapping / few-shot fine-tuning domain adaptation | DEFERRED | Phase 4 groundwork only (§1.5 of 0002) |

### Candidate A — GAN (P1-inspired)

| Item | Status | Where |
|---|---|---|
| Generator (residual-block backbone, Softplus tail, ensemble via noise + frozen dropout seed) | DONE | `src/models/generator.py` |
| Discriminator (separate HR/LR paths, anti-quantization noise) | DONE | `src/models/discriminator.py` |
| `L_D`, `L_GAN(G)`, `L_L1(G)` ensemble (Eqs. 1–4) | DONE | `src/training/losses_gan.py` |
| Frozen-dropout-seed ensemble inference | DONE | `ConditionalGenerator.sample()` |

### Candidate B — Regression + EDM residual diffusion (Avila 2026 / Mardani et al.)

| Item | Status | Where |
|---|---|---|
| Stage 1 regression U-Net (mean field, self-attention at coarsest resolution) | DONE | `src/models/regression_stage.py` |
| Stage 2 EDM residual diffusion, preconditioning (Eqs. 3–4) | DONE | `src/models/diffusion_residual.py` |
| Heun sampler + stochastic churn (Eqs. 6–7) | DONE | `src/models/diffusion_residual.py` |

### η-learning (Chang & Sapsis 2026)

| Item | Status | Where |
|---|---|---|
| W1 quantile representation (Eq. 13), Monte Carlo estimate (Eq. 14) | DONE | `src/training/eta_learning.py` |
| Tail-biased quantile sampling (`τ`) | DONE | `src/training/eta_learning.py` |
| ERM pre-training before activating the η term | DONE (schedule tracker; not yet wired into an actual training loop -- no trainable dataset exists yet) | `src/training/eta_learning.py::EtaLearningSchedule` |
| IICT tail-index refresh (Algorithm 1/2) | DONE (`update_tail_index`; periodic-refresh cadence via `EtaLearningSchedule`) | `src/training/eta_learning.py` |
| Configurable / gradient-norm-adaptive `λ` | DONE | `src/training/eta_learning.py::balance_lambda` |
| `ν0` sourcing per region (empirical vs. hypothesized GEVD) | DESIGNED (`docs/decisions/0002-implementacion-v2.md` §1.3); documented in full in `docs/ETA_LEARNING_NOTES.md` | `docs/ETA_LEARNING_NOTES.md` |

### Interpretability and calibrated products (Avila 2026)

| Item | Status | Where |
|---|---|---|
| Shapley attribution over input channels (exact enumeration + sampling fallback) | DONE | `src/evaluation/shapley_attribution.py` |
| Per-pixel distribution fitting (Weibull/Gamma/Log-normal/Burr XII/two-component Weibull mixture) + zero-inflation | DONE | `src/evaluation/pixel_distributions.py` |
| AIC/BIC family selection + bootstrap Anderson-Darling goodness-of-fit (rejects a lowest-AIC family that fails GoF) | DONE | `src/evaluation/pixel_distributions.py` |
| Quantile maps + threshold-exceedance products | DONE | `src/evaluation/pixel_distributions.py` |

### Deterministic baseline (continuity)

| Item | Status | Where |
|---|---|---|
| Original deterministic U-Net (daily CHIRPS/Oya) | DONE (archived, still runnable) | `legacy/deterministic_unet/` |
| Fresh deterministic baseline for the hourly contract, optional η-learning | DONE (model); η-learning wiring pending an actual training loop | `src/models/deterministic_resnet.py` |

### Operational ensemble forecasting (Phase 4)

| Item | Status | Where |
|---|---|---|
| Run trained model over an operational forecast ensemble (e.g. WeatherNext 2) | DEFERRED (documented plan only) | `docs/decisions/0002-implementacion-v2.md` §1.5 |
| Quantile-mapping calibration between ERA5 and the forecast product | DEFERRED (identified as a deliberate improvement over Avila 2026, which omits it) | same |
