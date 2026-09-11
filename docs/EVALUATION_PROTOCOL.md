# Evaluation protocol — v2-generative framework

PASO 8 and PASO 10 of `docs/decisions/0002-implementacion-v2.md`. Describes
*how* a trained candidate is scored, not the training procedure itself
(see `docs/architecture_v3.md` for architectures and
`docs/ETA_LEARNING_NOTES.md` for the extreme-event regularizer).

## Metrics

| Metric | Source | Module | Notes |
|---|---|---|---|
| CRPS | P1 Eq. 6-7 / P2 Eq. 1 | `src/evaluation/probabilistic_metrics.py::crps_ensemble` | Reduces exactly to MAE at ensemble size 1 (deterministic baseline). |
| FSS / mFSS / meFSS | P1 Eq. 9 | `probabilistic_metrics.py::{fractions_skill_score,mean_fss,mean_ensemble_fss}` | Scales/thresholds per P1 M184 (`FSS_SCALES_KM`, `FSS_THRESHOLDS_MM_H`). |
| Percentile-MAE (p95, p99) | P2 Eq. 2 | `probabilistic_metrics.py::percentile_mae` | |
| RAPSD | P1 M186 | `probabilistic_metrics.py::rapsd`/`mean_rapsd` | pysteps if installed, numpy FFT fallback otherwise. |
| Rank histogram | P1 M188 | `probabilistic_metrics.py::rank_histogram` | Calibration diagnostic, not a scalar score. |
| Linear eccentricity | P1 Eq. 10-12 | `probabilistic_metrics.py::linear_eccentricity` | Marked EXPERIMENTAL: moment-ellipse approximation of the 0.5-PCC contour, not the exact iso-line P1 describes. |
| R2 / RMSE / MAE / BIAS | continuity (legacy protocol) + P1 Eq. 13 (BIAS) | `src/evaluation/deterministic_metrics.py` | Kept per PASO 8 as complementary, not a replacement for the probabilistic metrics. |
| Full-field RMSE/SSIM | PASO 8 (new) | `deterministic_metrics.py::full_field_rmse_ssim` | Secondary diagnostic, see "Pricing the eta-learning regularizer" below. |
| Bilinear-interpolation baseline | P1 M199 / P2 M225 | `src/evaluation/baselines.py::bilinear_interpolation` | Every relative score is quoted against this lower bound. |
| rainFARM baseline | P1 M195 | `baselines.py::rainfarm` | **Not implemented.** Raises `NotImplementedError` rather than being silently substituted or skipped. |

## Geographic generalization protocol

Reuses the RainShift (P2) skeleton (`docs/decisions/0002-implementacion-v2.md`
§1.1): evaluate each candidate in-region (trained and evaluated on the same
region) and, once more than one training region is used, out-of-region
(trained on one region's data, evaluated zero-shot on another --
`mexico_pacifico` / `mexico_atlantico_golfo` / `puerto_rico_caribe`,
`src/data_preprocessing/regions.py`). Quantile mapping and few-shot
fine-tuning (P2 M255-M258) are the designated correction strategies if the
out-of-region drop is large; neither is implemented yet (tracked ABSENT in
`docs/architecture_v3.md`, deferred to Phase 4 groundwork alongside the
operational-ensemble quantile-mapping step).

## Cross-model comparison

`src/evaluation/model_comparison.py` aggregates per-`(model, eta_enabled)`
metric dictionaries into the three-way comparison PASO 8 requires:
deterministic baseline, Candidate A (GAN), Candidate B (regression +
residual diffusion), each evaluated with and without the eta-learning
regularizer. `eta_learning_contribution()` isolates, per model, the signed
effect of turning eta-learning on -- this is how "how much does each piece
contribute separately" gets answered once real training runs exist to feed
it.

## Pricing the eta-learning regularizer

Per PASO 8 and `docs/decisions/0002-implementacion-v2.md` §1.2: "se espera
un incremento modesto [en RMSE de campo completo], en línea con el ~8% de
RMSE reportado en el paper de referencia -- si el incremento observado es
mucho mayor, es señal de que algo no está bien calibrado."

`deterministic_metrics.py::eta_learning_cost_report(baseline_scores,
eta_scores)` computes the percent RMSE increase and SSIM decrease between a
model's eta-learning-off and eta-learning-on runs, and flags
`miscalibrated=True` when the RMSE increase exceeds
`flag_multiplier * expected_rmse_increase_pct` (defaults: multiplier 2x,
reference 8%, taken from Chang & Sapsis 2026, "Correcting statistical biases
of deep generative models": full-grid RMSE rose from 2.878 to 3.112 -- a
relative increase of 8.13% -- with SSIM declining only 0.30% relative, when
the eta-learning term was activated).

## Per-pixel calibrated distributions (the alert product)

Documented in full in `src/evaluation/pixel_distributions.py`'s module
docstring; summarized here as the evaluation-facing methodology (Avila 2026
§6.4, PASO 7):

1. Pool every generative sample (ensemble draws from Candidate A/B, plus
   Phase 4 operational-forecast-ensemble draws once implemented) at a pixel
   into one value set.
2. **Zero-inflation first.** Precipitation has a point mass at zero
   (probability of no rain), unlike Avila's always-positive wind speed --
   PASO 7's explicit caveat. `fit_pixel_distribution()` separates
   `p_zero = P(no rain)` from the continuous fit of the positive amounts.
3. Fit Weibull, Gamma, Log-normal, Burr XII, and a two-component Weibull
   mixture (EM) to the positive values by maximum likelihood.
4. Rank by AIC/BIC, **but do not accept the lowest-AIC family blindly**:
   validate every candidate with a parametric-bootstrap Anderson-Darling
   goodness-of-fit test (`bootstrap_ad_pvalue`) and select among the
   families that pass at `alpha=0.05`. If every candidate is rejected,
   fall back to the lowest-criterion family and flag
   `goodness_of_fit_passed=False`. This mirrors Avila 2026's own finding
   that a top-AIC family (Burr XII at the eye centre) can still be
   rejected by the goodness-of-fit test.
5. Derive the alert-facing products from the selected fit:
   `quantile_map()` (e.g. p90/p99 maps) and
   `ZeroInflatedFit.exceedance_probability(threshold)` (probability of
   exceeding an operational "atypical rain" threshold). spateGAN-ERA5 uses
   25 mm/h as its severe-warning threshold -- a starting reference for this
   project, not yet validated for our regions with the domain team
   (`docs/decisions/0002-implementacion-v2.md` OQ-5).

## Interpretability: Shapley attribution

`src/evaluation/shapley_attribution.py` (PASO 6, Avila 2026 §6.2): for a
trained model, compute Shapley values over its input channels with respect
to the predicted-peak-precipitation observable, to check empirically which
channels the model actually relies on -- not just which ones physical
argument suggests should matter. Exact enumeration is used at this
project's channel counts (mirroring Avila's 12-channel exact case);
permutation-sampling Monte Carlo is the fallback if a future channel set
makes exact enumeration impractical (`shapley_values()` dispatches
automatically via `max_exact_channels`).

## What is not yet evaluable

No prepared hourly ERA5->IMERG dataset or trained model exists yet (PASO 3
acquisition modules are implemented but have not been run -- no CDS/
Earthdata credentials, `docs/DATA_SOURCES.md`). Every metric and harness in
this document is implemented and unit-tested against synthetic/reference
values, not yet exercised against a real training run. `docs/architecture_v3.md`
tracks this status per component.
