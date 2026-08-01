# Precipitation Downscaling: Codebase and Results Review

**Review date:** 2026-07-31  
**Ground truths:** CHIRPS and Google Oya  
**Reviewed material:** source code, scripts, tests, documentation, training logs, checkpoints and metadata, current metric JSON files, comparison/bias figures, fast datasets, and representative raw and processed files under `/mnt/data-r2/RobertoRojas/downscaling/`.

## Executive summary

The CHIRPS pipeline has learned useful spatial signal, especially in the Sierra Madre Occidental (SMO), but its reported scores are biased by invalid-pixel handling and its global performance remains weak. The latest deterministic CHIRPS result improves raw RMSE from 7.46 to 5.10 mm/day and correlation from 0.262 to 0.421; in the hard-coded SMO window it reaches R² 0.290 and correlation 0.553. This is promising, but not yet a defensible final result because the evaluation includes ocean/out-of-coverage fill pixels, the implemented two-stage target differs from the documented method, and no simple downscaling baselines or uncertainty intervals are reported.

The Oya result is currently **not scientifically interpretable as model performance**. A substantial portion of 2024–2025 Oya rasters contains unhandled fill values (`3.4028235e+38` and values near `1e20`). These values propagate through the processed and fast datasets, then the evaluator clips them to 300 mm/day. This directly produces the horizontal stripes in the Oya comparison figures, extreme test totals, near-zero CSI, and the nearly uniform −100% bias map. Oya data integrity must be repaired and the complete Oya pipeline rebuilt before tuning the model.

The highest-value work is therefore not a larger GAN. The correct order is: repair data and masks, rebuild derived artifacts, establish trustworthy baselines and evaluation, then improve the deterministic model, and only afterward reintroduce adversarial and stochastic components.

## Result inventory and provenance

### Dataset and artifact counts

| Target | Metadata-valid train | Fast train | Validation | Test | Main checkpoint set |
|---|---:|---:|---:|---:|---|
| CHIRPS | 5,842 | 5,819 | 1,096 | 1,094 | GAN-1, GAN-2, best/final/epoch files, bias corrector |
| Oya | 5,813 | 5,813 | 1,095 | 1,090 | GAN-1, GAN-2, best/final/epoch files, bias corrector |

The CHIRPS fast cache is missing 23 records that the metadata index marks valid. Training used the 5,819 cached records, not all 5,842 records used to calculate normalization statistics. This does not necessarily explain the model behavior, but it breaks artifact parity and reproducibility.

There are two copies of result metrics:

- Current repository outputs: `outputs/{chirps,oya}/metrics_*.json`, updated on 2026-07-31.
- Older external metadata: `/mnt/data-r2/RobertoRojas/downscaling/era5_oya_mexico/metadata/metrics_*.json`, dated 2026-07-27.

They disagree materially, especially for stochastic metrics. The repository files are treated as the latest results below, while the external copies are retained as evidence that result provenance is not currently controlled.

### Latest reported metrics

| Metric | CHIRPS raw | CHIRPS corrected | Oya raw | Oya corrected |
|---|---:|---:|---:|---:|
| Global MAE (mm/day) | 1.181 | **1.107** | 6.821 | **6.573** |
| Global RMSE (mm/day) | 7.464 | **5.101** | **36.951** | 37.070 |
| Global R² | −1.014 | **0.059** | **−0.026** | −0.033 |
| Global correlation | 0.262 | **0.421** | −0.004 | 0.000 |
| CSI ≥1 mm/day | **0.382** | 0.353 | 0.000 | 0.000024 |
| CSI ≥10 mm/day | 0.216 | **0.242** | 0.000 | 0.000013 |
| CSI ≥25 mm/day | 0.100 | **0.103** | 0.000 | 0.000009 |
| KS statistic | 0.306 | **0.147** | 0.692 | **0.329** |
| CRPS (10 members) | — | 1.300 | — | 6.507 |

| SMO metric | CHIRPS raw | CHIRPS corrected | Oya raw | Oya corrected |
|---|---:|---:|---:|---:|
| MAE (mm/day) | **0.962** | 1.023 | 9.939 | **9.562** |
| RMSE (mm/day) | 3.537 | **3.333** | **48.505** | 48.635 |
| R² | 0.200 | **0.290** | **−0.035** | −0.040 |
| Correlation | 0.490 | **0.553** | −0.032 | 0.000 |

Bold values only indicate the better value within the raw/corrected pair. They do not imply acceptable skill. CHIRPS and Oya absolute errors should also not be compared directly until both targets have valid masks and consistent climatological sampling.

## Ground-truth findings

### CHIRPS

1. **There is real but limited predictive skill.** The latest corrected result has positive global R² (0.059), correlation 0.421, and better high-threshold CSI than the raw output. Skill is stronger in the SMO window (R² 0.290, correlation 0.553), consistent with useful large-scale and orographic signal.

2. **Bias correction improves distribution and RMSE but not all event metrics.** It reduces global RMSE by 31.7% and KS by 51.9%, but worsens ≥1 mm/day CSI and slightly worsens MAE in the SMO window. This is expected from a global quantile mapping that changes marginal distributions without repairing event location.

3. **The visual fields are physically recognizable but spatially displaced and over-sharpened.** Rainy-date figures show the model reproducing broad wet regions, yet intense precipitation is often shifted, coastal edges are exaggerated, and predictions saturate at the evaluator's 300 mm/day cap. GAN-2 does not consistently improve the displayed sample MAE relative to GAN-1.

4. **The bias map is dominated by mask/coastline behavior.** Strong edge bands and extensive negative bias appear over water and outside valid CHIRPS coverage. These pixels should never contribute to model selection or scientific scores.

5. **CHIRPS missing pixels become synthetic precipitation.** Raw CHIRPS arrays are NaN outside valid coverage. After normalization, `torch.nan_to_num(..., nan=0.0)` makes the target equal to the normalized mean, not physical zero. With current statistics, normalized zero denormalizes to approximately 0.539 mm/day. A sample audit consequently showed a repeated 0.539 mm/day median/floor. This affects training targets, validation, evaluation, ranking, and plots.

6. **Current CHIRPS claims are somewhat overstated in documentation.** The result is promising for a prototype, but weak global R², low extreme CSI, missing baselines, invalid-pixel leakage, and lack of confidence intervals prevent calling it a validated downscaling system.

### Oya

1. **The test target contains catastrophic fill-value contamination.** Multiple Oya fast test files contain the float32 maximum `3.402823466e+38`. Raw and processed examples also contain large finite values near `1e20`, although the GeoTIFF declares no nodata value. Examples include 2024-04-24, 2024-09-28, 2024-10-17, and 2024-11-15, with many additional affected days in 2024 and 2025.

2. **The evaluator hides rather than rejects corruption.** `denormalize()` clamps observations and predictions to 300 mm/day. As a result, invalid fill values become plausible-looking 300 mm/day rain. On 2024-11-15, nearly half of the processed target is larger than `1e20`; the corresponding visualization shows regular horizontal 300 mm/day stripes and reports a 145.79 mm/day GAN-2 MAE.

3. **The “rainiest Oya days” ranking selects corrupted files.** Top totals of roughly 48 million mm/day-pixels are driven by capped fill values, not meteorological events. These dates must not be used for qualitative model assessment.

4. **Latest Oya scores describe a near-constant dry model against contaminated observations.** Global and SMO correlations are effectively zero, R² is negative, and CSI is zero or near zero at every threshold. The current model output maxima in displayed cases are below 1 mm/day while valid Oya regions contain substantial rainfall.

5. **There are two inconsistent Oya aggregation paths.** `gee_extractor.py` uses valid-slot mean × 24 and masks pixels with fewer than 30 slots. `pipeline_runner.py`, the end-to-end entry point, uses sum × 0.5 without a per-pixel slot-count mask. The latter is vulnerable to missing 30-minute scans and is consistent with the observed scan-line artifacts.

6. **Oya is median-filtered twice in the normal workflow.** `npz_converter.py` applies a 3×3 median filter, then `prepare_fast_dataset.py` applies another 3×3 median filter by default. This changes the target twice, suppresses legitimate extremes, and prevents clear provenance. Median filtering also does not solve missing-scan or fill-value contamination.

7. **Oya invalid/masked pixels also become a positive floor.** With Oya statistics, normalized zero denormalizes to approximately 0.733 mm/day. Validation and training samples contain large repeated areas at this value after missing data are zero-filled in normalized space.

8. **Oya should remain a separate experimental target.** Once cleaned, it may still differ systematically from CHIRPS because it is an AI/satellite precipitation estimate rather than a station-blended product. It needs its own QC thresholds, coverage flags, climatology, and evaluation narrative; it should not be treated as interchangeable truth.

## Pipeline and methodology findings

### Implemented architecture differs from the documented architecture

- Documentation says GAN-1 predicts ERA5-Land precipitation at 10 km. The actual fast-dataset builder creates `real_10km` by bilinearly downsampling the final CHIRPS or Oya 5 km target. No independent ERA5-Land intermediate target is loaded.
- The surface input itself comes from `ECMWF/ERA5_LAND/HOURLY`, exported at approximately 25 km, while pressure-level inputs come from ERA5. Calling the first stage “ERA5 → ERA5-Land” is therefore misleading.
- Documentation describes PixelShuffle, but both generators use bilinear `nn.Upsample` followed by convolution.
- Documentation and `agents.md` describe a conditional Gaussian process/manifold alignment component, NMI, and ECDF evaluation. No CGP implementation, NMI metric, or explicit ECDF metric exists in the executable pipeline.
- The claimed Mexican shapefile evaluation mask is not used by the evaluator. The SMO is a hard-coded pixel rectangle (`rows 100:300`, `columns 220:340`) with no geographic or elevation validation.

These are not naming-only discrepancies: they change the scientific experiment being run. The project must either implement the stated method or update its research claims to match the code.

### Data preparation and masks

- `valid_flag=True` means that required files existed or conversion returned successfully; it does not validate target ranges, valid-pixel fraction, temporal completeness, CRS/transform consistency, or array finiteness.
- Existing processed NPZ files are automatically trusted and indexed as valid without reopening them for QC. This is how contaminated Oya files remain valid.
- Reprojection arrays are created with `np.empty`, and nodata/source masks are not explicitly propagated. Reprojection should initialize destination nodata and carry a validity mask.
- Negative target values are silently clipped to zero. For Oya, observed negative fractions such as −0.25 and −0.5 may be sentinel values and must be masked, not interpreted as no rain.
- Inputs and targets are resampled with bilinear interpolation. This is reasonable for atmospheric state variables but does not conserve precipitation totals; conservative/area-weighted remapping should be considered for accumulated precipitation.
- The land mask uses `elevation > -100 m` and falls back to all ones. It does not encode target availability or political/study-domain coverage. Losses require the intersection of land, domain, source-valid, and target-valid masks.
- The hard-coded crop assumes a fixed export transform. Although inspected files currently align to the intended 35–12°N, 120–84°W window, cropping by geospatial bounds is safer than relying on array indices.
- The CHIRPS index/statistics/cache mismatch shows that derived artifacts have no manifest or content hash tying them to their inputs and configuration.

### Normalization and units

- Statistics correctly use only the training split, but they ignore NaNs without storing a target-valid mask for later stages.
- ERA5-Land total precipitation remains in meters while plots label it as mm/day. It is normalized consistently for learning, but displayed physical values are off by a factor of 1,000 and direct input/target comparisons are misleading.
- The code assumes all precipitation transformations use `log1p` but does not record a complete preprocessing version in caches or checkpoints.
- The output clamp at normalized ±5 and the later physical 300 mm/day clamp can hide instability and create saturation. Invalid observations must be rejected before any physical clipping; prediction clipping should be reported as a diagnostic rate.

### Training behavior

- Both latest large-model runs stop after only 5–6 epochs despite requesting 30. CHIRPS validation metrics fluctuate strongly; Oya never achieves positive validation R². A patience of three on mean all-pixel MAE is too sensitive and favors dry climatology.
- Discriminator losses rapidly approach zero while generator losses stay high. This is classic discriminator domination, not balanced adversarial training.
- Pixel losses are multiplied by a mask and then averaged over the entire rectangle. This scales the loss by the valid fraction; it should instead divide by the number of valid pixels.
- Only the L1 pixel loss is land-masked. Discriminator, spectral, occurrence, validation, and evaluation terms still see invalid/fill regions. These regions can dominate both optimization and early stopping.
- The occurrence loss is unmasked. Its targets therefore include normalized fill pixels as wet or dry depending on the target statistics.
- The uncommitted “extreme focal loss” is a masked L1 term, not focal loss. It uses a global mean over mostly zeroed pixels, has no normalized valid-count denominator, and is not exposed in configuration or logged separately.
- The two stages are trained sequentially, and GAN-2 receives GAN-1 output only. There is no teacher-forcing schedule, end-to-end fine-tuning, or ablation proving that stage 2 and physics channels add skill.
- Only PyTorch is seeded. DataLoader workers, NumPy stochastic correction, Python random, and deterministic backend behavior are not fully controlled.
- Final checkpoints contain weights only. They omit optimizer/scheduler state, epoch, best metric, preprocessing fingerprint, model arguments, source commit, and random seeds.

### Bias correction and probabilistic output

- Global quantile mapping pools every date and pixel. It cannot correct spatial displacement, seasonal bias, elevation-dependent bias, or regional climatology.
- The GPD is fitted to pooled pixels above a global 95th percentile, which mixes regimes and creates massive spatial dependence/pseudoreplication.
- The stochastic spectral filter is derived from pooled residuals and reused everywhere. It is not conditioned on the forecast, season, location, or rain occurrence.
- Stochastic generation uses unseeded global NumPy randomness. Repeated evaluations are not reproducible.
- The older external metric files report implausible stochastic MAE near 144–147 mm/day and CRPS near 78–82, whereas current outputs report CRPS 1.30 and 6.51. This large change without a run manifest demonstrates unstable result provenance.
- Ten ensemble members are too few for stable tail verification. Reliability, rank/PIT histograms, spread-skill, and threshold Brier scores are absent.

### Evaluation

- Metrics flatten all dates and pixels, overweighting spatial/temporal dependence and dry/invalid regions. They need per-day and per-region aggregation with uncertainty intervals.
- No persistence, climatology, bilinear ERA5, quantile-mapped ERA5, or deterministic CNN baseline is reported. Positive R² alone does not prove added value from downscaling.
- CSI is aggregated globally. Report probability of detection, false-alarm ratio, frequency bias, equitable threat score, and neighborhood/FSS scores for displacement-tolerant precipitation verification.
- KS compares a random set of individual pixels as if independent and measures only the marginal distribution. It should be accompanied by wet-day frequency, quantile bias, seasonal distributions, and block-bootstrap uncertainty.
- The relative-bias map divides by `mean_obs + 0.1`, saturates at ±100%, and includes invalid/dry cells. It should mask low-climatology and invalid cells and be paired with absolute bias.
- Test observations are clipped to 300 mm/day. Observations should never be silently clipped as part of QC; rejected and capped counts must be reported separately.
- Current figures use independently repeated color bars and fixed 0–100 display limits while titles can report maxima of 300. Add shared scales, valid-domain outlines, dates, and difference/error panels.

## Code quality, tests, and reproducibility

| Area | Finding | Impact |
|---|---|---|
| Version control | `.gitignore` ignores the entire `scripts/` directory | Core data preparation/training utilities are not reproducibly versioned |
| Fast-data script | `prepare_fast_dataset.py` calls `tqdm` without importing it | A clean execution raises `NameError` |
| Configuration | Importing `Config` creates external directories and prints messages | Imports have side effects; tests and tools mutate environment unexpectedly |
| Tests | `pytest` is not installed; tests are mostly executable integration scripts | No automated regression suite can currently be collected or run |
| Preprocessing test | `test_preprocessing.py` rewrites the live index and one processed file, then restores them | A failed/interrupted test can damage real data |
| GEE tests | Sampling/auth tests can authenticate or submit external exports | They are not isolated unit tests and should be explicitly gated |
| Error handling | Broad `except Exception` blocks often log and continue | Partial datasets can be reported as successfully completed |
| Existing files | Converter trusts any pre-existing NPZ and marks it valid | Stale or corrupt artifacts survive indefinitely |
| Destructive utilities | Repair scripts delete processed files directly | Operations need dry-run, manifest, quarantine, and confirmation controls |
| Dependencies | Requirements are unpinned | Environments and numerical results cannot be reproduced reliably |
| Documentation | Commands and architecture descriptions do not match current CLIs/code | Researchers may run the wrong experiment or report nonexistent components |

Credential files and `.env` are ignored and are not currently tracked, which is correct. Do not include their contents in diagnostics or reports.

## Recommended improvement roadmap

### P0 — Make the data trustworthy

1. Define a shared precipitation QC contract for raw and processed targets:
   - finite values only;
   - explicit dataset-specific nodata/sentinel handling;
   - plausible range checks before transformation (for example, flag rather than clip values above a documented daily threshold);
   - per-file valid fraction and 30-minute slot completeness;
   - transform, CRS, bounds, and shape validation;
   - quarantine failed dates and set `valid_flag=False` with a reason code.
2. Consolidate Oya aggregation into one implementation. Use per-pixel valid-slot counts, mask incomplete pixels, preserve the count/coverage field, and verify whether the native band is an instantaneous rate or interval accumulation before choosing mean × 24 or sum × 0.5.
3. Remove median filtering from conversion and fast-cache creation. If retained as an experiment, apply it exactly once, after masking, with the filter choice recorded in a manifest and compared against an unfiltered target.
4. Preserve `target_valid_mask`, `input_valid_mask`, and study-domain/land masks in every NPZ. Fill tensors only after normalization, and never treat filled values as observations or include them in losses/metrics.
5. Rebuild Oya processed files, normalization statistics, fast caches, rainy-day rankings, checkpoints, bias correctors, metrics, and figures from clean raw data. Rebuild CHIRPS caches as well to eliminate the 23-file mismatch and positive missing-data floor.
6. Add a preflight report that blocks training on nonfinite values, sentinel values, cache/index count differences, invalid dates, or unexpected distribution shifts by year/split.

**P0 acceptance gate:** zero nonfinite/sentinel target values entering transformations; every sample has a validity mask and QC record; cache counts match the accepted index; train/validation/test target quantiles are plausible by year; corrupted Oya dates are excluded or re-exported.

### P1 — Establish defensible evaluation

1. Build geospatial masks from raster transforms and the intended Mexico/SMO polygons. Evaluate only the intersection of target-valid coverage and the requested region.
2. Add four baselines for each target: zero/dry climatology, daily/seasonal climatology, bilinear ERA5 precipitation in correct mm/day units, and a small deterministic residual CNN/U-Net trained with masked loss.
3. Report per-date and pooled MAE/RMSE/R²/correlation; wet-day frequency; CSI/POD/FAR/frequency bias at 1, 10, and 25 mm/day; FSS at multiple neighborhoods; and quantile bias through at least the 99.9th percentile.
4. Stratify results by season, elevation band, region, rain intensity, and target coverage. Use blocked bootstrap intervals by date/month rather than treating pixels as independent.
5. Store one immutable run directory containing configuration, source commit/diff state, dataset manifest hash, checkpoint, bias-corrector parameters, metrics, figures, environment versions, seeds, and timestamps.
6. Select checkpoints with a masked composite score that reflects occurrence, amount, and spatial skill; do not use unmasked all-pixel MAE alone.

**P1 acceptance gate:** the learned model beats bilinear ERA5 and climatology on held-out years for both continuous and event metrics; all scores can be reproduced from a run manifest; no invalid pixels enter any metric.

### P2 — Improve the deterministic model

1. Resolve the experiment definition. Either ingest a real independent ERA5-Land 10 km target for GAN-1, or describe stage 1 honestly as supervised coarse prediction of a downsampled final target. The latter may be simplified to one multiscale network.
2. Train a strong deterministic baseline before adversarial training. Recommended starting point: residual U-Net/ResNet with static DEM and land mask, log1p or a two-part occurrence/amount head, masked Huber/L1 loss, and conservative precipitation inputs in mm/day.
3. Normalize masked losses by valid-pixel count. Apply validity masks consistently to pixel, occurrence, spectral, validation, and any discriminator losses.
4. Use a two-part precipitation objective: calibrated wet/dry probability plus conditional positive amount. Add event-balanced sampling or quantile-weighted loss for heavy rain only after data QC; log every component independently.
5. Run ablations for atmospheric bands, DEM, upslope, spectral prior, stage 2, adversarial loss, and extreme weighting. Keep a component only if it improves held-out spatial/event skill.
6. Tune physical priors from data and verify units. The spectral model currently uses domain-mean winds and fixed time constants, and the upslope field is only a proxy; both need documented dimensional interpretation and ablation evidence.

**P2 acceptance gate:** deterministic results improve over all baselines across multiple held-out years and regions; GAN-2/physics additions provide statistically supported incremental skill; output saturation is rare and reported.

### P3 — Add probabilistic and adversarial refinement

1. Reintroduce a discriminator only after deterministic convergence. Use masked/conditional discrimination, weaker or adaptive discriminator updates, and validation criteria tied to precipitation skill rather than adversarial loss.
2. Prefer a probabilistic head or conditional residual model over unconditional additive spectral noise. Calibrate occurrence and amount jointly, and condition uncertainty on season/location/intensity.
3. If quantile mapping remains, fit it with cross-validation and condition it by season/region while preserving held-out test independence. Treat spatial correction separately from marginal calibration.
4. Increase ensemble size for evaluation and add CRPS, Brier scores, reliability diagrams, rank/PIT histograms, spread-skill, and extreme-event calibration.

**P3 acceptance gate:** probabilistic output improves CRPS/Brier score over deterministic and climatological ensembles, is reliable by threshold, and does not degrade deterministic event location.

## Target-specific next experiments

### CHIRPS

1. Re-evaluate the existing checkpoint with correct masks before retraining; this will establish how much of the current score comes from invalid pixels.
2. Train the deterministic masked baseline and compare it with the current GAN using identical splits and baselines.
3. Run physics/stage ablations, then tune heavy-rain sampling and occurrence/amount losses.
4. Only retain quantile mapping if it improves both distributional and event metrics on validation without degrading spatial skill.

### Oya

1. Stop model tuning and mark all current Oya metrics/figures as invalid for scientific comparison.
2. Audit native Oya metadata and aggregation units, re-export affected days with slot-count bands, and generate annual coverage/value-range summaries.
3. Compare clean Oya with CHIRPS on common valid pixels by date, season, and intensity to quantify irreducible target disagreement.
4. Train first on a conservative clean subset with high slot completeness. Expand coverage only after the QC gates remain stable.

## Suggested automated tests

- Unit tests for Oya daily aggregation with 48, 30, and fewer than 30 valid slots.
- Sentinel/nodata tests covering NaN, infinity, float32 maximum, values near `1e20`, and negative sentinels.
- Round-trip normalization tests proving zero rain maps to the correct normalized value and missing data remains masked.
- Geospatial alignment tests asserting CRS, transform, bounds, resolution, and mask overlap rather than simple intersection.
- Masked-loss and masked-metric tests with analytically known values.
- Cache-manifest tests that fail on missing, extra, stale, or differently configured files.
- Checkpoint load/resume tests for both model sizes and complete training state.
- CRPS tests against a trusted implementation and deterministic-ensemble special cases.
- End-to-end smoke tests using synthetic local rasters only; live GEE export tests should require an explicit integration marker and credentials.

## Final assessment

| Target | Current status | Recommended decision |
|---|---|---|
| CHIRPS | Promising prototype with moderate regional skill, but evaluation and masking are not yet rigorous | Repair masks/evaluation, establish baselines, then retrain a deterministic model before further GAN tuning |
| Oya | Data-contaminated and effectively unskilled; current metrics are invalid as scientific evidence | Stop tuning, repair/re-export and QC the target, rebuild every derived artifact, then restart from baselines |

The project has enough infrastructure and CHIRPS signal to justify continued work. The main risk is currently experimental validity rather than model capacity. A clean, masked, reproducible deterministic baseline is the shortest path to trustworthy improvements for both ground truths.

## Implementation status

The recommended clean-data path is implemented under the `v2_clean` namespace.
It includes unified Oya aggregation with slot counts, versioned geospatial
preprocessing and QC, explicit masks, manifest-checked fast caches, a
deterministic multiscale residual U-Net, masked training objectives, fixed
validation-only search, three-seed evaluation, baselines, and the global R²
acceptance gate. Operational commands and safeguards are documented in
`docs/pipeline.md`.

The implementation has passed local unit tests and a one-record CHIRPS
end-to-end smoke test, including a finite forward/backward pass. The repository
was subsequently reduced to this maintained workflow; legacy GAN, repair,
scratch, and duplicate preprocessing code was removed. External checkpoints
remain untouched as historical baselines. The full Oya re-export, full dataset
rebuild, training search, and final held-out evaluation remain long-running
execution steps; no R² ≥ 0.40 result is claimed until those steps complete.
