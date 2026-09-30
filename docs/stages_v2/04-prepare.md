# Stage 4 — Prepare (train-only normalization + final training tensors)

## Purpose and inputs

Converts stage 2's aligned-but-raw-physical-units NPZ samples into the
normalized tensors models actually train on: clips ERA5 to IMERG's
observed range, log-transforms precipitation-like channels and orography,
and z-score normalizes everything using statistics computed **once**,
**only from the training regions' training-split days**, then applied
identically to every region — including `puerto_rico_caribe`, which never
contributes a single pixel to computing them. This is RainShift's own
recipe (P2 matrix items M216–M219,
`docs/audit/matriz_trazabilidad_metodologica.md`), reused because
`docs/decisions/0002-implementacion-v2.md` adopts RainShift as the
experimental skeleton. Implementation:
`src/data_preprocessing/prepare_hourly_dataset.py`.

Input: `dataset_index_hourly_<region>.csv` and the per-day NPZs from
[Build / alignment](02-build.md). Recommended precondition (not yet
enforced by an orchestrator): [stage 3's](03-validate.md)
`validate_processed(region)["accepted"]`.

```bash
python -m src.data_preprocessing.prepare_hourly_dataset --stage stats
python -m src.data_preprocessing.prepare_hourly_dataset --stage prepare --region mexico_pacifico
python -m src.data_preprocessing.prepare_hourly_dataset --stage prepare --region puerto_rico_caribe
```

## M216 — clip ERA5 to IMERG's range

"Clip ERA5 precipitation using min/max of IMERG as thresholds" — corrects
ERA5's known tendency to overestimate weak rain and produce unrealistic
extremes relative to the satellite-observed target. `compute_imerg_range()`
scans every **training**-split day across **both training regions**
(`mexico_pacifico`, `mexico_atlantico_golfo` —
`prepare_hourly_dataset.TRAINING_REGIONS`) and takes the global min/max of
valid IMERG pixels. This single, pooled range (not one per region) is the
clip bound applied to both ERA5 channels (cp, lsp) everywhere the framework
runs — training regions and the zero-shot evaluation region alike — so a
model never sees an input contract at inference time that differs from what
it saw in training.

## M217 — log transform

\[
\tilde{x} = \log(\max(x, 0) \cdot 1000 + 10^{-5})
\]

applied to the ERA5 input channels (after clipping), the IMERG target, and
elevation (orography). The land-sea mask is left untransformed in \([0,1]\)
(M218).

**Deviation, documented explicitly, not silent:** M217's formula assumes a
non-negative physical quantity. This project's elevation field spans open
ocean, where NASADEM reads small negative bathymetric noise (observed
minimum: −71 m — see [Build / alignment](02-build.md)). The `max(x, 0)`
clamp already present in the formula for precipitation is applied to
elevation too, for the same reason precipitation gets it: without it, the
log of a negative number would be undefined. RainShift's own study domain
may not have had this issue; this project's does, hence the explicit note.

## M219 — pooled-variance normalization

Per-region log-space moments (`_region_log_moments()`) are computed
independently for each training region, then combined via RainShift's
*specific* pooled formula — not the naive combined-sample variance one
might reach for instead:

\[
\mu_{pooled} = \frac{1}{n}\sum_i \mu_i,
\qquad
\sigma_{pooled}^2 = \frac{1}{n}\sum_i(\sigma_i^2+\mu_i^2) - \mu_{pooled}^2
\]

with \(n\) = number of **regions** (2), not pixels or days — so a region
with more accepted training days does not silently dominate the pooled
statistics the way a naive "concatenate every pixel from every region and
take one mean/std" approach would. `pooled_stats()` implements this exactly
and is unit-tested against a hand-computed example
(`tests/test_prepare_hourly_dataset.py::PooledStatsTest`).

The normalized value is \(z = (\tilde{x}-\mu_{pooled})/\sigma_{pooled}\);
physical units are recovered with
\(x = \max\!\big(0,\ (\exp(z\sigma_{pooled}+\mu_{pooled}) - 10^{-5})/1000\big)\)
(`inverse_log_transform`).

## Prepared tensors

One NPZ per `(region, day)`, at the same per-region canonical grid shape as
the build stage (no further resampling):

```text
<LOCAL_DATA_DIR>/v2_generative_hourly_prepared/<region>/<split>/<date>.npz
├── inputs_norm         float32 (24, H, W, 2)  -- normalized log(clip(era5)) [cp, lsp]
├── target_norm         float32 (24, H, W, 1)  -- normalized log(imerg)
├── invariants_norm     float32 (H, W, 2)      -- [land_mask (raw 0/1), normalized log(elevation)]
├── input_valid_mask    bool    (24, H, W)
├── target_valid_mask   bool    (24, H, W)
├── basin_mask          bool    (H, W)
├── date, region, split, preprocessing_version
```

`preprocessing_version = "v2_generative_hourly_prepared"` — a third,
distinct namespace from both the build stage's `"v2_generative_hourly"` and
the legacy pipeline's `"v2_clean"`.

## Artifacts and provenance

```text
<LOCAL_DATA_DIR>/v2_generative_hourly_prepared/metadata/
├── norm_stats.json                    -- one shared file, not per-region
│   {training_regions, log_transform, era5_clip_range_imerg_mm_per_hr,
│    pooled: {cp, lsp, target, elevation: {mean, std, count}},
│    per_region: {<region>: {...same per-channel stats...}}}
└── manifest_prepared_<region>.json    -- one per region (train, val, test,
                                           and zero-shot eval regions alike)
```

`norm_stats.json` records **both** the pooled statistics actually used for
normalization **and** each contributing region's own statistics before
pooling — kept for auditability (e.g. to check whether one training region
is systematically wetter/drier than the other, which the pooled numbers
alone would hide).

## Tests

`tests/test_prepare_hourly_dataset.py` — 9 tests. The log transform is
checked against the exact RainShift formula, its negative-input clamp, and
a round-trip; `pooled_stats()` against a hand-computed two-region example
(including the boundary case of two identical regions collapsing to their
shared value, and the empty-input error); `compute_stats()` and
`prepare_region()` against a small two-synthetic-region setup with known
constant per-region values, verifying the clip range, that a value outside
the pooled IMERG range is actually clipped before its log-mean is computed,
that the pooled mean is the simple average of two regions' own means, and a
full normalize -> denormalize round-trip recovering the original physical
IMERG value. A dedicated test builds `compute_stats()` from **one** region
only and then successfully `prepare_region()`s the **other** — the same
zero-shot shape production will use for `puerto_rico_caribe`. Also
smoke-tested against 5 real days from each training region
(`mexico_pacifico`, `mexico_atlantico_golfo`, 2015-06-01..05): clip range
`[0.0, 47.9]` mm/hr, pooled target log-mean ≈ −8.83 (close to the
"no-rain" constant `log(1e-5) ≈ −11.51`, consistent with a heavily
zero-inflated rain distribution — a physically sensible sanity check, not
just a shape check). 174 tests pass total.

## Known limitations

- `compute_stats()` has not yet been run over the *full* historical
  training-split range (2010–2019 × 2 regions, per
  `build_hourly_dataset.TRAIN_YEARS`) — only smoke-tested on 5 real days
  per region. Running it at full scale is a deliberate, explicit next
  action (mirrors how the acquisition stage's full-scale run was a
  separate, user-triggered step, not something done automatically).
- No orchestrator yet enforces stage 3's validation gate before this stage
  runs (same gap noted throughout `docs/stages_v2/`).

## Connection to the next stage

Stage 5 (PLANNED — validate prepared) should confirm every prepared file
exists, has the expected keys, and that `norm_stats.json`'s per-region
counts match the build stage's own accepted-day counts, before stage 6
(training) is authorized to start — mirroring the legacy pipeline's
"prepared validation is the final gate before training begins"
(`docs/stages/06-validate-prepared.md`).
