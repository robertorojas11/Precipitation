# Stage 5 — Statistics, tensors, and climatology

## Purpose

Preparation transforms validated processed samples into efficient model-ready
tensors. It performs three ordered sub-stages: train-only statistics, prepared
cache generation, and train-only monthly climatology.

## Train-only normalization

Statistics use only rows whose split is `train`. ERA5 precipitation and target
precipitation use

\[
x'=\log(1+\max(x,0)).
\]

For each feature channel (c), valid training pixels produce

\[
\mu_c=\frac{1}{N_c}\sum_i x_{ic},\qquad
\sigma_c=\sqrt{\frac{1}{N_c}\sum_i x_{ic}^2-\mu_c^2}.
\]

The target uses the same equations on log precipitation. Standard deviations
below (10^{-8}) become 1 to avoid division by zero. Validation and test pixels
never affect these values, preventing leakage.

The normalized value is

\[
z=(x'-\mu)/\sigma,
\]

and physical precipitation is recovered with

\[
x=\max(0,\exp(z\sigma+\mu)-1).
\]

## Prepared tensors

Each date is stored locally under
`data/v2_clean/<target>/<split>/<date>.npz`:

- `inputs_25km`: 18 normalized atmospheric channels at 92 × 144;
- `input_valid_mask_25km`;
- `phys_dem_10km`: upslope, spectral response, and elevation at 230 × 360;
- `real_10km` and its validity mask;
- `real_5km`: normalized target at 460 × 720;
- target and land masks at 5 km;
- `era5_precip_5km_norm`: interpolated ERA5 precipitation baseline;
- seasonal sine/cosine features;
- date.

Atmospheric and physical continuous fields use bilinear interpolation.
Validity masks use nearest-neighbor interpolation. Target downsampling is
masked area averaging:

\[
\bar y = \frac{\operatorname{area}(yM)}
{\max(\operatorname{area}(M),10^{-6})},
\]

and a coarse cell is valid only when averaged mask coverage is at least 0.999.

Season is encoded continuously for day-of-year (d):

\[
\phi=2\pi(d-1)/365.2425,\qquad
s=(\sin\phi,\cos\phi).
\]

## Monthly climatology

For each calendar month and grid cell, the climatology is the arithmetic mean
of physical target precipitation over valid training dates only. The NPZ stores
both `precipitation_mm[12,460,720]` and per-cell valid counts. It becomes a
fixed evaluation baseline.

## Outputs and next contract

`norm_stats_<target>.json`, `fast_manifest_<target>.json`, and
`monthly_climatology_<target>.npz` are written under versioned metadata. The
manifest includes exact split counts and a deterministic record hash.

```bash
python -m src.data_preprocessing.prepare_dataset --target chirps --stage all
```

Prepared validation must confirm every file and split before training begins.
