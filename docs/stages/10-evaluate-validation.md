# Stage 10 — Frozen validation evaluation

## Purpose

This stage evaluates the three-seed ensemble on the validation split after
training. It checks ensemble loading, preprocessing inversion, baselines, event
metrics, temporal stratification, and acceptance logic before the held-out test
set is opened.

## Physical reconstruction and masks

Normalized targets and ERA5 baselines are converted back to mm/day with

\[
P=\max(0,\exp(z\sigma+\mu)-1).
\]

The evaluation mask is exactly

\[
M=M_{target,5km}\land M_{land,5km}.
\]

No observation is capped or filled into the valid set.

## Continuous metrics

Across pooled valid pixels, with observations (y_i), predictions
(\hat y_i), and count (N):

\[
MAE=\frac1N\sum_i|y_i-\hat y_i|,
\]

\[
RMSE=\sqrt{\frac1N\sum_i(y_i-\hat y_i)^2},
\]

\[
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}
{\sum_i(y_i-\bar y)^2}.
\]

The implementation accumulates sums in float64 without retaining all pixels.

## Event metrics

For thresholds 1, 10, and 25 mm/day, true positives (TP), false positives
(FP), and false negatives (FN) define

\[
CSI=\frac{TP}{TP+FP+FN},\quad
POD=\frac{TP}{TP+FN},
\]

\[
FAR=\frac{FP}{TP+FP},\quad
FBias=\frac{TP+FP}{TP+FN}.
\]

## Baselines and uncertainty

The ensemble is compared with interpolated ERA5, train-only monthly
climatology, and a zero/dry forecast. Metrics are also accumulated by calendar
year. R² uncertainty uses a month-block bootstrap: year-month accumulators are
sampled with replacement 2,000 times using seed 42, preserving blocks rather
than pretending pixels are independent.

The result is `metrics_val.json`. It is diagnostic; the formal R² ≥ 0.40 claim
belongs to the untouched test split.
