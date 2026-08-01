# Stage 12 — Scientific outputs and visualization

## Purpose

Reporting turns the frozen test ensemble into inspectable spatial products. It
repeats deterministic inference with the same checkpoint and manifest checks as
evaluation, uses only target-valid land pixels, and does not alter metrics or
models.

## Spatial accumulators

For each grid cell (x), over its (N_x) valid test days:

\[
\bar y(x)=\frac{1}{N_x}\sum_t y(x,t),\qquad
\bar{\hat y}(x)=\frac{1}{N_x}\sum_t\hat y(x,t),
\]

\[
Bias(x)=\bar{\hat y}(x)-\bar y(x),
\]

\[
RMSE(x)=\sqrt{\frac{1}{N_x}
\sum_t(\hat y(x,t)-y(x,t))^2},
\]

\[
RelativeBias(x)=100\frac{Bias(x)}{\bar y(x)}.
\]

Relative bias is shown only where (\bar y\ge0.1) mm/day, avoiding unstable
percentages over effectively dry cells. `valid_count` is saved and plotted so
coverage differences remain visible.

## Figures

The stage produces:

- mean absolute bias in mm/day with a symmetric diverging scale;
- relative bias percentage with dry-climatology cells masked;
- per-cell RMSE;
- valid sample count;
- training-loss and validation-R² histories for all three seeds;
- six highest observed-mean precipitation dates by default.

Each selected-day figure contains observed precipitation, interpolated ERA5,
model ensemble, and signed model-minus-observed error. Observation, ERA5, and
model panels share a scale capped for display at their combined valid 99th
percentile; underlying data and metrics are not clipped. Error uses a symmetric
98th-percentile scale.

## Machine-readable output

`spatial_diagnostics_test.npz` contains float32 observation mean, prediction
mean, bias, relative bias, RMSE, and uint32 valid count. These arrays allow
future GIS or publication plotting without rerunning inference.

The directory is:

```text
outputs/v2_clean/<target>/final_report/
├── report.md
├── spatial_diagnostics_test.npz
└── figures/
    ├── mean_bias_mm_day.png
    ├── relative_bias_percent.png
    ├── rmse_map.png
    ├── valid_sample_count.png
    ├── training_history.png
    └── selected_days/*.png
```

`report.md` summarizes target, split, ensemble size, R², RMSE, MAE, and the
R² ≥ 0.40 decision from `metrics_test.json`.

## Limitations

Current maps use model-grid row/column axes rather than political boundaries,
and selected examples prioritize domain-mean wetness. Future additions should
reuse the saved arrays for geographic overlays, seasonal panels, elevation
strata, and neighborhood verification without changing the frozen evaluation.
