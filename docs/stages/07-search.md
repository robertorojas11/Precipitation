# Stage 7 — Short validation-only search

## Goal

The search identifies promising capacity, regularization, learning-rate, and
heavy-event weighting choices without using test observations. It deliberately
uses a small fixed grid so selection remains auditable and computationally
bounded.

## Search space

The Cartesian product contains 16 trials:

| Parameter | Values |
|---|---|
| AdamW learning rate | (10^{-4}, 3\times10^{-4}) |
| Base feature width | 32, 64 |
| Heavy-event weight | 2, 4 |
| Spatial dropout | 0, 0.1 |

Each trial uses seed 42, one day of atmospheric context, and at most 15 epochs.
Batch size, workers, and device come from the pipeline command. Early stopping
can end a run sooner.

## Model summarized

The deterministic multiscale U-Net encodes 18 atmospheric channels at 25 km,
upsamples to 10 km, and conditions on three physical channels, normalized
latitude/longitude, seasonal sine/cosine, and ERA5 precipitation. It then
upsamples to 5 km and conditions again on coordinates, season, and ERA5.
Residual blocks use 3×3 convolutions, GroupNorm, SiLU, optional Dropout2d, and
residual addition.

There are two output heads:

\[
p_{wet}=\sigma(l),
\]

\[
a_{log}=\operatorname{softplus}
(\log(1+P_{ERA5})+r),
\]

\[
\hat P=p_{wet}[\exp(a_{log})-1].
\]

Thus prediction separates occurrence from positive amount while retaining an
ERA5 residual baseline.

## Ranking and output

Every epoch reports training loss and masked validation R²/RMSE/MAE. The best
checkpoint maximizes pooled validation R². Trials are ranked by their maximum
validation R² and written to
`outputs/v2_clean/<target>/search_results.json`.

```bash
python -m src.training.search --target chirps --stage search
```

Only the best two parameter sets advance to full candidate training.
