# CHIRPS v3 training and validation

## Objective and evidence standard

This workflow targets pooled, masked, pixel-day R² in physical millimetres greater than 0.80. CHIRPS is used only as the supervised target; it is never supplied as a predictor. The target is a research objective, not a promised outcome. The `feasibility` stage measures target-derived 10 km and 25 km oracle ceilings first. If the 25 km oracle is below 0.80, more architecture search cannot by itself establish that the available coarse information supports the goal.

## Data and leakage controls

The workflow reads the immutable `data/v2_clean/chirps` tensors and the corresponding source normalization metadata. It does not rewrite v2. A sample contains 18 coarse atmospheric fields, three 10 km physical fields, ERA5 precipitation, masks, season encodings, and the CHIRPS target. Targets and ERA5 are converted back to physical millimetres before loss or metrics. Atmospheric fields are re-standardized using statistics computed only from each fold's training years; training refuses to start if these artifacts are missing.

Centered contexts of 1, 3, 5, or 7 days are supported for retrospective reanalysis. Context is never allowed to cross a training/validation boundary. Five expanding-window folds validate 2016–2017, 2018–2019, 2020–2021, 2022–2023, and 2024–2025. The final 2026 holdout stays locked until an index proves at least 365 accepted, unique dates.

Terrain features are derived deterministically from elevation: elevation, slope, sine and cosine of aspect, curvature, and local relief at two spatial scales. These are static predictors, not target leakage.

## Model and optimization

Every context day passes through the same atmospheric encoder. Learned spatial attention (or a mean ablation) fuses time. Atmospheric features join the physical 10 km fields and seasonal sine/cosine, are upsampled, and join the 5 km terrain encoder. The decoder predicts a log-space residual over ERA5 plus a rain-occurrence probability. Their product gives non-negative precipitation.

For valid land pixels, the principal loss is

`MSE(prediction_mm, observation_mm) / Var(observation_mm)`.

Minimizing this term directly maximizes pooled R² because `R² = 1 - SSE/SST`. Smaller log-Huber, occurrence cross-entropy, and spatial-gradient losses stabilize dry days, extremes, and spatial structure. AdamW, cosine decay, mixed precision, gradient clipping, accumulation, early stopping, and exponential moving-average weights are used. Validation and saved best checkpoints use EMA weights.

## Stages and commands

Always prove the network disk is usable first:

```bash
./venv/bin/python pipeline_v3.py --stages storage_check
```

Create the experiment contract, compute training-only normalization for every fold, run information-ceiling diagnostics, and create the 120-job search manifest:

```bash
./venv/bin/python pipeline_v3.py --stages contract prepare_folds feasibility search_manifest --budget 120
```

Run phases separately so results can be reviewed between compute commitments:

```bash
./venv/bin/python pipeline_v3.py --stages train_search --phase representation --budget 120
./venv/bin/python pipeline_v3.py --stages train_search --phase architecture --budget 120
./venv/bin/python pipeline_v3.py --stages train_search --phase optimization --budget 120
./venv/bin/python pipeline_v3.py --stages train_search --phase rolling --budget 120
```

The screening phases compare context/fusion, capacity/dropout, and optimizer settings on the earliest and latest folds. Rolling confirmation must cover all five folds. Selection uses `mean R² - 0.25 × standard deviation`, discouraging a model that succeeds only in favorable years.

Freeze only after one identical configuration has completed all folds:

```bash
./venv/bin/python pipeline_v3.py --stages freeze
```

After 2026 is complete and acquired by the data pipeline, unlock it with:

```bash
./venv/bin/python pipeline_v3.py --stages verify_2026 --holdout-index /path/to/2026_index.csv
```

Every stage streams child output to UTC timestamped logs under `outputs/v3_chirps/chirps/logs`, records success/failure and duration in `pipeline_state.json`, fails immediately on errors, and supports artifact-aware resume. Evaluation of a checkpoint is available with `python -m src.chirps_v3.evaluate --checkpoint PATH`.

Stages resolve their prerequisites automatically. In particular, requesting `train_search` runs a fresh network-storage integrity probe and creates any missing contract, fold-statistics, feasibility, and search-manifest artifacts before launching training. Therefore the training command is safe to use directly; completed expensive prerequisites are skipped, but storage is checked again on every invocation.
