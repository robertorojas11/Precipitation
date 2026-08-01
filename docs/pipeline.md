# Pipeline operations

Run commands from the repository root with the environment variables in
`env.example` configured.

## Oya export

Oya exports contain daily precipitation and valid 30-minute slot count. Pixels
with fewer than 30 slots are invalid.

```bash
python -m src.data_extraction.export_oya --start-year 2004 --end-year 2025
python -m src.data_preprocessing.validate_dataset --target oya --stage raw
```

Exports are resumable and stored below
`$RAW_DATA_DIR/v2_clean/oya/`. Existing external checkpoint files and older
raw data are not modified.

## Build and prepare

```bash
python -m src.data_preprocessing.build_dataset --target chirps
python -m src.data_preprocessing.validate_dataset --target chirps --stage processed
python -m src.data_preprocessing.prepare_dataset --target chirps --stage all
python -m src.data_preprocessing.validate_dataset --target chirps --stage fast
```

Repeat these commands with `--target oya`. Do not train until validation
succeeds and prepared split counts match the accepted index.

## Model selection and training

```bash
python -m src.training.search --target chirps --stage search
python -m src.training.search --target chirps --stage candidates
python -m src.training.search --target chirps --stage final
```

Repeat for Oya. Search and candidate selection use validation data only. If a
one-day model misses validation R² 0.40, candidate training includes a
three-day-context model without crossing split boundaries.

## Frozen evaluation

```bash
python -m src.training.evaluate \
  --run-dir outputs/v2_clean/chirps/final_seed17 \
            outputs/v2_clean/chirps/final_seed42 \
            outputs/v2_clean/chirps/final_seed73 \
  --split test
```

The result includes pooled and yearly R², block-bootstrap uncertainty, fixed
baselines, MAE/RMSE, and event metrics. Repeat for Oya.

## Acceptance criteria

- Pooled test R² >= 0.40 for each target.
- Every test-year R² >= 0.20.
- Month-bootstrap R² lower 95% bound >= 0.35.
- Better R² than ERA5 and monthly climatology.
- No sentinel or non-finite value in any valid target pixel.

A failed target remains a failed result. Do not change the test mask, clip
observations, tune against test years, or use post-processing to claim success.
