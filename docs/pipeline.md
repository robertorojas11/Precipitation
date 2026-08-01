# Pipeline operations

Run commands from the repository root with the environment variables in
`env.example` configured.

## Automated orchestrator

`pipeline.py` is the primary entry point. It executes each maintained module
as an isolated child process, stops at the first failed quality gate by default,
and records every command, output line, return code, duration, and artifact.

Preview the complete workflow without changing data:

```bash
./venv/bin/python pipeline.py --target both --stages all --dry-run
```

Run the complete workflow:

```bash
./venv/bin/python pipeline.py \
  --target both \
  --stages all \
  --start-year 2004 \
  --end-year 2025 \
  --device cuda \
  --batch-size 4 \
  --num-workers 8
```

Run selected stages for one target:

```bash
./venv/bin/python pipeline.py \
  --target chirps \
  --stages build validate_processed prepare validate_prepared
```

Run a contiguous range:

```bash
./venv/bin/python pipeline.py \
  --target oya \
  --from-stage acquire \
  --to-stage validate_prepared
```

Available stages, in dependency order:

1. `storage_check`
2. `acquire`
3. `validate_raw`
4. `build`
5. `validate_processed`
6. `prepare`
7. `validate_prepared`
8. `search`
9. `candidates`
10. `train_final`
11. `evaluate_validation`
12. `evaluate_test`
13. `report`

The storage check is mandatory even when it is omitted from `--stages`. Before
the first child process starts, it probes every distinct configured external
directory using a 1 MiB write, `fsync`, atomic rename, checksum-verified read,
and cleanup. It also checks mount availability and free capacity. The default
minimum is 5 GiB and can be changed with `--minimum-free-gib`. A failed probe
stops the pipeline before acquisition, preprocessing, training, or evaluation.

CHIRPS acquisition is intentionally marked not applicable because the current
source index points to the existing CHIRPS and shared ERA5 archive. Oya's
`acquire` stage performs the corrected slot-aware re-export. Both targets are
still checked by `validate_raw`.

Resume is enabled by default: completed stages with their expected artifact are
skipped. Use `--no-resume` only when intentionally rebuilding a stage. Oya
acquisition is internally resumable by local file presence. Use
`--continue-on-error` only for diagnostics; model training should normally
stop at a failed data gate.

Each invocation writes:

```text
logs/pipeline/<run-id>/
├── options.json
├── events.jsonl
├── pipeline.log
├── chirps/
│   └── <stage>.log
└── oya/
    └── <stage>.log
```

`events.jsonl` is machine-readable and includes target, stage, state,
timestamp, duration, return code, and expected artifact. Child stdout and
stderr are combined in the corresponding stage log and streamed live to the
master log.

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

## Output report

The `report` stage uses the frozen three-seed ensemble and test masks to write:

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
    └── selected_days/
```

Selected-day panels compare observations, ERA5, the model ensemble, and signed
error on a shared valid mask. Relative bias excludes cells whose observed
climatology is below 0.1 mm/day.

## Acceptance criteria

- Pooled test R² >= 0.40 for each target.
- Every test-year R² >= 0.20.
- Month-bootstrap R² lower 95% bound >= 0.35.
- Better R² than ERA5 and monthly climatology.
- No sentinel or non-finite value in any valid target pixel.

A failed target remains a failed result. Do not change the test mask, clip
observations, tune against test years, or use post-processing to claim success.
