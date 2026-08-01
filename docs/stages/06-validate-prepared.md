# Stage 6 — Validate prepared training data

## Purpose

This is the final authorization gate before GPU work. It detects missing
prepared directories, stale one-record smoke caches, schema mistakes, broken
masks, nonfinite valid targets, and cache/index mismatches.

## Required schema

Every prepared NPZ must contain:

- atmospheric inputs and their 25 km mask;
- physical/DEM conditioning at 10 km;
- 10 km target and mask;
- 5 km target and mask;
- 5 km land mask;
- normalized 5 km ERA5 baseline;
- seasonal features.

The validator checks the physical target reconstructed storage array only at
`target_valid_mask_5km`. It requires at least one valid pixel, finite values,
and no magnitude at or above (10^{10}).

## Completeness

For every split (s\in\{train,val,test\}):

\[
N_{prepared,s}=N_{accepted\ index,s}>0.
\]

This explicit positive-count condition prevents the prior failure where a
one-record training smoke cache was considered valid because both its
validation and test expectations were accidentally zero.

The orchestrator also verifies that a reusable preparation manifest has split
counts exactly equal to the current processed index. If an upstream stage runs,
downstream artifacts are not resumed solely by existence.

## Command and connection

```bash
python -m src.data_preprocessing.validate_dataset \
  --target chirps --stage fast \
  --output outputs/v2_clean/chirps/validation/prepared.json
```

Training starts only when the output says `accepted=true`. The next stage uses
training samples for gradient updates and validation samples for selection.
Test samples remain untouched.
