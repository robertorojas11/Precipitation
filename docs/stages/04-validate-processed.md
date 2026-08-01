# Stage 4 — Validate processed samples

## Purpose

Processed validation proves that the build output is internally consistent
before statistics are calculated. This prevents corrupt, stale, incomplete, or
smoke-test artifacts from influencing normalization and model selection.

## Checks

The validator reads
`$LOCAL_DATA_DIR/v2_clean/metadata/dataset_index_<target>.csv`, selects only
`accepted=True`, and opens every referenced NPZ. It requires:

- `target`, `target_valid_mask`, `input_valid_mask`, `land_mask`, and
  `preprocessing_version`;
- either embedded `inputs`, `upslope`, `spectral`, and `elevation`, or a
  valid existing `feature_source_npz`;
- at least one target-valid pixel;
- finite valid target values;
- no valid value whose absolute magnitude reaches (10^{10});
- one artifact for every accepted index row;
- nonempty train, validation, and test splits.

Invalid filled pixels may contain zero because their masks exclude them. The
validator intentionally inspects `target[mask]`, not the full storage array.

## Command and result

```bash
python -m src.data_preprocessing.validate_dataset \
  --target chirps --stage processed \
  --output outputs/v2_clean/chirps/validation/processed.json
```

The JSON contains split counts and structured errors. Exit status 0 requires
`accepted=true`; all other states stop the orchestrator.

## Connection to preparation

Only a successful result authorizes train-only statistics and prepared caches.
Because statistics summarize millions of pixels, allowing one bad valid value
through this boundary could contaminate every subsequent sample. Preparation
therefore depends on verified processed artifacts rather than raw files.
