# Repository guidance

## Scope

Maintain one deterministic precipitation-downscaling workflow for CHIRPS and
Oya. Both targets use the same code and split policy but retain independent
statistics, manifests, runs, and evaluation results.

## Rules

- Run Python entry points as modules from the repository root.
- Preserve explicit source-valid, target-valid, and land masks.
- Calculate normalization statistics from training data only.
- Select models on validation years; do not tune against test years.
- Never silently clip or replace invalid observations.
- Keep dataset and run manifests with hashes and source provenance.
- Add or update local unit tests with behavior changes.
- Treat external checkpoints as immutable historical artifacts unless a run was
  produced by the current architecture and matching dataset manifest.

## Maintained commands

```bash
python -m src.data_preprocessing.build_dataset --target chirps
python -m src.data_preprocessing.prepare_dataset --target chirps --stage all
python -m src.data_preprocessing.validate_dataset --target chirps --stage fast
python -m src.training.search --target chirps --stage search
python -m src.training.evaluate --run-dir RUN_DIRECTORY --split test
```

See `docs/pipeline.md` for the complete workflow and acceptance gates.
