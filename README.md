# Precipitation downscaling

Deterministic 25 km to 5 km daily precipitation downscaling for two independent
ground truths: CHIRPS and Oya.

The maintained workflow is documented in [docs/pipeline.md](docs/pipeline.md).
Its structure and scientific constraints are documented in
[architecture.md](architecture.md). Historical findings that motivated the
refactor remain in [CODEBASE_RESULTS_REVIEW.md](CODEBASE_RESULTS_REVIEW.md).
Detailed scientific and operational documentation for every executable stage
starts at [docs/stages/README.md](docs/stages/README.md).

## Quick checks

```bash
python -m unittest discover -s tests
python -m src.data_preprocessing.validate_dataset --target chirps --stage fast
```

Preview or run the automated workflow:

```bash
./venv/bin/python pipeline.py --target both --stages all --dry-run
./venv/bin/python pipeline.py --target both --stages all
```

Every non-dry pipeline run performs a mandatory network-storage integrity
preflight before launching its first processing command.

Data and checkpoints under `/mnt/data-r2/RobertoRojas/downscaling/` are
external artifacts and are not deleted by repository maintenance.
