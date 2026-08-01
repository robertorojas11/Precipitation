# Pipeline stage documentation

This directory documents the executable workflow defined by
`pipeline.py`. The documents describe the maintained implementation—not the
removed GAN prototype—and should be read in numerical order.

| Stage | Document | Primary result |
|---:|---|---|
| 0 | [Storage preflight](00-storage-check.md) | Proof that configured storage is mounted, writable, readable, and has enough capacity |
| 1 | [Data acquisition](01-acquire.md) | Slot-aware daily Oya GeoTIFFs; CHIRPS/ERA5 use the indexed archive |
| 2 | [Raw validation](02-validate-raw.md) | Domain-aligned QC decision and split counts |
| 3 | [Dataset build](03-build.md) | Aligned, masked, versioned processed NPZ samples |
| 4 | [Processed validation](04-validate-processed.md) | Verified processed artifacts and index parity |
| 5 | [Preparation](05-prepare.md) | Train-only statistics, tensors, climatology, and manifest |
| 6 | [Prepared validation](06-validate-prepared.md) | Final authorization to train |
| 7 | [Short search](07-search.md) | Ranked validation-only hyperparameter trials |
| 8 | [Full candidates](08-candidates.md) | Fully trained candidate models |
| 9 | [Final training](09-train-final.md) | Three frozen-seed checkpoints |
| 10 | [Validation evaluation](10-evaluate-validation.md) | Ensemble validation metrics and gates |
| 11 | [Test evaluation](11-evaluate-test.md) | Frozen held-out scientific result |
| 12 | [Reporting](12-report.md) | Bias/RMSE maps, examples, histories, arrays, and Markdown report |

## Shared experimental contract

- Targets are independent: CHIRPS and Oya never share normalization statistics,
  checkpoints, or acceptance decisions.
- The active artifact namespace is `v2_clean`.
- The model grid is EPSG:4326, 0.05°, 460 × 720 cells, covering longitude
  −120° to −84° and latitude 12° to 35°.
- Invalid observations are represented by masks and never silently converted to
  zero rain, clipped into range, used in losses, or included in metrics.
- Statistics and climatology use training data only. Hyperparameters use
  validation data only. Test data is opened only after the configuration and
  seeds are frozen.
- The principal success threshold is pooled test R² ≥ 0.40 for each target,
  accompanied by yearly, uncertainty, and baseline gates.

## Orchestration and provenance

Every invocation creates `logs/pipeline/<run-id>/options.json`,
`events.jsonl`, a master log, and one log per target/stage. Resume decisions
are artifact-aware: failed validation files, smoke manifests, mismatched split
counts, and incomplete prepared data cannot authorize later stages. If an
upstream stage executes, selected downstream stages execute again.

Use [the operations guide](../pipeline.md) for commands and this directory for
scientific and technical details.

The shared [data, software, and compute reference](data-and-resources.md)
documents channel order, units, target interpretation, storage layout,
dependencies, resource requirements, provenance, and known limitations.
