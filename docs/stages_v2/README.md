# v2-generative pipeline stage documentation

This directory documents the executable workflow for the **v2-generative**
framework (`docs/decisions/0002-implementacion-v2.md`) — the hourly
ERA5→IMERG cascade over `mexico_pacifico`, `mexico_atlantico_golfo`, and
`puerto_rico_caribe`. It is the direct counterpart of
`docs/stages/README.md`, which documents the **archived** deterministic
daily CHIRPS/Oya pipeline (`legacy/deterministic_unet/`) — that document is
not superseded, the two pipelines are independent and coexist.

Read in numerical order. A stage is documented in full only once it is
implemented; earlier stages are complete, later ones are listed here as
**PLANNED** so the full roadmap is visible even before each piece exists
(project rule: don't leave a gap undocumented, mark it explicitly instead).

| Stage | Document | Status | Primary result |
|---:|---|---|---|
| 0 | [Storage preflight](00-storage-check.md) | DONE (reused unchanged) | Proof that configured storage is mounted, writable, readable, and has enough capacity |
| 1 | [Acquisition](01-acquire.md) | DONE | Per-region, per-day ERA5 (cp+lsp) and IMERG GeoTIFFs via Google Earth Engine |
| 2 | [Build / alignment](02-build.md) | DONE | Aligned, masked, versioned hourly NPZ samples per region/day |
| 3 | [Validation gates](03-validate.md) | DONE | Read-only raw/processed integrity audits; acceptance gate before normalization |
| 4 | [Prepare (normalization)](04-prepare.md) | DONE | Train-only pooled normalization statistics (RainShift M216-M219), final training tensors, per-region manifest |
| 5 | [Validate prepared](05-validate-prepared.md) | DONE | Final, read-only authorization-to-train gate |
| 6 | [Training](06-train.md) | DONE (infrastructure; no real-scale run yet) | Training loops + checkpoints for the deterministic baseline, Candidate A (GAN), and Candidate B (regression + residual diffusion), with optional η-learning |
| 7 | [Evaluation](07-evaluate.md) | DONE (infrastructure; no real-scale run yet) | CRPS/FSS/percentile-MAE/RMSE-SSIM scores, geographic-generalization (zero-shot Puerto Rico/Caribe) results, cross-model comparison |
| 8 | [Interpretability and calibrated products](08-interpretability.md) | DONE (infrastructure; no real-scale run yet) | Shapley attribution, per-pixel quantile/exceedance maps |

## Shared experimental contract

- Three regions, three independent grids (`src.data_preprocessing.build_hourly_dataset.region_grid`)
  at ERA5's native ~0.25° and IMERG's native ~0.1° resolution — **not** one
  fixed domain like the legacy pipeline's 460×720 grid. Every region has its
  own shape, its own manifest, its own dataset index.
- Training regions: `mexico_pacifico`, `mexico_atlantico_golfo`.
  Zero-shot evaluation region: `puerto_rico_caribe` (never trained on,
  per `src/data_preprocessing/regions.py`'s `role` field) — this is a
  **training-time** decision; the build stage processes all three regions
  identically and leaves the "don't train on this one" choice to stage 6.
- Provisional year split (`build_hourly_dataset.TRAIN_YEARS`/`VAL_YEARS`/`TEST_YEARS`):
  train 2010–2019, validation 2020–2021, test 2022–2023. Documented as OQ-6
  in `docs/decisions/0002-implementacion-v2.md` — a project decision, not a
  value from any paper; revisit once real per-region data-quality coverage
  is known (e.g. if IMERG's provisional/permanent boundary near 2025
  eventually pushes the usable range).
- Invalid observations are represented by masks and never silently
  converted to zero rain, clipped into range, used in losses, or included
  in metrics — same rule as the legacy pipeline
  (`docs/stages/README.md` § "Shared experimental contract").
- CHIRPS is **not** part of this cascade (daily-only, excluded per
  `docs/decisions/0002-implementacion-v2.md` §1.4); the land/sea mask here
  is derived from NASADEM elevation alone (no CHIRPS coverage to
  corroborate it), a known limitation carried over from the legacy
  pipeline's own land-mask caveat.

## Orchestration and provenance

Every stage that touches per-region artifacts writes:

- A CSV index (`dataset_index_hourly_<region>.csv` for the build stage) with
  one row per processed unit, including acceptance/rejection reasons.
- A hashed manifest (`src.utils.provenance.write_manifest`) recording the
  git revision, a deterministic content hash over every record, and
  stage-specific metadata (grid definition, region summary, accepted/
  rejected counts).

`src/data_preprocessing/pipeline_hourly.py` orchestrates stages 1
(validate raw) through 5 (validate prepared) end to end -- validate raw ->
build -> validate processed -> compute stats -> prepare -> validate
prepared -- mirroring the legacy daily pipeline's own `pipeline.py`
(resumable, durably-logged, stage-ordered), with every validation stage a
hard gate that halts the run on `accepted: false` unless
`--continue-on-error` is passed. Resumability reuses the real validation
gates as the "is this stage already done" check, not a separate
heuristic. Stage 6 (training) is deliberately **not** a stage here -- it
is model-specific and explicitly user-triggered per candidate
(`docs/stages_v2/06-train.md`), so the orchestrator's scope stops at
"build a trainable, validated dataset", the same scope PASO 1-4 of
`docs/decisions/0002-implementacion-v2.md` give stages 1-5.

```bash
python -m src.data_preprocessing.pipeline_hourly
```

Before this existed, each stage had to be run by hand via its own
module's CLI, in the right order, with no automatic halt on a failing
gate -- still possible (every module's CLI still works standalone), just
no longer the only way.
