# `legacy/` — archived pre-migration code

This directory holds code that was **archived, not deleted**, during the
migration to the v2-generative framework
(`docs/decisions/0002-implementacion-v2.md`, `docs/MIGRATION.md`). It was
first archived on the same terms during the reverted attempt #1
(2026-09-01) and re-archived on `master` on 2026-09-11 once attempt #2
started, per the same user decisions (OQ-1).

Nothing here is imported by the active v3 code. Everything here is still
runnable. The pre-migration state is also captured as the git tag
**`pre-migration-v2_clean`**.

---

## `legacy/deterministic_unet/` — deterministic daily CHIRPS/Oya downscaling

The original `v2_clean` pipeline: a deterministic two-head (occurrence × amount)
multiscale U-Net that downscales 25 km → 5 km **daily** precipitation for the
CHIRPS and Oya ground truths, trained with a composite non-adversarial loss and
evaluated with R²/RMSE/MAE + block bootstrap.

**Why archived.** The audit
(`docs/audit/auditoria_codigo_vs_papers.md`, §1.1 and §22) found this is a
different research problem from either reference paper: it is deterministic, not
probabilistic; daily, not hourly/sub-hourly; single-region; and its metrics
(R²/RMSE/MAE) are not the papers' CRPS/FSS/RAPSD. `architecture.md` itself
declares "This project does not claim ... a probabilistic model." The v3 line
keeps the *engineering* (masks, train-only stats, hashed manifests) and drops
the *deterministic model* as the headline result — but the model stays here,
runnable, as a deterministic baseline for future comparison (P2 compares a
deterministic ResNet against generative models; this U-Net can play that role
for the daily task). The active v2-generative framework does **not** reuse
this module directly — per `docs/decisions/0002-implementacion-v2.md` §5.4 it
gets a fresh `src/models/deterministic_resnet.py` baseline adapted to the new
hourly ERA5→IMERG data contract, optionally evaluated with the η-learning
regularizer (§1.2) as a third comparison point alongside the two generative
candidates.

Files (moved verbatim, only import paths rewritten
`src.models.multiscale_unet` → `legacy.deterministic_unet.multiscale_unet`,
`src.data_preprocessing.dataset` → `legacy.deterministic_unet.dataset`,
`src.training.train` → `legacy.deterministic_unet.train`):

| File | Role |
|---|---|
| `multiscale_unet.py` | `MultiscalePrecipUNet` model |
| `dataset.py` | `PrecipitationDataset` (reads prepared `data/v2_clean/<target>` tensors) |
| `train.py` | masked training loop; `compute_losses()` (BCE + SmoothL1 + physical + gradient + multiscale) |
| `search.py` | validation-only hyperparameter search / candidates / final |
| `evaluate.py` | held-out evaluation, baselines, yearly + block-bootstrap scores |
| `report.py` | bias/RMSE maps, selected-day panels, Markdown report |

**Data it still needs** (unchanged, still in `src/data_preprocessing/`):
`build_dataset.py`, `prepare_dataset.py`, `validate_dataset.py`,
`quality.py`, `physics.py`.

**How to keep training / evaluating it**

```bash
# build + prepare (unchanged)
python -m src.data_preprocessing.build_dataset   --target chirps
python -m src.data_preprocessing.prepare_dataset --target chirps --stage all
python -m src.data_preprocessing.validate_dataset --target chirps --stage fast

# train / search / evaluate (archived entry points)
python -m legacy.deterministic_unet.search   --target chirps --stage search
python -m legacy.deterministic_unet.search   --target chirps --stage final
python -m legacy.deterministic_unet.evaluate --run-dir outputs/legacy_v2_clean_deterministic/chirps/final_seed42 --split test
python -m legacy.deterministic_unet.report   --run-dir outputs/legacy_v2_clean_deterministic/chirps/final_seed42 --split test
```

Its historical checkpoints, run histories and reports were **moved** (not
deleted) from `outputs/v2_clean/` to
`outputs/legacy_v2_clean_deterministic/` with structure intact. `run-dir`
arguments must point there.

---

## `legacy/chirps_v3/` + `legacy/pipeline_v3.py` — walk-forward CHIRPS study

An in-progress experimental line (`TemporalTerrainNet`, expanding-window folds
`fold_2016_2017` … `fold_2024_2025`, feasibility/oracle diagnostics, EMA
weights, R²-aligned loss). It was **not documented as active** in
`architecture.md`; the audit marked its status "no determinable". User
decision, made in attempt #1 (2026-09-01) and re-confirmed for attempt #2
(2026-09-11): **archive it**
(`docs/decisions/0002-implementacion-v2.md` OQ-1). Its fold scheme and
feasibility ceilings are not carried into the v2-generative framework, but
the code is preserved for reference.

Import paths rewritten `src.chirps_v3` → `legacy.chirps_v3`. Its outputs remain
under `outputs/v3_chirps/`. Docs: `docs/CHIRPS_V3_TRAINING.md` (kept, with a
migration banner).

```bash
python -m legacy.pipeline_v3 --stages storage_check
python -m legacy.pipeline_v3 --stages contract prepare_folds feasibility search_manifest --budget 120
```

---

## Reverting the migration

See `docs/MIGRATION.md` § "How to revert". In short: `git checkout
pre-migration-v2_clean` restores every source file to its pre-migration path;
`outputs/legacy_v2_clean_deterministic/` can be moved back to `outputs/v2_clean/`.
