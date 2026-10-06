# Stage 3 — Validation gates (raw and processed)

## Purpose

Two independent, read-only gates that audit — but never mutate — the
outputs of the previous two stages. Mirrors the legacy pipeline's
two-gate pattern (`docs/stages/02-validate-raw.md` before build,
`docs/stages/04-validate-processed.md` after it), consolidated into one
module here since both gates share the same result shape and CLI.
Implementation: `src/data_preprocessing/validate_hourly_dataset.py`.

```bash
python -m src.data_preprocessing.validate_hourly_dataset --region mexico_pacifico --stage raw
python -m src.data_preprocessing.validate_hourly_dataset --region mexico_pacifico --stage raw --full-hash-check
python -m src.data_preprocessing.validate_hourly_dataset --region mexico_pacifico --stage processed
```

Both print a JSON report (`{region, stage, accepted, counts, errors}`) to
stdout, optionally also to `--output <path>`, and exit non-zero when
`accepted` is `false` — wired as a blocking gate in
`src/data_preprocessing/pipeline_hourly.py`, and safe to run standalone at
any time as an integrity audit (neither gate writes or deletes anything).

## `--stage raw`: acquisition integrity

Reads both `manifest_era5_ee_<region>.json` and `manifest_imerg_<region>.json`
([Acquisition](01-acquire.md)) and, for every record:

1. Confirms the file exists on disk at the manifest's recorded path
   (`missing_file` if not).
2. With `--full-hash-check` (off by default — recomputing SHA-256 over
   every file is slow at 5,113 files/region/dataset; existence-only is the
   fast default path, matching this project's existing "fast vs. full"
   validation-depth convention), recomputes the file's SHA-256 and compares
   it to the manifest's recorded value (`hash_mismatch` if not; this is a
   corruption/tamper signal, not something normal operation produces).
3. Collects the set of days present in each manifest and reports any day
   present in one dataset but not the other
   (`era5_without_matching_imerg` / `imerg_without_matching_era5`, with a
   count and a 5-item sample) — a day missing from only one side cannot be
   built at all (`build_day`'s `missing_source_file` rejection), so this
   catches that gap *before* spending a build run's time discovering it one
   rejected day at a time.

Run against the completed acquisition (2026-09-29, all three regions,
2010–2023): `accepted: true`, `{"era5_ee": 5113, "imerg": 5113}` for every
region, zero asymmetric-coverage days.

## `--stage processed`: build integrity

Reads `dataset_index_hourly_<region>.csv` ([Build / alignment](02-build.md))
and, for every row marked `accepted`:

1. Confirms the NPZ exists (`missing_file`).
2. Confirms every expected key is present
   (`inputs`, `target`, `input_valid_mask`, `target_valid_mask`, `land_mask`,
   `elevation`, `basin_mask`, `preprocessing_version` — `missing_keys`).
3. Confirms the hour dimension is exactly 24 on both `inputs` and `target`
   (`unexpected_hour_dimension`).
4. Confirms no sentinel/non-finite value hides inside a pixel its own mask
   calls valid, for both `target` (against `target_valid_mask`) and
   `inputs` (against `input_valid_mask`, broadcast across the channel
   axis) — `empty_valid_mask` / `nonfinite_valid_values` /
   `sentinel_valid_values` (reusing
   `src.data_preprocessing.quality.FLOAT_FILL_THRESHOLD`, the same `1e10`
   threshold the legacy pipeline uses). A file that fails to load at all
   is reported as `unreadable` rather than raising.
5. Recomputes per-split counts and compares them against the index's own
   expected counts per split (`index_artifact_count_mismatch`).
6. Recomputes the manifest's content hash **from the manifest's own
   embedded `records` field** (not from a fresh re-read of the CSV) and
   compares it to the manifest's own stored `manifest_sha256`
   (`manifest_hash_mismatch`). This check is deliberately *not* "re-read
   the CSV and hash that": `csv.DictReader` stringifies every field (a
   Python `True` becomes the string `"True"`), so hashing CSV-re-read
   records against a hash computed at build time over the original
   Python-typed records would mismatch on every single run, tampering or
   not — found and fixed during this stage's own test-writing (a bug in
   the validator itself, not in the build stage). Hashing the manifest's
   own embedded records against its own stored hash instead checks
   internal self-consistency: it catches a manifest.json whose
   `manifest_sha256` field (or a subset of its `records`) was hand-edited
   or corrupted independently of the other.

## Tests

`tests/test_validate_hourly_dataset.py` — 12 tests. `raw`-stage tests use
hand-built manifests (no real acquisition data): matching/complete
manifests pass, a missing manifest or missing file is caught, asymmetric
era5/imerg day coverage is flagged, and `--full-hash-check` catches a
tampered SHA-256 that the fast existence-only path misses.
`processed`-stage tests reuse `tests/test_build_hourly_dataset.py`'s
synthetic small region to run a real `build_region()` and then deliberately
break its output one way at a time — delete an NPZ, corrupt an NPZ's bytes,
inject a `NaN` into a pixel its own valid mask calls true, hand-edit the
manifest's hash field — confirming each is individually detected, plus the
"everything is fine" baseline case and a missing-index case. Also run
against the real completed acquisition (`--stage raw`,
`mexico_pacifico`): accepted, 5,113/5,113.

## Connection to the next stage

Stage 4 (Prepare) requires `validate_processed(region)["accepted"]`
before computing normalization statistics — enforced automatically when
run through `src/data_preprocessing/pipeline_hourly.py`
(`docs/stages_v2/README.md`'s "Orchestration and provenance"); still only
a manual precondition when a stage's own CLI is run standalone.
