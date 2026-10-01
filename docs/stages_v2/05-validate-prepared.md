# Stage 5 — Validate prepared (final authorization to train)

## Purpose

The final read-only gate before any training loop runs, mirroring the
legacy pipeline's own last-gate role (`docs/stages/06-validate-prepared.md`:
"the final authorization gate before GPU work"). Added as a third stage
function, `validate_prepared()`, alongside stage 3's `validate_raw()` and
`validate_processed()` in the same module —
`src/data_preprocessing/validate_hourly_dataset.py` — since all three share
the same `{region, stage, accepted, counts, errors}` result shape and CLI.

Input: `manifest_prepared_<region>.json` and `norm_stats.json`
([Prepare](04-prepare.md)), cross-checked against
`dataset_index_hourly_<region>.csv` ([Build / alignment](02-build.md)).
Recommended precondition (not yet enforced by an orchestrator, same gap
noted throughout `docs/stages_v2/`): stage 4 has actually run for the
region being validated.

```bash
python -m src.data_preprocessing.validate_hourly_dataset --region mexico_pacifico --stage prepared
```

Prints the same JSON report shape as the other two gates
(`{region, stage, accepted, counts, errors}`), optionally also to
`--output <path>`, exits non-zero when `accepted` is `false`. Nothing in
this stage mutates or deletes anything.

## What it checks

1. **`norm_stats.json` exists** at
   `<prepared_root>/metadata/norm_stats.json` (`missing_norm_stats` if not).
   This file is shared across all regions (not per-region, per
   [Prepare](04-prepare.md)'s artifact layout), so its absence fails every
   region's validation identically — correctly reflecting that no region
   can have been meaningfully prepared without it.
2. **`manifest_prepared_<region>.json` exists and loads**
   (`missing_manifest` — short-circuits immediately, since every other
   check depends on the manifest's record list).
3. **Per record** (one row per prepared NPZ, each tagged with its split):
   - File exists on disk (`missing_file`).
   - All of `PREPARED_KEYS` are present — `inputs_norm`, `target_norm`,
     `invariants_norm`, `input_valid_mask`, `target_valid_mask`,
     `basin_mask`, `preprocessing_version` (`missing_keys`).
   - `target_norm`'s valid pixels (per `target_valid_mask`) are non-empty
     (`empty_valid_mask`), finite (`nonfinite_valid_values`), and below the
     shared `FLOAT_FILL_THRESHOLD` sentinel magnitude of `1e10`
     (`sentinel_valid_values`) — the same three checks stage 3's
     `validate_processed()` runs on the *pre*-normalization tensors,
     repeated here because normalization (log transform, z-score) is itself
     capable of introducing a fresh `NaN`/`inf` that didn't exist in the
     processed input (e.g. a future change to the clip range or stats could
     produce `log(0)` somewhere the processed-stage check would never see).
   - A file that fails to load at all is reported as `unreadable` rather
     than raising, matching stage 3's convention.
4. **Per-split completeness, exact, not just non-empty**: the count of
   records that passed every per-file check above, grouped by split, is
   compared against the **build** stage's own accepted-row count for that
   same split (read fresh from `dataset_index_hourly_<region>.csv`, not
   cached) — `prepared_count_mismatch` if they differ. This is the
   legacy pipeline's `N_{prepared,s} = N_{accepted\ index,s} > 0` rule
   (`docs/stages/06-validate-prepared.md`), with one deliberate
   adjustment: the legacy version also requires the count to be strictly
   positive, which this version does not, because this framework runs each
   region through the full build stage identically regardless of training
   role, and a region/split combination can legitimately have zero accepted
   days if that date range was simply never built. The invariant this
   stage actually needs — and enforces — is *exactness* against whatever
   the build stage produced, not an unconditional "greater than zero":
   catching a **stale partial prepare run** (fewer prepared files than the
   build stage accepted, e.g. a prepare process that crashed partway
   through) is the point of the check, not ruling out intentionally-empty
   splits.

## A found-and-fixed bug: `compute_stats()` writing to the real path during tests

While writing this stage's own tests, two of them failed in a way that
pointed at the *producer* (stage 4), not the validator:

```text
FAIL: test_freshly_prepared_region_passes
AssertionError: False is not true : [{'error': 'missing_norm_stats', ...}]

ERROR: test_missing_norm_stats_is_detected
FileNotFoundError: [Errno 2] No such file or directory: '.../norm_stats.json'
```

Root cause: `prepare_hourly_dataset.compute_stats()` had no `output_root`
override parameter — every call, test or real, wrote `norm_stats.json` to
the hardcoded real path (`_prepared_root()`, i.e.
`Config.LOCAL_DATA_DIR / "v2_generative_hourly_prepared"`), unlike
`prepare_region()` in the same module, which already respected an
`output_root` override. Every test that called `compute_stats()` inside an
isolated `tempfile.TemporaryDirectory()` was therefore silently writing a
real file to the real network-mounted storage path instead of its intended
tmp directory — and this stage's `validate_prepared()` tests were the first
to actually notice, because they check for `norm_stats.json`'s existence
*inside the test's own prepared_root*, which never received it.

This also explains a real artifact found on disk during this investigation:
`/mnt/data-r2/.../v2_generative_hourly_prepared/metadata/norm_stats.json`
existed, containing the 5-day smoke-test statistics mentioned in
[Prepare](04-prepare.md)'s "Known limitations" — correct at the time it was
written, but **not** full-historical-scale statistics, and left in place it
could have been mistaken for them by a future real `prepare_region()` run
that doesn't explicitly recompute stats first. It has been deleted.

Fix: added `output_root: Path | None = None` to `compute_stats()`'s
signature, with `root = Path(output_root or _prepared_root())` replacing
the previous unconditional `root = _prepared_root()` — the same
injectable-root pattern used by every other stage function in this
project. All four test call sites
(`tests/test_validate_hourly_dataset.py`,
`tests/test_prepare_hourly_dataset.py` ×3) were updated to pass
`output_root=` explicitly, so no test in this suite touches the real
storage path again.

## Tests

`tests/test_validate_hourly_dataset.py::ValidatePreparedTest` — 5 tests,
reusing `tests/test_build_hourly_dataset.py`'s synthetic small-region
fixture to run a real `build_region()` → `compute_stats()` →
`prepare_region()` chain (2 real synthetic days) and then deliberately
break the output one way at a time: delete a prepared NPZ, truncate the
manifest's own record list to simulate a stale partial prepare run (an
independent check from the missing-file case — a record can go missing
from the manifest without any file actually disappearing from disk), and
delete `norm_stats.json` — each confirmed individually detected, plus the
"everything is fine" baseline (`accepted: true`, correct per-split counts)
and a missing-manifest short-circuit case. `ValidateDispatchTest` gained
one more case confirming `--stage prepared` reaches `validate_prepared()`.
180 tests pass total across the full suite.

## Known limitations

- Not yet run against a full-historical-scale prepared dataset — stage 4
  itself has not been run at that scale yet (per
  [Prepare](04-prepare.md)'s own "Known limitations"), so this gate has
  only been exercised against synthetic and small real smoke-test data.
- No orchestrator yet enforces this gate before a training loop starts
  (same gap noted throughout `docs/stages_v2/`); it must be run manually.

## Connection to the next stage

Stage 6 (training) should require `validate_prepared(region)["accepted"]`
for every region it trains on before starting — the same "last gate before
GPU work" role the legacy pipeline's own stage 6 document describes. No
`train.py` exists yet for any of the three candidate models
(deterministic baseline, Candidate A GAN, Candidate B regression+diffusion)
— building one, and deciding whether to first run `compute_stats()` and
`build_region()` at full historical scale, are the next user-directed
steps, not assumed by this stage.
