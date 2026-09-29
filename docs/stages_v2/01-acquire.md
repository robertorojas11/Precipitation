# Stage 1 — Acquisition (ERA5 cp+lsp, IMERG)

## Purpose and inputs

Downloads the two raw inputs the rest of the framework depends on, per
region: pure ERA5 `convective_precipitation` + `large_scale_precipitation`
(hourly), and IMERG V07 precipitation (half-hourly, reduced to hourly).
Both are acquired through **Google Earth Engine**, not the Copernicus CDS
API or direct NASA GES DISC access — a deliberate revision made 2026-09-21
(`docs/decisions/0002-implementacion-v2.md`, "Revisión 2026-09-21") because
the project has Earth Engine + Drive credentials, not CDS/Earthdata ones,
and Earth Engine happens to expose exactly the needed variables directly.
There is no upstream pipeline dependency for this stage; it is the entry
point of the whole framework. Full source-level detail:
`docs/DATA_SOURCES.md`; this document is the stage-level operational
summary, including the incidents found running it at scale for the first
time (2026-09-21 through 2026-09-29).

## Method

**ERA5** (`src/data_extraction/era5_ee.py`): pulls `ECMWF/ERA5/HOURLY`
(pure ERA5, not ERA5-Land), bands `mean_convective_precipitation_rate` /
`mean_large_scale_precipitation_rate` (kg/m²/s). For each day, all 24
hourly images in the collection are stacked into one 48-band image via
`ee.ImageCollection.toBands()` (band order: chronological by hour, `[cp,
lsp]` pair per hour — see [Build / alignment](02-build.md) for exactly how
these bands are parsed back out) and the rate is converted to accumulated
depth for that hour (`rate[kg/m²/s] × 3600 s = mm`), matching what the CDS
API would have returned in metres (after a ×1000 unit conversion) had that
backend been used instead.

**IMERG** (`src/data_extraction/imerg.py`): pulls `NASA/GPM_L3/IMERG_V07`,
band `precipitation` (mm/hr, calibrated Final-Run-equivalent estimate),
half-hourly. The day's 48 half-hourly images are similarly stacked via
`toBands()`.

**Delivery, both**: one Earth Engine batch export-to-Drive task per
`(region, day)`, with a **flat** Drive filename —
`"<era5|imerg>_<region>_<YYYY-MM-DD>.tif"`, no `/` characters — submitted in
batches of 20 (`TASK_BATCH_SIZE`, `export_oya.py`'s `_submit_in_batches`,
reused by both modules) and polled every 60 s until the batch completes.
Completed exports are pulled from Drive via
`src.data_extraction.drive.sync_prefixed()`, scoped to the calling
process's own region (`name contains "<prefix>_<region>_"`), paginated
across the full result set, written to
`$V2_RAW_DATA_DIR/{era5_ee,imerg}/<region>/<year>/<month>/`, and (when
possible — see below) trashed from Drive. A hashed manifest
(`manifest_{era5_ee,imerg}_<region>.json`) is (re)written after each
month's batch.

Resumable by local file presence: re-running `--start-year`/`--end-year`
skips any day whose file already exists on disk.

```bash
python -m src.data_extraction.era5_ee --start-year 2010 --end-year 2023 --region mexico_pacifico
python -m src.data_extraction.imerg   --start-year 2010 --end-year 2023 --region mexico_pacifico
```

## Credentials

Google Earth Engine + Drive, via **personal OAuth**
(`earthengine authenticate`), not the project's service account
(`auth/service_account.json`) — service accounts have no personal Drive
storage quota and cannot receive `Export.image.toDrive` output at all (see
"Operational incidents" below). `earth_engine.py`'s `initialize()` already
tries personal OAuth first and only falls back to the service account, so
no code change was needed once this was understood — only running
`earthengine authenticate --auth_mode=notebook --force` once (the
`gcloud`-based default auth mode isn't usable in this environment; the
`notebook` mode gives a URL + manual code paste instead).

**The OAuth token expires every 7 days** if the Google Cloud OAuth consent
screen is left in "Testing" publish status (the default for a new project)
— this happened mid-run on 2026-09-28 (see below) and needs the same
`earthengine authenticate` command re-run each time it recurs, unless the
consent screen is published, which removes the 7-day limit.

## Artifacts and provenance

```text
$V2_RAW_DATA_DIR/
├── era5_ee/<region>/<year>/<month>/era5_<region>_<date>.tif   (48 bands, mm/hour)
│   └── manifest_era5_ee_<region>.json
└── imerg/<region>/<year>/<month>/imerg_<region>_<date>.tif    (48 bands, mm/hr half-hourly)
    └── manifest_imerg_<region>.json
```

Completed 2026-09-29 for all three regions, 2010–2023: 30,678 files total
(5,113 days × 3 regions × 2 datasets), each manifest recording 5,113
records with per-file SHA-256.

## Operational incidents found running this at scale (all fixed in code)

Documented in full because the fixes are non-obvious and the failure modes
would otherwise recur silently or be re-debugged from scratch:

1. **"Service accounts do not have storage quota."** Exporting via the
   service account fails outright; must authenticate with a personal
   Google account instead (see "Credentials" above).
2. **Export identity ≠ sync identity.** Once exporting via personal OAuth,
   the file lands in the *user's* Drive, but `drive.py`'s sync step
   authenticates as the *service account* (`GOOGLE_DRIVE_CREDENTIALS`) —
   two different identities. Downloading (read) works once the export
   folder is shared with the service account, but trashing (write) still
   fails with HTTP 403 unless that share is explicitly granted **Editor**,
   not just Viewer. Fixed: `drive._try_trash` catches the failure and logs
   a warning instead of raising; an untrashed export is a Drive-storage
   nuisance, never a data-loss risk (`src/data_extraction/drive.py`).
3. **Unanchored Drive query collisions.** `name contains "era5_"` also
   matched an unrelated legacy export (`era5_pl_2025-*.tif`) sharing the
   same Drive folder. Fixed: `drive.parse_prefixed_export_name` validates
   the parsed region against `src.data_preprocessing.regions.list_regions()`
   and rejects anything that doesn't genuinely belong to this pipeline;
   `sync_prefixed()` additionally scopes its query to
   `"<prefix>_<region>_"` once a region is known, eliminating the 3×
   redundant cross-region listing this also caused when three regions ran
   in parallel.
4. **No pagination.** `files().list()` only returns one page (Drive v3
   default: 100 results); without following `nextPageToken`, once more than
   ~100 files matched a query, only a subset was ever visited. Fixed:
   `drive._list_all()`.
5. **Silent indefinite hang, not a crash.** Two acquisition processes
   stopped producing any log output or progress for 3+ days, still listed
   as running, unresponsive to `SIGTERM` (needed `SIGKILL`). Root cause:
   `ee.data.getTaskStatus()` (polled every 60 s while waiting for export
   tasks) performs a blocking socket read with no explicit timeout anywhere
   in the `earthengine-api`/`httplib2` call chain — a dropped connection
   with no RST just blocks forever instead of raising, so the retry
   decorator never gets a chance to act; there is no exception to catch.
   Fixed: `socket.setdefaulttimeout(120)` at import time in
   `src/utils/retries.py`, so any such blocking call raises
   `socket.timeout` instead (already retried, and if still stuck,
   eventually raises and lets the process crash and be restarted — a stuck
   process should fail loudly, not hang silently forever).
6. **7-day OAuth token expiry** (see "Credentials" above) — surfaces as
   `google.auth.exceptions.RefreshError: invalid_grant: Token has been
   expired or revoked`, requiring a manual re-authentication and process
   restart; every module resumes correctly from local file presence once
   re-authenticated.

None of these lost any data: every failure mode above either left the
process safely resumable (crash) or already-downloaded files intact
(hang), never silently corrupted or skipped a file without a matching
`ValueError`/manifest record.

## Connection to the next stage

[Build / alignment](02-build.md) reads directly from
`$V2_RAW_DATA_DIR/{era5_ee,imerg}/<region>/<year>/<month>/`; it does not
consult this stage's manifests (a missing source file for a given day is
simply detected and rejected at build time, `reject_reasons:
["missing_source_file"]`), so the two stages can be run independently and
the build stage naturally tracks acquisition progress as it happens rather
than waiting for a fully completed manifest.
