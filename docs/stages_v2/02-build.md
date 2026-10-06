# Stage 2 — Build / alignment

## Purpose and inputs

Converts each `(region, day)` of raw acquisition output into one aligned,
mask-aware hourly sample. This is the hourly, per-region analogue of the
legacy pipeline's `03-build.md` — the underlying idea (reproject every
source onto one canonical grid, propagate masks explicitly, version the
output) is unchanged, but there is no single fixed domain here: each of the
three regions has its **own** canonical grid, since their bounding boxes
differ (see `docs/DATA_SOURCES.md`).

Inputs, per `(region, day)`:

- `$V2_RAW_DATA_DIR/era5_ee/<region>/<year>/<month>/era5_<region>_<date>.tif`
  — 48 bands (24 hours × [cp, lsp]), accumulated mm per hour, ERA5's native
  ~0.25° grid (see [Acquisition](01-acquire.md)).
- `$V2_RAW_DATA_DIR/imerg/<region>/<year>/<month>/imerg_<region>_<date>.tif`
  — 48 bands (half-hourly snapshots), mm/hr, IMERG's native ~0.1° grid.
- `$RAW_DATA_DIR/dem/nasadem_mexico_1km.tif` — NASADEM elevation. Despite
  the filename (a holdover from the legacy pipeline, which only used a
  cropped view of it), the underlying raster spans lon [-133.47°, -38.65°],
  lat [4.97°, 37.43°] — confirmed to cover all three v2-generative regions,
  including `puerto_rico_caribe`, which the legacy pipeline's *cropped*
  460×720 domain (lon [-120°, -84°]) did not reach.

Implementation: `src/data_preprocessing/build_hourly_dataset.py`.

```bash
python -m src.data_preprocessing.build_hourly_dataset --region mexico_pacifico
python -m src.data_preprocessing.build_hourly_dataset --region mexico_pacifico --start 2015-01-01 --end 2015-12-31 --limit 30
```

## The canonical per-region grid

`region_grid(region_name)` defines a clean north-up grid at 0.1°
resolution (`IMERG_RESOLUTION_DEG`, matching IMERG's native resolution)
spanning the region's bounding box **exactly** — width/height computed as
`round((lon_max − lon_min) / 0.1)` / `round((lat_max − lat_min) / 0.1)`,
origin at `(lon_min, lat_max)`. This is deliberately **not** the same grid
any particular downloaded file happens to use: Earth Engine snaps every
export to its own global pixel grid, whose origin offset varies slightly
per region (confirmed empirically — e.g. `mexico_pacifico`'s downloaded
IMERG file spans lat up to 32.300147°, not exactly 32.0° or 32.5°). Every
source raster (ERA5, IMERG, DEM) is reprojected *onto* this
self-defined canonical grid via `rasterio.warp.reproject`, bilinear
resampling, so the pipeline's own grid definition is authoritative and
independent of Earth Engine's export-time snapping. Measured shapes at the
time of writing:

| Region | Canonical shape (H, W) |
|---|---|
| `mexico_pacifico` | (200, 160) |
| `mexico_atlantico_golfo` | (150, 140) |
| `puerto_rico_caribe` | (130, 260) |

## Per-day processing

1. **ERA5.** All 48 bands of the day's GeoTIFF are reprojected onto the
   canonical grid in a single call (bilinear), then split back into
   `(24, H, W, 2)` `[cp, lsp]` using `parse_era5_hour_bands()`, which reads
   each band's **description** (`"<YYYYMMDD>T<HH>_mean_convective_
   precipitation_rate"`, embedded by `era5_ee.py`'s `toBands()` call) rather
   than assuming a fixed stride — a deliberate robustness choice: if Earth
   Engine's `toBands()` band ordering ever changed, this raises
   `ValueError` immediately instead of silently misattributing an hour's
   data to the wrong band index.
2. **IMERG.** All 48 half-hourly bands are reprojected onto the same
   canonical grid, then reduced to `(24, H, W)` hourly means via
   `src.data_extraction.imerg.halfhourly_to_hourly` (RainShift, P2 M203) —
   the same function the acquisition stage's own local reduction uses,
   reused rather than reimplemented.
3. **Elevation.** NASADEM reprojected onto the canonical grid once per
   region (bilinear) and **cached across the whole region's build run**
   (`build_region()` computes it once, not once per day) — it never
   changes day to day. A land mask is derived as `elevation > 0.0`
   (`LAND_ELEVATION_THRESHOLD_M`). NASADEM has no explicit nodata value in
   this archive; ocean samples were confirmed to read exactly `0 m`
   (min observed value in the whole raster: −71 m, plausible bathymetric
   noise near coastlines), so a plain threshold is workable. This is
   **approximate**, and — unlike the legacy pipeline, which corroborated
   its elevation threshold with CHIRPS's own stable coverage mask — there
   is no second, independent signal here to cross-check it against, since
   CHIRPS is excluded from this cascade by design
   (`docs/decisions/0002-implementacion-v2.md` §1.4). Land mask is an
   **informational conditioning channel** (orography-adjacent), not a
   validity filter — precipitation over open ocean is physically
   meaningful (it is literally where tropical cyclones matter most) and
   must not be excluded from the target.
4. **Region mask.** `src.data_preprocessing.regions.region_mask()` applied
   to the canonical grid's coordinate arrays — the same
   basin-polygon-intersected mask used by acquisition's region scoping,
   now applied at pixel level. This *is* a validity filter: a pixel outside
   the region's basin polygon (e.g. an inland point captured by the
   bounding box but outside the tropical-cyclone basin, per
   `docs/DATA_SOURCES.md`) is invalid for both input and target.
5. **Validity masks.**
   `input_valid_mask = ensure_finite_features(inputs) & basin_mask`,
   `target_valid_mask = isfinite(target) & (|target| < 1e10) & basin_mask`
   — reusing `src.data_preprocessing.quality.ensure_finite_features` and
   the same `1e10` sentinel-value threshold the legacy pipeline uses
   (`quality.FLOAT_FILL_THRESHOLD`). A day with **zero** valid target
   pixels across all 24 hours is rejected outright
   (`reject_reasons: ["no_valid_target_pixels"]`) rather than written as an
   all-invalid sample.
6. **Split assignment.** `year_split(day.year)` — train 2010–2019,
   validation 2020–2021, test 2022–2023 (provisional, OQ-6 in
   `docs/decisions/0002-implementacion-v2.md`). Applied identically to
   every region; `puerto_rico_caribe`'s "never train on this region" status
   is enforced later, at training time, by simply not drawing training
   batches from it — the region's `role` field
   (`src/data_preprocessing/regions.py`) already records this, so the build
   stage does not need its own special case for it.

## Reprojection edge effect (known, handled)

Reprojecting a source raster onto the canonical grid can leave a thin
border of pixels with no source coverage (`NaN`) where the canonical grid
extends fractionally beyond the source file's own footprint. For IMERG,
if *both* half-hour slots at such a pixel are `NaN`,
`halfhourly_to_hourly`'s `nanmean` correctly returns `NaN` but also emits a
"Mean of empty slice" warning per occurrence — harmless (those pixels are
excluded downstream by the finite-value validity mask), just noisy at the
scale of thousands of days, so `reproject_imerg_day()` suppresses that
specific warning explicitly rather than leaving it to spam the build log.

## Artifacts and provenance

Each accepted day produces one NPZ:

```text
<LOCAL_DATA_DIR>/v2_generative_hourly/processed/<region>/<year>/<month>/<date>.npz
├── inputs             float32 (24, H, W, 2)   -- [cp, lsp], accumulated mm/hour
├── target              float32 (24, H, W, 1)   -- IMERG, mm/hr hourly mean
├── input_valid_mask    bool    (24, H, W)
├── target_valid_mask   bool    (24, H, W)
├── land_mask           bool    (H, W)          -- static per region
├── elevation           float32 (H, W)          -- static per region
├── basin_mask          bool    (H, W)          -- static per region
├── date, region, crs, transform, preprocessing_version
```

`build_region()` additionally writes, under
`<LOCAL_DATA_DIR>/v2_generative_hourly/metadata/`:

- `dataset_index_hourly_<region>.csv` — one row per processed day: date,
  split, accepted flag, output path, rejection reasons, per-source SHA-256.
- `manifest_hourly_<region>.json` — hashed manifest
  (`src.utils.provenance.write_manifest`): git revision, deterministic
  content hash over every index record, grid definition, region summary,
  accepted/rejected counts.

`preprocessing_version = "v2_generative_hourly"`
(`HOURLY_DATASET_VERSION`) — a distinct namespace from the legacy
pipeline's `"v2_clean"`, so the two never collide on disk or in a manifest
hash.

## Tests

`tests/test_build_hourly_dataset.py` — 13 tests, entirely on small
synthetic GeoTIFFs (a temporary ~2°×2° region backed by a synthetic basin
polygon, mirroring `tests/test_regions.py`'s pattern), no real acquired
data required: canonical-grid shape/coordinate correctness (cross-checked
against `rasterio.transform.xy`), ERA5 band-description parsing (valid
input, missing hour, malformed description, missing description), land
mask thresholding, year-split boundaries, a full `build_day()` round-trip
verifying reprojected values match known uniform source fields, missing-
source rejection, out-of-range-year rejection, and a `build_region()`
run over a 3-day range verifying the index/manifest are written correctly.
Also smoke-tested against real downloaded data for one day/region
(`mexico_pacifico`, 2015-06-01) during development — accepted, 678,768
valid target-pixel-hours out of 24×200×160 = 768,000 possible.

## Known limitations

- Elevation-only land mask, not cross-checked against any second
  observational source (see "Per-day processing", step 3).
- The 2010–2019/2020–2021/2022–2023 year split is a project default, not
  derived from any paper or from measured per-region data quality — revisit
  once real coverage/quality statistics are available per region
  (`docs/decisions/0002-implementacion-v2.md` OQ-6).

## Connection to the next stage

Stage 4 (Prepare) will read these per-day NPZ files, compute train-only
normalization statistics, and assemble the final training tensors/splits.
A raw/processed validation gate (stage 3, PLANNED) is expected to sit
between this stage and Prepare, mirroring the legacy pipeline's two-gate
pattern (`02-validate-raw.md` before build, `04-validate-processed.md`
after) — not yet implemented; the CSV index and manifest this stage already
produces are the inputs such a gate would consume.
