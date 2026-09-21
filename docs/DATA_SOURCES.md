# Data sources — v2 generative framework

This document describes the data sources introduced or re-scoped by
`docs/decisions/0002-implementacion-v2.md`. It complements, and does not
replace, `docs/stages/data-and-resources.md`, which documents the data
contract of the archived deterministic CHIRPS/Oya pipeline (`legacy/`).

## Region shapefiles (`ATLANTICO_SHP_PATH` / `PACIFICO_SHP_PATH`)

`src/utils/config.py` has carried these two environment variables since
before this migration; they were unused by any code path until now (PASO 2).

**What they actually contain.** Both are single-polygon shapefiles, EPSG:4326,
read with `geopandas`:

| Variable | Path (this deployment) | Vertices | Bounding box (lon, lat) |
|---|---|---:|---|
| `ATLANTICO_SHP_PATH` | `.../shape_files/atlantico_shp/atlantico_shp_grande.shp` | 23 | lon [-98.92°, -38.65°], lat [5.07°, 32.52°] |
| `PACIFICO_SHP_PATH` | `.../shape_files/pacifico_shp/pacifico_shp_grande.shp` | 12 | lon [-133.47°, -80.67°], lat [4.97°, 33.75°] |

These are **not** coastlines or administrative divisions. They are
whole-basin "area of responsibility" polygons matching the Atlantic and
Eastern Pacific tropical-cyclone basins used operationally by NHC/CPHC and
by Avila (2026) (§2: *"the two basins are both represented... Eastern Pacific
and North Atlantic storms"*). Point-in-polygon checks confirm this:
Acapulco and Manzanillo (Mexico's Pacific coast) fall inside `PACIFICO`
only; Veracruz, Cancún, San Juan (Puerto Rico), and Barbados all fall inside
`ATLANTICO` only; Mexico City (inland) falls inside neither. The
`ATLANTICO` polygon already geometrically contains the Caribbean and Puerto
Rico — it is not restricted to the open North Atlantic.

**Do they already separate México-Pacífico from México-Atlántico/Golfo?**
Yes, for basin membership (which coast a point is on). No, for defining a
*bounded region* usable as a RainShift-style training/evaluation area — the
polygons span entire ocean basins (Baja California to South America for
`PACIFICO`; the Gulf of Mexico to the mid-Atlantic for `ATLANTICO`), which
is far larger than any region this project can realistically acquire dense
ERA5/IMERG/Oya data for in one pass.

**Resolution adopted:** `src/data_preprocessing/regions.py` defines named
regions as a bounding box **intersected with** the relevant basin polygon
(`region_mask(..., require_basin=True)`, the default). This keeps inland or
out-of-basin pixels out of a region while bounding the area to something
acquisition-sized:

| Region | Basin | Bounding box (lon, lat) | Role | Rationale |
|---|---|---|---|---|
| `mexico_pacifico` | `pacifico` | [-120°, -104°] × [12°, 32°] | train/eval | Mexico's Pacific coast, Baja California to Chiapas. |
| `mexico_atlantico_golfo` | `atlantico` | [-98°, -84°] × [15°, 30°] | train/eval | Gulf of Mexico + Mexico's Caribbean coast, Tamaulipas to Quintana Roo. |
| `puerto_rico_caribe` | `atlantico` | [-85°, -59°] × [10°, 23°] | eval | Greater + Lesser Antilles arc. **New region for this project** — not a coastal/administrative boundary, a working bounding box confirmed with the project owner on 2026-09-11 (see `docs/decisions/0002-implementacion-v2.md` §1.4). Comparable in scale to RainShift's 20°×20° evaluation regions. |

`puerto_rico_caribe` was added specifically because neither the prior fixed
domain (`build_dataset.py`, lon [-120°,-84°] / lat [12°,35°]) nor either
basin polygon alone singles out the Caribbean/Puerto Rico as an evaluation
area — the fixed domain excludes it entirely (it stops at lon -84°), and the
`ATLANTICO` polygon is too large to serve as a bounded region on its own.

Any additional region is a new `Region` entry in the same registry, so
scaling geographic coverage later is configuration, not a pipeline rewrite
(mirrors the RainShift `A1 ⊂ A2 ⊂ A3 ⊂ A4` region-union mechanism, PASO 1.1).

## Ground-truth cascade and ERA5 pure inputs

**Revision 2026-09-21: acquisition backend switched to Google Earth
Engine.** The first version of this section (PASO 3, 2026-09-11) specified
the Copernicus CDS API for ERA5 and direct NASA GES DISC HTTP access for
IMERG, gated on CDS/Earthdata credentials that were not available. The
project owner has Earth Engine + Drive service-account credentials instead
(`auth/service_account.json`, already used by `export_oya.py`). Checked
against the live Earth Engine Data Catalog (2026-09-21):

- `ECMWF/ERA5/HOURLY` (the **pure** ERA5 product, not ERA5-Land) exposes
  `mean_convective_precipitation_rate` and
  `mean_large_scale_precipitation_rate` directly — exactly the two P1
  variables, at ERA5's native ~0.25° grid. No CDS credential needed.
- `NASA/GPM_L3/IMERG_V07` exposes the calibrated `precipitation` band
  (mm/hr, Final-Run-equivalent), half-hourly. No Earthdata Login token
  needed.

Both are now acquired the same way `export_oya.py` already acquires Oya:
Earth Engine batch export-to-Drive, synced locally via
`src.data_extraction.drive.sync_prefixed`. The physical variables and their
role in the pipeline are unchanged from the original decision — only the
backend differs. `docs/decisions/0002-implementacion-v2.md` records this as
a revision note under PASO 3.1/3.2, not a new decision.

### Input: pure ERA5 cp + lsp

`src/data_extraction/era5_ee.py` (replaces the removed `era5_cds.py`)
pulls `ECMWF/ERA5/HOURLY`, converts the rate bands to accumulated mm per
hour (`rate[kg/m^2/s] * 3600`), and exports one 48-band GeoTIFF per
`(region, day)` (24 hours × 2 variables) to Drive, scoped to one named
region (`src.data_preprocessing.regions`). Delivery:
`$V2_RAW_DATA_DIR/era5_ee/<region>/<year>/<month>/era5_<region>_<date>.tif`,
with a hashed manifest recording the band layout. Fails explicitly if
Earth Engine initialization fails; never falls back to synthetic data.

### Primary target: IMERG V07 Final Run

`src/data_extraction/imerg.py` (rewritten in place) pulls
`NASA/GPM_L3/IMERG_V07`'s half-hourly `precipitation` band and exports one
48-band GeoTIFF per `(region, day)` to Drive. `halfhourly_to_hourly()`
still reduces the 48 half-hourly bands to 24 hourly means (RainShift, P2
M203) — now applied to the locally downloaded GeoTIFF instead of remote
HDF5 granules. Delivery:
`$V2_RAW_DATA_DIR/imerg/<region>/<year>/<month>/imerg_<region>_<date>.tif`,
with a hashed manifest. Recent granules may be marked "provisional" by the
provider (permanent V07 products end 2025-09-30 ahead of a V08
transition) — acceptable for training data, but worth knowing if a result
looks like it shifted after a later re-run once NASA finalizes a granule.

### Secondary/refinement target: Oya

Oya (`src/data_extraction/export_oya.py`, via Google Earth Engine) is kept
as the secondary, refinement-stage target, now region-scoped: passing
`--region <name>` clips the export to that region's bounding box and writes
to `$V2_RAW_DATA_DIR/oya/<region>/` instead of the legacy
`$RAW_DATA_DIR/v2_clean/oya/` path (omitting `--region` reproduces the
legacy unscoped export unchanged, so `pipeline.py`'s existing `acquire`
stage for the archived deterministic pipeline is unaffected). The
region-scoped export uses a flat Drive filename
(`oya_<region>_<date>.tif`, no embedded "/") and syncs via the same
`drive.sync_prefixed` helper as ERA5/IMERG — a bug fix made alongside the
2026-09-21 revision: Google Drive treats "/" in an export's `fileNamePrefix`
as a literal character in the file's `name`, not a path separator, so an
embedded "/" broke local path reconstruction on sync.

**Known limitations of Oya** (documented explicitly, per PASO 3.2, so it is
never treated as equivalent in quality to IMERG):

- It is an indirect, IR/VIS-based nowcast product, not a direct
  microwave/radar retrieval like IMERG.
- Its training reportedly used IMERG as a partial reference, so treating Oya
  as fully independent ground truth for the same task IMERG already covers
  double-counts some of the same information.
- Weaker performance is expected in arid/mountainous zones (consistent with
  `docs/stages/data-and-resources.md`'s existing note that its
  retrieval/model error differs from station-blended products).
- No guarantee of continued provider availability (`projects/
  global-precipitation-nowcast/assets/global_estimation` is a third-party
  Earth Engine asset, not a versioned mission product like IMERG).

### Two-stage cascade vs. single-step design

Documented in `docs/architecture_v3.md` § "Data cascade design": the
pipeline can train either (a) a single step ERA5 → final target, or (b) a
two-stage cascade, ERA5 → IMERG (stage 1, supervised) → Oya (stage 2,
higher-detail refinement). Both options are designed for; which one is used
for a given training run is a configuration choice, not a hard-coded
pipeline shape.

### Credentials status (2026-09-21)

The project owner has Earth Engine + Drive service-account credentials
(`auth/service_account.json`), which is now sufficient for both ERA5 and
IMERG acquisition — no separate CDS API key or NASA Earthdata Login token
is required (§ revision note above). `era5_ee.py` and `imerg.py` are
implemented and unit-tested (path construction, manifest round-trips, the
flat-filename parsing that makes Drive sync correct — no real Earth Engine
or Drive network calls in tests), but **no acquisition has been executed
and no data has been downloaded** as of this writing; running
`python -m src.data_extraction.era5_ee ...` / `python -m
src.data_extraction.imerg ...` for at least one region is the next concrete
step toward a real training run.

## Legacy sources (unchanged)

CHIRPS and ERA5-Land remain exactly as documented in
`docs/stages/data-and-resources.md`. They are daily-only and continue to
serve the archived deterministic pipeline (`legacy/deterministic_unet/`
after PASO 4); they are not part of the hourly ERA5→IMERG→Oya cascade.
