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

### Input: pure ERA5 cp + lsp

`src/data_extraction/era5_cds.py` acquires `convective_precipitation` (cp)
and `large_scale_precipitation` (lsp) from the Copernicus CDS
(`reanalysis-era5-single-levels`), hourly, on ERA5's native ~0.25° grid,
scoped to one named region (`src.data_preprocessing.regions`). This is a new
module: no ERA5-pure (CDS-based) acquisition code previously existed in
`src/` — the pre-existing `raw/era5_pl` archive (pure ERA5 500/850 hPa
channels documented in `docs/stages/data-and-resources.md`) was acquired by
an external, unversioned process. `era5_cds.py` fills that gap for cp/lsp
specifically and is the acquisition path going forward for this input.
Delivery: one NetCDF per `(variable, month, region)` under
`$V2_RAW_DATA_DIR/era5_cds/<region>/<variable>/`, with a hashed manifest.
Requires `CDSAPI_URL` + `CDSAPI_KEY` (or `~/.cdsapirc`); fails explicitly,
never falls back to synthetic data (§ Credentials below).

### Primary target: IMERG V07 Final Run

`src/data_extraction/imerg.py` downloads the half-hourly `GPM_3IMERGHH.07`
product from NASA GES DISC, subsets it to a region bounding box, and
averages it to hourly (RainShift, P2 M203) to match ERA5's temporal
resolution. Delivery: one NetCDF/day under
`$V2_RAW_DATA_DIR/imerg/<region>/`, with a hashed manifest. Requires
`EARTHDATA_TOKEN` (or username/password, or `~/.netrc`); same
fail-explicitly policy as ERA5.

### Secondary/refinement target: Oya

Oya (`src/data_extraction/export_oya.py`, via Google Earth Engine) is kept
as the secondary, refinement-stage target, now region-scoped: passing
`--region <name>` clips the export to that region's bounding box and writes
to `$V2_RAW_DATA_DIR/oya/<region>/` instead of the legacy
`$RAW_DATA_DIR/v2_clean/oya/` path (omitting `--region` reproduces the
legacy unscoped export unchanged, so `pipeline.py`'s existing `acquire`
stage for the archived deterministic pipeline is unaffected).

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

### Credentials status (2026-09-11)

Neither CDS nor Earthdata credentials are available yet (confirmed with the
project owner). `era5_cds.py` and `imerg.py` are implemented and tested
(credential-gating, resume, manifest logic — no network calls), but **no
acquisition has been executed and no data has been downloaded** this round,
matching the same policy from the reverted attempt #1.

## Legacy sources (unchanged)

CHIRPS and ERA5-Land remain exactly as documented in
`docs/stages/data-and-resources.md`. They are daily-only and continue to
serve the archived deterministic pipeline (`legacy/deterministic_unet/`
after PASO 4); they are not part of the hourly ERA5→IMERG→Oya cascade.
