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

Documented in `docs/decisions/0002-implementacion-v2.md` §1.4 and expanded
here once the PASO 3 acquisition modules land (ERA5 cp/lsp extension, IMERG
V07, and the Oya secondary-refinement adaptation). See that section for the
current status.

## Legacy sources (unchanged)

CHIRPS and ERA5-Land remain exactly as documented in
`docs/stages/data-and-resources.md`. They are daily-only and continue to
serve the archived deterministic pipeline (`legacy/deterministic_unet/`
after PASO 4); they are not part of the hourly ERA5→IMERG→Oya cascade.
