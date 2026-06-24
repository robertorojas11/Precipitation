# Step 1: Data Acquisition

**Status:** Nearly complete (reference only — verify before moving to Step 2)

## Objective
Acquire all low-resolution inputs (ERA5) and dual high-resolution targets (CHIRPS & Oya) for the Mexico domain, covering the full overlapping temporal range.

## Domain & Period
- **Spatial domain:** Lat 12°N – 35°N, Lon 84°W – 120°W (Mexico)
- **Temporal range:** 2004 – present (ensures ERA5 / CHIRPS / Oya overlap)

## Required Datasets

### 1. ERA5 (Low-Resolution Input)
- **Resolution:** 0.25° (~28 km)
- **Variables required:**
  - Total precipitation (tp)
  - Pressure-level winds: u, v, w (vertical velocity) at multiple levels (e.g. 1000, 925, 850, 700, 500 hPa)
  - Surface geopotential / orography
- **Source:** Copernicus Climate Data Store (CDS) ERA5 hourly/daily products

### 2. ERA5-Land (Intermediate Target for GAN-1)
- **Resolution:** 0.1° (~10 km)
- **Variables required:** Total precipitation (tp)
- **Source:** Copernicus CDS ERA5-Land

### 3. CHIRPS (Pipeline A Ground Truth)
- **Resolution:** 0.05° (5 km)
- **Variable:** Daily precipitation
- **Source:** Climate Hazards Center (UCSB), CHIRPS v2.0

### 4. Oya (Pipeline B Ground Truth)
- **Resolution:** 0.05° (5 km)
- **Temporal resolution:** 30-minute
- **Variable:** Precipitation estimate (geostationary VIS-IR derived)
- **Source:** Google Research nowcasting product

### 5. High-Resolution Topography (DEM)
- **Resolution:** Should match or exceed target output resolution (5 km, ideally 1 km)
- **Source:** SRTM, GMTED, or equivalent
- **Required derived fields:** elevation, terrain gradients (∂z/∂x, ∂z/∂y)

## Verification Checklist (Before Proceeding to Step 2)
- [ ] All datasets cover the full Mexico bounding box (12°N–35°N, 84°W–120°W)
- [ ] Temporal overlap confirmed across ERA5, ERA5-Land, CHIRPS, and Oya for 2004–present
- [ ] No missing time steps / large data gaps in any dataset
- [ ] Spatial grids and coordinate reference systems (CRS) documented for each dataset
- [ ] DEM acquired and aligned to the same domain
- [ ] Raw files stored in a consistent directory structure with metadata (units, resolution, time range) recorded
- [ ] File formats noted (NetCDF/GRIB for ERA5 family, GeoTIFF/NetCDF for CHIRPS/Oya/DEM)

## Output of This Step
A local data archive containing raw, unprocessed files for all five datasets above, organized by dataset and time period, ready for the preprocessing pipeline in Step 2.

## Next Step
Proceed to **step2.md** — Data Preprocessing & Physical Models.
