# Step 1: Data Extraction & Acquisition

## Overview
This step retrieves all raw data required for the comparative downscaling experiment. The output of this step is a set of structured, paired `.npz` files on local disk containing ERA5 inputs and the two competing high-resolution targets: **CHIRPS** (Pipeline A) and **Oya** (Pipeline B).

**Goal:** Produce matched (input, target) samples indexed by date, stored under `data/` in the format consumed by the preprocessing step.

---

## 1. Geographic Domain & Coverage

The project uses **two named basin shapefiles** stored in `data/shape_files/`. They define the real study domain — which is significantly larger than Mexico alone.

### Basin Shapefiles

| Basin | File | LON range | LAT range |
| ----- | ---- | --------- | --------- |
| **Atlantic** | `atlantico_shp/atlantico_shp_grande.shp` | −98.9° to −38.7° | 5.1°N to 32.5°N |
| **Pacific** | `pacifico_shp/pacifico_shp_grande.shp` | −133.5° to −80.7° | 5.0°N to 33.8°N |

> [!IMPORTANT]
> The shapefiles are **irregular polygons**, not simple rectangles. They cover the continental land masses and adjacent ocean zones of the Gulf of Mexico and the Eastern Pacific respectively.

### Countries Covered

The **union** of both basins spans the following territories:

| Country / Territory | In Atlantic Basin | In Pacific Basin |
| ------------------- | :---: | :---: |
| **Mexico** | Yes (Gulf coast, Yucatan) | Yes (Pacific coast) |
| **Guatemala** | Yes | Yes |
| **Belize** | Yes | No |
| **Honduras** | Yes | Yes |
| **El Salvador** | Yes | Yes |
| **Nicaragua** | Yes | Yes |
| **Costa Rica** | Yes | Yes |
| **Panama** | Yes | Yes |
| **Cuba** | Yes | No |
| **Caribbean Islands** | Yes (western) | No |
| **Eastern Pacific Ocean** | No | Yes |

### Extraction Domain Polygon (Convex Hull)

All GEE exports use the **9-point convex hull** computed from the union of all shapefile vertices. This is the tightest polygon that fully contains both basins with no wasted area, and avoids exporting large rectangular regions of open ocean.

| Vertex | Longitude | Latitude | Location description |
| ------ | --------- | -------- | -------------------- |
| 1 | -133.471 | 18.626 | NW — open Pacific |
| 2 | -124.199 | 33.753 | N — Baja California North |
| 3 | -61.429 | 32.525 | NE — Caribbean North |
| 4 | -38.653 | 29.721 | E — open Atlantic North |
| 5 | -38.653 | 18.490 | E — open Atlantic Mid |
| 6 | -38.689 | 5.288 | SE — Atlantic equatorial |
| 7 | -53.015 | 5.069 | S — Guyana coast |
| 8 | -80.666 | 4.970 | S — Panama/Colombia |
| 9 | -117.667 | 5.101 | SW — Eastern Pacific equatorial |

> [!NOTE]
> During evaluation and masking, the individual basin shapefiles are applied to separate Atlantic-facing from Pacific-facing dynamics. During extraction, the convex hull polygon is used.

```python
# Use this in gee_extractor.py
DOMAIN_POLYGON = ee.Geometry.Polygon([[
    [-133.471, 18.626],  # NW  -- open Pacific
    [-124.199, 33.753],  # N   -- Baja California North
    [-61.429,  32.525],  # NE  -- Caribbean North
    [-38.653,  29.721],  # E   -- open Atlantic North
    [-38.653,  18.490],  # E   -- open Atlantic Mid
    [-38.689,   5.288],  # SE  -- Atlantic equatorial
    [-53.015,   5.069],  # S   -- Guyana coast
    [-80.666,   4.970],  # S   -- Panama/Colombia
    [-117.667,  5.101],  # SW  -- Eastern Pacific equatorial
]])
```

---

## 2. Input Data: ERA5

| Property         | Value                                                        |
| ---------------- | ------------------------------------------------------------ |
| **GEE ID**       | `ECMWF/ERA5_LAND/HOURLY`                                     |
| **Resolution**   | 0.25° (~25km)                                                |
| **Cadence**      | Hourly → **aggregated to daily** (sum for precip, mean for others) |
| **Variables**    | See table below                                              |

### ERA5 Bands to Extract

| # | Band Name                              | Units   | Role                              |
|---|----------------------------------------|---------|-----------------------------------|
| 1 | `total_precipitation_hourly`           | m/day   | Low-res precipitation signal      |
| 2 | `temperature_2m`                       | K       | Convection driver                 |
| 3 | `dewpoint_temperature_2m`              | K       | Surface moisture                  |
| 4 | `surface_pressure`                     | Pa      | Orographic signature              |
| 5 | `u_component_of_wind_10m`              | m/s     | Zonal moisture flux               |
| 6 | `v_component_of_wind_10m`              | m/s     | Meridional moisture flux          |
| 7 | `surface_solar_radiation_downwards_hourly` | J/m² | Convective initiation driver  |
| 8 | `surface_sensible_heat_flux_hourly`    | J/m²    | Boundary layer instability        |
| 9 | `surface_latent_heat_flux_hourly`      | J/m²    | Evapotranspiration signal         |
| 10| `volumetric_soil_water_layer_1`        | frac    | Antecedent soil moisture          |

**Pressure Level Bands** (from `ECMWF/ERA5/DAILY`):

| Band | Levels (hPa)                                     |
|------|--------------------------------------------------|
| `temperature` | 1000, 925, 850, 700, 600, 500, 400, 300, 200 |

> [!NOTE]
> ERA5-Land does not include pressure-level temperature. These 9 bands must be extracted separately from the `ECMWF/ERA5/DAILY` collection and spatially resampled to match the ERA5-Land grid.

---

## 3. Target A: CHIRPS (Pipeline A)

| Property        | Value                                         |
| --------------- | --------------------------------------------- |
| **GEE ID**      | `UCSB-CHG/CHIRPS/DAILY`                       |
| **Resolution**  | 0.05° (~5km)                                  |
| **Cadence**     | Daily                                         |
| **Band**        | `precipitation` (mm/day)                      |
| **Availability**| 1981-01-01 to present                         |

**Notes for extraction:**
- Clip to the domain polygon and reproject to `EPSG:4326`.
- Mask pixels with `precipitation < 0` (fill value = -9999).
- Export as multi-band GeoTIFF per month (or per week for smaller file sizes).

---

## 4. Target B: Oya (Pipeline B)

| Property        | Value                                                                   |
| --------------- | ----------------------------------------------------------------------- |
| **GEE ID**      | `projects/global-precipitation-nowcast/assets/global_estimation`       |
| **Resolution**  | 0.05° (~5km)                                                            |
| **Cadence**     | 30-minute → **aggregated to daily**                                     |
| **Band**        | `precipitation` (mm/hr → converted to mm/day by summing hourly slices) |
| **Availability**| 2004-01-01 to present                                                   |

**Notes for extraction:**
- Use `.filterDate()` and `.select('precipitation')`.
- Aggregate: `collection.sum()` within each UTC day (48 images per day).
- Convert units: 30-min images are mm/hr, so `sum() × 0.5` gives mm/day.
- Clip to the domain polygon, reproject to `EPSG:4326`.

---

## 5. Topography: NASADEM/SRTM

| Property        | Value                        |
| --------------- | ---------------------------- |
| **GEE ID**      | `NASA/NASADEM_HGT/001`       |
| **Resolution**  | ~30m (resampled to 1km)      |
| **Band**        | `elevation` (meters)         |

**Notes:**
- Export a single static GeoTIFF at 1km resolution clipped to the domain polygon.
- This is used by the Upslope and Spectral physics models in Step 2.
- Reproject to `EPSG:4326`.

---

## 6. Data Splits

Both pipelines use the same chronological splits:

| Split      | Period          | # Days (approx) | Purpose                                  |
| ---------- | --------------- | ---------------- | ---------------------------------------- |
| Training   | 2004-01-01 – 2015-12-31 | ~4383  | Model learning; covers ENSO variability  |
| Validation | 2016-01-01 – 2017-12-31 | ~730   | Hyperparameter tuning & model selection  |
| Test       | 2018-01-01 – 2019-12-31 | ~730   | Final held-out evaluation                |

> [!NOTE]
> Start year is constrained to **2004** by Oya's earliest availability. CHIRPS data from 2004 onward will be used even though it is available from 1981.

---

## 7. Directory Layout

After Step 1 completes, the `data/` directory should have this structure:

```
data/
├── shape_files/          # Mexico shapefiles (already present)
├── raw/
│   ├── era5/
│   │   └── YYYY/MM/
│   │       └── era5_YYYY-MM-DD.tif
│   ├── chirps/
│   │   └── YYYY/MM/
│   │       └── chirps_YYYY-MM-DD.tif
│   ├── oya/
│   │   └── YYYY/MM/
│   │       └── oya_YYYY-MM-DD.tif
│   └── dem/
│       └── nasadem_mexico_1km.tif
├── processed/
│   ├── chirps/
│   │   ├── train/   → paired .npz files
│   │   ├── val/
│   │   └── test/
│   └── oya/
│       ├── train/
│       ├── val/
│       └── test/
├── metadata/
│   ├── dataset_index_chirps.csv
│   ├── dataset_index_oya.csv
│   ├── norm_stats_chirps.json   ← populated in Step 2
│   └── norm_stats_oya.json      ← populated in Step 2
└── logs/
```

Each `.npz` file contains:
- `inputs`: `ndarray` of shape `(H, W, 10+9)` — ERA5 surface + pressure level bands, resampled to 5km.
- `target`: `ndarray` of shape `(H, W, 1)` — CHIRPS or Oya daily precipitation in mm/day.
- `date`: ISO date string (e.g., `"2007-08-14"`).

---

## 8. Scripts to Implement

### `src/data/gee_extractor.py`

```
Responsibilities:
  - Authenticate with GEE using credentials from .env
  - Functions:
      export_era5(year, month, drive_folder)
      export_chirps(year, month, drive_folder)
      export_oya(year, month, drive_folder)
      export_dem(drive_folder)
  - Each function creates a GEE Export.image.toDrive() task.
  - Returns task ID for monitoring.
  - CLI: python src/data/gee_extractor.py --dataset [era5|chirps|oya|dem]
          --start YYYY-MM --end YYYY-MM
```

### `src/data/drive_downloader.py`

```
Responsibilities:
  - Poll GEE export tasks until COMPLETED or FAILED.
  - Download finished GeoTIFFs from Google Drive to data/raw/{dataset}/YYYY/MM/.
  - CLI: python src/data/drive_downloader.py --dataset [era5|chirps|oya]
```

### `src/data/npz_converter.py`

```
Responsibilities:
  - Read matched ERA5 + target (CHIRPS or Oya) GeoTIFFs for each date.
  - Spatially resample ERA5 from 0.25° to the 5km target grid using bilinear interp.
  - Save paired (inputs, target) to data/processed/{target}/{split}/YYYY-MM-DD.npz.
  - Write dataset_index_{target}.csv at data/metadata/.
  - CLI: python src/data/npz_converter.py --target [chirps|oya]
```

---

## 9. Key Implementation Details

### GEE Authentication
Use the service account key defined in `.env`:
```python
import ee
from dotenv import load_dotenv
import os

load_dotenv()
credentials = ee.ServiceAccountCredentials(
    email=None,
    key_file=os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
)
ee.Initialize(credentials)
```

### ERA5 Daily Aggregation in GEE
```python
era5 = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY") \
    .filterDate(start, end) \
    .filterBounds(mexico_bbox) \
    .select(SURFACE_BANDS)

# Sum precipitation, mean for all others
daily_precip = era5.select(['total_precipitation_hourly']).sum()
daily_others = era5.select(OTHER_BANDS).mean()
daily_image   = daily_precip.addBands(daily_others)
```

### Oya Daily Aggregation in GEE
```python
oya = ee.ImageCollection("projects/global-precipitation-nowcast/assets/global_estimation") \
    .filterDate(start, end) \
    .filterBounds(mexico_bbox) \
    .select(['precipitation'])

# mm/hr × 0.5hr per image = mm per 30-min slot; sum gives mm/day
daily_oya = oya.sum().multiply(0.5)
```

### Spatial Resampling for ERA5
ERA5 is 0.25° and must be resampled to the 5km CHIRPS/Oya grid before saving to `.npz`. Use `rasterio.warp.reproject` with bilinear resampling:
```python
with rasterio.open(era5_tif) as src:
    data, _ = rasterio.warp.reproject(
        source=rasterio.band(src, list(range(1, src.count + 1))),
        destination=np.empty((src.count, target_height, target_width)),
        src_crs=src.crs,
        dst_crs=target_crs,
        dst_transform=target_transform,
        resampling=rasterio.enums.Resampling.bilinear
    )
```

---

## 10. Expected Step Completion Criteria

- [ ] All GEE export tasks run successfully for ERA5, CHIRPS, Oya, and DEM.
- [ ] GeoTIFFs downloaded from Drive to `data/raw/`.
- [ ] `.npz` files generated for all 3 splits, for both CHIRPS and Oya tracks.
- [ ] `dataset_index_chirps.csv` and `dataset_index_oya.csv` written with columns: `date`, `split`, `era5_path`, `target_path`, `valid_flag`.
- [ ] A visual sanity check notebook (`notebooks/01_first_images.ipynb`) confirms spatial alignment of ERA5 inputs vs CHIRPS/Oya targets over Mexico.

---

## 11. Glossary

| Term | Definition |
|------|-----------|
| **GEE** | Google Earth Engine – cloud geospatial platform used for querying and exporting gridded climate data. |
| **ERA5** | ECMWF 5th-generation atmospheric reanalysis at 0.25° / hourly resolution. Used as the low-resolution model input. |
| **CHIRPS** | Climate Hazards Group InfraRed Precipitation with Station data. 5km daily blended satellite+station product (1981–present). |
| **Oya** | Google Research quasi-global precipitation product at 5km / 30-min. Derived from geostationary VIS-IR using a U-Net (2004–present). |
| **NASADEM** | NASA digital elevation model at 30m, used for orographic physics models. |
| **NPZ** | NumPy compressed archive format storing multiple named arrays per date. |
| **Bilinear resampling** | Spatial interpolation method that estimates pixel values using a weighted average of the 4 nearest neighbours. Used to resample ERA5 from 0.25° to the 5km target grid. |
| **`dataset_index.csv`** | Master file listing every sample path, its split, date, and validity flag. Primary entry point for the Step 2 dataloaders. |