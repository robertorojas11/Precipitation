### Data Extraction via Google Earth Engine (GEE)

**Description:**

The first step of the pipeline is a two-stage scripted extraction of paired input/output data. First, atmospheric states from ERA5-Land Hourly (~11km) and precipitation estimates from Oya 5km are spatially filtered to the Atlantic and Pacific basins bounding box (Lat 4.9°N–33.8°N, Lon 38.6°W–133.5°W) and temporally aligned to an hourly cadence. These paired images are exported from Google Earth Engine (GEE) as background batch tasks directly to a Google Drive intermediary.

In the second stage, a local Python management script uses the Google Drive API to download the exported GeoTIFF files and convert them into compressed NumPy format (`.npz`) on the dedicated training server's local filesystem. This hybrid workflow ensures high-volume data reliability while maintaining a zero-cost footprint on Google Cloud Storage.

---
**Datasets:**
- **Input — ERA5-Land Hourly**

  - GEE Collection: `ECMWF/ERA5_LAND/HOURLY`
  - Spatial resolution: ~11,132 m
  - Temporal cadence: 1 hour
  - Availability: 1950-01-01 to near real-time
  - Selected bands (13 total):
 
|#|Band|Units|Physical Role|
|---|---|---|---|
|1|`temperature_2m`|K|Near-surface air temperature; convection driver|
|2|`dewpoint_temperature_2m`|K|Near-surface moisture content|
|3|`surface_pressure`|Pa|Orographic signal; used to derive relative humidity|
|4|`u_component_of_wind_10m`|m/s|Zonal moisture flux|
|5|`v_component_of_wind_10m`|m/s|Meridional moisture flux; Gulf of Mexico onshore flow|
|6|`total_precipitation_hourly`|m|Coarse-resolution precipitation signal to be downscaled|
|7|`runoff_hourly`|m|Soil saturation proxy; precipitation memory|
|8|`surface_runoff_hourly`|m|Infiltration proxy; complements sub-surface runoff|
|9|`surface_solar_radiation_downwards_hourly`|J/m²|Convective initiation driver (afternoon storms)|
|10|`surface_net_solar_radiation_hourly`|J/m²|Net surface energy available for heating|
|11|`surface_sensible_heat_flux_hourly`|J/m²|Boundary layer instability driver|
|12|`surface_latent_heat_flux_hourly`|J/m²|Evapotranspiration and moisture recycling signal|
|13|`volumetric_soil_water_layer_1`|Vol. fraction|Top 7 cm soil moisture; antecedent land-surface condition|

  > **Note on accumulated bands:** Bands 6–12 use the `_hourly` suffixed
  > variants provided by the GEE Data team. These are computed as the
  > difference between consecutive forecast steps, avoiding the midnight
  > accumulation reset present in the base (non-suffixed) bands.


- **Target — Oya Quasi-Global Precipitation**

  - GEE Collection: `projects/global-precipitation-nowcast/assets/global_estimation`
  - Spatial resolution: 5,000 m
  - Temporal cadence: 30 minutes (aggregated to hourly for alignment)
  - Availability: 2004-01-01 to near real-time
  - Output band: `precipitation` (mm/hr)
  - License: CC-BY-4.0

  >**Known limitations:** Retrieval accuracy degrades at higher latitudes
  > and in arid or high-altitude regions (e.g., Sierra Madre Occidental,
  > northern Mexico). Oya has not yet undergone formal peer review; this
  > must be documented in the methodology. Contact: oya-team@google.com.

---
  
**Historical Time Range & Data Splits:**

The overlapping availability window between ERA5-Land Hourly (1950–present)
and Oya (2004–present) spans from **2004-01-01 to present**. The following
chronological splits are defined to prevent data leakage:

| Split      | Period                  | Duration   | Purpose                                                     |
| ---------- | ----------------------- | ---------- | ----------------------------------------------------------- |
| Training   | 2004-01-01 – 2016-12-31 | ~13 years  | Model learning; covers multiple ENSO cycles                 |
| Validation | 2017-01-01 – 2018-12-31 | 2 years    | Hyperparameter tuning; out-of-sample monitoring             |
| Test       | 2019-01-01 – 2020-07-09 | ~1.5 years | Final held-out evaluation; simulates operational deployment |

---
**Expectations:**

By the end of this step the following artifacts will exist on the dedicated
training server's local filesystem:

- Compressed NumPy archive files (`.npz`) organized by split, year, and
  month under a defined root directory
  (e.g., `/data/era5_oya_mexico/{split}/YYYY/MM/`), where `{split}` is
  one of `train`, `val`, or `test`.
- Each `.npz` file contains one aligned hourly sample with two arrays:
	  - `inputs`: shape `(1156, 3796, 13)` — the 13 ERA5-Land bands resampled
	    to 5 km over the expanded Atlantic and Pacific domain.
	  - `target`: shape `(1156, 3796, 1)` — the Oya precipitation band in mm/hr.
- A companion `metadata.json` file written alongside each `.npz`, recording
  the UTC timestamp, ERA5 image ID, Oya image ID, and alignment strategy
  used (top-of-hour snapshot or bracketed mean), to ensure full
  reproducibility.
- A single `dataset_index.csv` written at the dataset root upon completion
  of bulk export. It lists every sample path, its split assignment, UTC
  timestamp, and a validity flag (non-null pixel ratio ≥ 0.95). This index
  is the primary entry point for the `tf.data` pipeline in the training step.
- A single `norm_stats.json` written at the dataset root after the
  preprocessing step. It stores per-band mean and standard deviation
  computed exclusively from the training split (`2004–2016`).
  Data stored in `.npz` files is raw and unnormalized.

---

**Glossary:**

- **GEE:** Google Earth Engine — cloud-based geospatial analysis platform used for dataset querying and pixel-level data extraction.
- **ERA5-Land:** ECMWF land-component reanalysis at ~11 km / 1-hour resolution, produced by replaying ERA5 forcing over a land surface model.
- **Oya:** Google Research quasi-global precipitation estimate at 5 km / 30-minute resolution, derived from geostationary satellite VIS-IR channels.
- **Google Drive Intermediary:** A 5TB storage space used to buffer multi-terabyte GEE exports before they are processed locally, avoiding GCS costs.
- **export_gee_to_drive.py:** Python script that triggers and monitors background GEE export tasks.
- **download_and_convert.py:** Python script that uses the Google Drive API to download exported GeoTIFFs and convert them to the final `.npz` format.
- **NPZ:** NumPy compressed archive format (`.npz`). Stores multiple named arrays in a single gzip-compressed file.
- **Temporal alignment:** The process of matching Oya's 30-minute timestamps to ERA5-Land's hourly timestamps before pairing inputs and targets.
- **Accumulation reset:** ERA5-Land accumulated variables (precipitation, fluxes) reset to zero at midnight UTC each day. The `_hourly` disaggregated bands correct for this.