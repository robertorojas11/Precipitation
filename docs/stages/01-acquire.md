# Stage 1 — Data acquisition

## Data sources

The model consumes daily atmospheric predictors, topography, and one of two
ground truths:

- ERA5-Land hourly surface fields, including total precipitation and near-surface
  thermodynamic, pressure, wind, radiation, heat-flux, and soil-water variables.
- ERA5 pressure-level temperature, wind, and relative humidity at 500 and
  850 hPa.
- NASADEM elevation.
- CHIRPS daily precipitation as the station-blended target.
- Google Oya global precipitation nowcast rates as the alternative target.

The current CHIRPS, ERA5, pressure-level, and DEM files are reused through
`$LOCAL_DATA_DIR/metadata/dataset_index_chirps.csv`; therefore acquisition is
reported as not applicable for CHIRPS. Oya is re-exported because the old daily
aggregation admitted missing scans and fill values.

## Oya daily aggregation

Oya provides half-hourly precipitation rates in mm/hour. For each pixel, valid
slots satisfy finite, nonnegative, non-sentinel conditions. If valid rates are
(r_t) and their count is (n), the daily estimate is

\[
P_{day} = \frac{1}{n}\sum_{t=1}^{n} r_t \times 24.
\]

This is mean rate × 24 hours, not a sum that treats missing scans as zero.
Coverage is accepted only when (n \ge 30) of the nominal 48 daily slots.
The export contains two bands:

1. `precipitation` in mm/day;
2. `slot_count`, the number of valid half-hourly observations.

Pixels below the coverage threshold remain masked. GeoTIFF nodata is −9999,
the CRS is EPSG:4326, nominal scale is 5,566 m, and cloud-optimized output is
requested.

## Execution and resilience

```bash
python -m src.data_extraction.export_oya --start-year 2004 --end-year 2025
```

Earth Engine uses user OAuth when possible because Drive exports need user
storage quota, with service-account fallback for initialization. Tasks run in
batches of 20. The process polls terminal task states, fails on
`FAILED`/`CANCELLED`, downloads completed GeoTIFFs from Drive, verifies local
presence, and moves downloaded Drive objects to trash. Months already present
locally are skipped, making acquisition resumable.

Files are organized as
`$RAW_DATA_DIR/v2_clean/oya/YYYY/MM/oya_YYYY-MM-DD.tif`.

## Connection to raw validation

Acquisition only establishes that files were exported. It does not establish
scientific usability. The next stage independently checks presence, slot-count
coverage, geographic alignment, finite values, sentinels, physical range, and
split parity. Training never consumes an acquisition result directly.
