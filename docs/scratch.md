# GEE Data Extraction Discoveries (Scratch Investigations)

This document summarizes the technical findings from the diagnostic scripts used to debug the Earth Engine extraction pipeline.

## 1. ERA5-Land Surface Data
*   **Asset**: `ECMWF/ERA5_LAND/HOURLY`
*   **Availability**: Confirmed 24 hourly images per day exist for problematic periods (e.g., April 2004).
*   **Bands**: Verified availability of all requested surface bands:
    *   `total_precipitation_hourly`
    *   `temperature_2m`, `dewpoint_temperature_2m`
    *   `surface_pressure`
    *   `u_component_of_wind_10m`, `v_component_of_wind_10m`
    *   `surface_solar_radiation_downwards_hourly`
    *   `surface_sensible_heat_flux_hourly`, `surface_latent_heat_flux_hourly`
    *   `volumetric_soil_water_layer_1`
*   **Finding**: Data is present and individual images are healthy. "Error in map" during reduction was likely due to the large area clipping or mismatched atmospheric band requests in the same task.

## 2. ERA5 Atmospheric/Pressure Levels
*   **Asset**: `ECMWF/ERA5/HOURLY` (Global)
*   **Mistake**: The pipeline was initially looking in `ECMWF/ERA5/DAILY`, which only contains surface aggregates.
*   **Band Naming**: In GEE's public catalog, atmospheric levels are **not** separate metadata properties. They are baked into band names using an `hPa` suffix.
*   **Available Levels**: Discovered that the public `ECMWF/ERA5/HOURLY` collection **only includes two levels**:
    *   `500hPa`
    *   `850hPa`
*   **Correction**: Updated the pipeline to request `temperature_500hPa`, `u_component_of_wind_500hPa`, etc., instead of the 9 levels originally planned (1000, 925, 850, 700, 600, 500, 400, 300, 200).

## 3. Geometric & Spatial Clipping
*   **Centroid Check**: Diagnostic `reduceRegion` calls at the center of the `DOMAIN_POLYGON` returned `None`.
*   **Reason**: The study area's center is in the Pacific Ocean. ERA5-Land (Surface) only has data over land.
*   **Resolution**: Clipping and exports are safe as long as the polygon contains the Mexican landmass, which it does.

## 4. Authentication Flow
*   **Constraint**: Remote server (hurakan) lacks `gcloud` and cannot open browsers.
*   **Discovery**: The standard `earthengine authenticate` command fails because it defaults to `gcloud` mode.
*   **Solution**: Use `ee.Authenticate(auth_mode='notebook')` which uses the `code.earthengine.google.com` flow, generating a persistent token in `~/.config/earthengine/credentials`.

## 5. Script Summary
| Script | Purpose | Outcome |
| :--- | :--- | :--- |
| `check_era5_gap.py` | Verify 24h coverage | Success: 24/24 images found. |
| `check_bands.py` | Verify band names | Success: Surface bands confirmed. |
| `find_pressure_bands.py` | Search atmospheric data | Discovery: Only 500/850hPa available. |
| `inspect_all_props.py` | Check metadata properties | Outcome: No `level` property; used band names. |
| `reproduce_failure.py` | Get raw error from GEE | Outcome: Identified band mismatch error. |
