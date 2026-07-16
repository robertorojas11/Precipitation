# Implementation Plan: Step 1 - Data Extraction & Acquisition

This plan covers the implementation of the data ingestion pipeline for the Saha & Ravela (2024) comparative experiment (CHIRPS vs. Oya).

## 1. Cleanup
- **[DELETE]** `main.py` (Outdated entry point)
- **[DELETE]** `tests/` (Old test suite if any - will be replaced)

## 2. Environment Setup
- Ensure `venv` is active and updated using `requirements.txt`.
- Configure `.env` based on `env.example` with valid GEE/GCP credentials.

## 3. Script Implementation (src/data/)

### A. GEE Extractor (`gee_extractor.py`)
- **Purpose:** Automate GEE exports for ERA5, CHIRPS, Oya, and NASADEM.
- **Key Features:**
    - Domain definition using the 9-point convex hull polygon.
    - Daily aggregation logic for ERA5 and Oya.
    - Task status monitoring via GEE API.

### B. Drive Manager (`drive_manager.py`)
- **Purpose:** Download GeoTIFFs from Google Drive.
- **Key Features:**
    - Google Drive API integration.
    - Directory organization: `data/raw/{dataset}/YYYY/MM/`.

### C. NPZ Converter (`npz_converter.py`)
- **Purpose:** Pair and resample raw data into `.npz` archives.
- **Key Features:**
    - Rasterio-based bilinear resampling of ERA5 (0.25°) to 5km target grids.
    - Creation of `dataset_index_{target}.csv` for Pipeline A and B.

## 4. Test Scripts (Proof of Concept)

Prior to full implementation, the following tests will be executed:
- `tests/test_gee_auth.py`: Verifies GEE initialization and polygon validity.
- `tests/test_gee_sampling.py`: Triggers a 1-day export for ERA5, CHIRPS, and Oya to verify aggregation logic.
- `tests/test_raster_alignment.py`: Loads one downloaded ERA5 and Target TIF to verify spatial overlap and CRS.

## 5. Success Metrics
- **Extraction Completeness:** 100% of requested days (2004-2025) exported and downloaded.
- **Data Alignment:** 0-pixel offset between ERA5 resampled grid and target grids.
- **Storage Efficiency:** `.npz` files use compressed format, keeping total dataset size within local disk limits.
- **Reproducibility:** `dataset_index.csv` correctly maps every date to its corresponding `.npz` file.

## User Review Required

> [!IMPORTANT]
> 1. **GEE Project ID:** Do you have a specific Google Cloud Project ID configured for GEE billing/tasks?
> 2. **Storage Limit:** Do you have enough local disk space (est. 100-200GB) for the raw GeoTIFFs and processed `.npz` files?
> 3. **Drive Folder:** Should I create a specific folder name in your Google Drive for these exports?

## Open Questions
- Should the `npz_converter.py` handle ocean masking using the shapefiles during this step, or should that be deferred to the training dataloader?
