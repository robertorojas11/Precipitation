# Precipitation Downscaling (Mexico)

## Project Overview
- **Goal:** Downscale 25km ERA5 atmospheric states to 5km probabilistic precipitation maps for Mexico (Lat 12°N-35°N, Lon 84°W-120°W).
- **Core Models:** Saha & Ravela (2024) — ESRGAN-style RRDB GAN with Conditional Gaussian Process (CGP) and orographic physics priors.

## Architecture & Code Boundaries
- **Data Ingestion:** GEE extraction to **Google Drive**. Local Python scripts download GeoTIFFs and convert to **.npz** files under `data/processed/`.
- **Preprocessing:** Z-score normalization per variable using training-split statistics. No spatial patching — full-domain grids.
- **Physical Models:**
  - **Upslope Model:** Clausius-Clapeyron forced lifting over ERA5 pressure levels.
  - **Spectral Model:** FFT-based linear theory of orographic precipitation.
- **Statistical Model:**
  - **CGP (Conditional Gaussian Process):** KD-Tree manifold alignment (k=5) of precipitation fields.
- **GAN Architecture:**
  - **GAN-1:** ERA5 (0.25°) → ERA5-Land (0.1°) using RRDB + PixelShuffle upsampling.
  - **GAN-2:** Upscaled target (0.1°) → final target (5km) using RRDB.
- **Bias Correction:** Stochastic injection + GPD optimal estimation + back projection (Python).

## Comparative Experiment
Two parallel pipelines are being trained and evaluated:
- **Pipeline A (CHIRPS):** Uses CHIRPS (station-blended, 5km daily) as the high-resolution target.
- **Pipeline B (Oya):** Uses Google's Oya dataset (AI-derived, 5km 30-min) aggregated to daily as the target.

## Development Workflow
1. Extract ERA5, CHIRPS, and Oya from GEE → Google Drive (`src/data/gee_extractor.py`).
2. Download GeoTIFFs and convert to `.npz` (`src/data/drive_downloader.py`, `src/data/npz_converter.py`).
3. Compute per-band normalization statistics for each pipeline (`src/preprocessing/norm_calculator.py`).
4. Pre-compute physics channels: Upslope & Spectral (`src/preprocessing/physics_models.py`).
5. Train GAN-1 and GAN-2 separately per pipeline using `train_experiment.py --target [chirps|oya]`.
6. Evaluate: CSI, KS statistic, ECDF bias, NMI over the Sierra Madre Occidental terrain.

## Setup & Quirks
- Auth credentials are set in `.env` (see `.env.example`). Never commit the actual `.env`.
- The Mexico shapefile is at `data/shape_files/` and used for evaluation masking, not training filtering.
- Time range is constrained to **2004–2025** due to Oya's earliest availability (2004).
- CUDA device index must be updated per training machine.