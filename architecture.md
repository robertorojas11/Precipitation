# Project Architecture: Saha & Ravela (2024)

## Overview
The project implements the **Saha & Ravela (2024)** statistical-physical downscaling pipeline for the Mexico and Central America domain. It downscales 25km ERA5 atmospheric states to 5km resolution using a 2-stage ESRGAN-style RRDB network enhanced by physical priors (Upslope and Spectral models) and manifold alignment (Conditional Gaussian Process).

## Comparative Experiment
Two parallel pipelines are maintained for evaluation:
- **Pipeline A (CHIRPS):** 5km target, station-blended ground truth.
- **Pipeline B (Oya):** 5km target, AI-derived 30-min nowcasting (aggregated to daily).

## Directory Structure
- `src/data_extraction/`: Data extraction and ingestion (`gee_extractor.py`, `drive_manager.py`, `npz_converter.py`).
- `src/data_preprocessing/`: Data normalization (`norm_calculator.py`) and physics model pre-computation (`physics_models.py`).
- `src/models/`: Neural network definitions in PyTorch (`rrdb_gan.py`).
- `src/training/`: Training loops and experiment management (`train_experiment.py`).
- `src/utils/`: Helper scripts and utilities for the downscaling process.
- `/mnt/data-r2/RobertoRojas/downscaling`: Local storage for `raw`, `processed`, and `shape_files` (moved from `./data`).
- `docs/`: Step-by-step documentation.

## Methodology Phases
1. **Extraction:** Querying ERA5, CHIRPS, Oya, and DEM from GEE to Google Drive, then local conversion to `.npz`.
2. **Physics & Statistics:** Pre-computing Upslope/Spectral orographic models and CGP manifold.
3. **Modeling:** Training Stage 1 (ERA5 to ERA5-Land) and Stage 2 (Super-resolution to 5km).
4. **Bias Correction:** Python-based GPD stochastic injection and optimal estimation.