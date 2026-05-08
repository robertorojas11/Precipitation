# Project Architecture

## Overview
The **Precipitation Downscaling** project aims to downscale 25km ERA5 atmospheric states to 5km probabilistic precipitation maps for the Mesoamerican cyclone basins (Pacific and Atlantic). The core relies on the OYA Architecture (deterministic U-Net) and FGN Framework (probabilistic noise injection).

## Directory Structure Changes
The project structure has evolved to clearly separate extraction, preprocessing, and modeling concerns:
- `src/data_extraction/`: Scripts for querying Google Earth Engine (`export_gee_to_drive.py`) and downloading results (`download_and_convert.py`). *(Previously named `src/data`)*
- `src/data_preprocessing/`: Responsible for ingesting `.npz` arrays into `tf.data.Dataset` pipelines. Contains logic for normalization (`compute_norm_stats.py`) and dataset slicing (`pipeline.py`).
- `src/models/`: Neural network definitions (OYA U-Net and FGN layer).
- `src/utils/`: Common utilities like configuration and centralized logging.
- `data/`: Local storage. Data extraction targets `./data/era5_oya_mesoamerica`. Contains shapefiles under `data/shape_files/`.

## Pipeline Implementation Details

### Step 1: Extraction
We extract a dynamic geographical bounding box (1156 x 3796 grid at 5km resolution) constructed from `atlantico_shp_grande.shp` and `pacifico_shp_grande.shp`. These are processed into `.npz` files locally.

### Step 2: Preprocessing
The preprocessing pipeline leverages TensorFlow (`tf.data.Dataset`) to avoid memory bottlenecks on the GPU server.
- **Normalization:** `compute_norm_stats.py` generates `norm_stats.json` for Z-scoring the 13 ERA5-Land bands.
- **Patching & Filtering:** The full images are split into 128x128 patches. To handle severe rain/no-rain imbalances, fully dry patches are filtered out of the training stream, and wet patches receive a higher sample weight (`w_wet`), configured in `pipeline_config.yaml`.