# Complete Guide for the Precipitation Downscaling Pipeline in Mexico

This guide documents the entire statistical downscaling pipeline for precipitation, from downloading data via Google Earth Engine (GEE) to the final evaluation with stochastic bias correction.

---

## 1. Pipeline Architecture

The downscaling pipeline is structured into the following phases:
* **Phases 1-2 (Extraction & Alignment):** Satellite and reanalysis data download, regridding to common reference grids, masking the Mexican territory, and quality control.
* **Phases 3-4 (Physical Models):** Pre-computation of upslope topographic moisture flux ($W_{orographic} = \vec{V} \cdot \nabla z$) and Smith & Barstad (2004) linear spectral theory model in the Fourier domain.
* **Phases 5-6 (Generative Models - GANs):**
  * **GAN-1 (18 variables at 10 km):** ESRGAN (RRDB) for downscaling ERA5 (25 km) to ERA5-Land (10 km).
  * **GAN-2 (Topographic conditioning at 5 km):** ESRGAN (RRDB) that concatenates the GAN-1 output with the DEM and physical model fields to produce high-resolution precipitation estimates (5 km).
* **Phase 7 (Bias Correction):** Stochastic post-processing using empirical Quantile Mapping (QM), Generalized Pareto Distribution (GPD) tail correction for extremes, and spectral spatially-correlated noise injection using 2D FFT.

---

## 2. Environment Setup & Configuration

### Python Virtual Environment
Create and install the project dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables Config (`.env`)
Create a `.env` file in the root directory based on `.env.example`:
```ini
# Google Earth Engine & Google Cloud Authentication
EARTHENGINE_API_KEY=your_ee_api_key_here
GOOGLE_CLOUD_PROJECT_ID=precipitation-downscaling
GOOGLE_APPLICATION_CREDENTIALS=auth/gcp_credentials.json
GOOGLE_DRIVE_CREDENTIALS=auth/drive_credentials.json

# Storage & Paths
GCS_BUCKET_NAME=your_gcs_bucket_name
LOCAL_DATA_DIR=/mnt/data-r2/RobertoRojas/downscaling/era5_oya_mexico
```

---

## 3. Data Extraction (Phase 1)

Scripts in `src/data_extraction` download the satellite and reanalysis datasets (ERA5, ERA5-Land, CHIRPS, Oya, and DEM) for the Mexico bounding box domain (Lat 12°N – 35°N, Lon 84°W – 120°W) using Google Earth Engine API.

### 1. Google Earth Engine Authentication
Authenticate your GEE account on your local machine:
```bash
./venv/bin/python scripts/authenticate_gee.py
```

### 2. Run Data Extraction
Download ERA5 and the desired target dataset (`oya` or `chirps`):
```bash
# For Target Oya (Download from 2004 to present)
./venv/bin/python src/data_extraction/pipeline_runner.py --start_year 2004 --end_year 2026 --target oya

# For Target CHIRPS
./venv/bin/python src/data_extraction/pipeline_runner.py --start_year 2004 --end_year 2026 --target chirps
```

---

## 4. Data Preprocessing & Physical Models (Phases 2, 3, and 4)

The preprocessing pipeline aligns all variables to their respective grids (25 km, 10 km, and 5 km), applies the log-transform to precipitation (inverted by `expm1`), computes Z-score normalization and statistics, computes orographic wind fluxes, and saves the processed `.npz` files to the NAS.

### Preprocess and Compute Normalization Stats
```bash
# For Target Oya
./venv/bin/python src/data_preprocessing/preprocess_pipeline.py --target oya --calculate_stats

# For Target CHIRPS
./venv/bin/python src/data_preprocessing/preprocess_pipeline.py --target chirps --calculate_stats
```

### Fast Dataset Generation on Local NVMe SSD
To train the neural networks at maximum speed and avoid network latency bottlenecks when accessing the NAS, generate a local cache on the fast NVMe SSD drive:
```bash
# Generate fast dataset for Oya
./venv/bin/python scripts/prepare_fast_dataset.py --target oya

# Generate fast dataset for CHIRPS
./venv/bin/python scripts/prepare_fast_dataset.py --target chirps
```
This process caches the preprocessed samples in `data/fast_dataset/{target}/{split}/`, ready to be loaded by PyTorch's `FastPrecipDataset`.

---

## 5. Neural Network Training (Phases 5 and 6)

The production training script reads directly from the local SSD cache and runs on CUDA, using a combination of Pixel L1 loss, VGG Perceptual loss, and adversarial PatchGAN loss.

### 1. Train Oya Pipeline (20 Epochs)
To train the GAN-1 and GAN-2 generator models using the Oya target:
```bash
nohup ./venv/bin/python src/training/train_experiment.py --target oya --epochs 20 --batch_size 16 --device cuda > train_oya.log 2>&1 &
```
*Model weights and checkpoints will be saved in `/mnt/data-r2/RobertoRojas/downscaling/era5_oya_mexico/checkpoints/oya/`.*

### 2. Train CHIRPS Pipeline (50 Epochs)
To train the GAN-1 and GAN-2 generator models using the CHIRPS target:
```bash
nohup ./venv/bin/python src/training/train_experiment.py --target chirps --epochs 50 --batch_size 16 --device cuda > train_chirps.log 2>&1 &
```
*Model weights and checkpoints will be saved in `/mnt/data-r2/RobertoRojas/downscaling/era5_oya_mexico/checkpoints/chirps/`.*

---

## 6. Visualization & Pipeline Stages Comparison

You can generate a 2x2 comparison grid plot displaying the denormalized precipitation at different physical scale resolutions (Input ERA5 25km, GAN-1 10km, GAN-2 final 5km, and Ground Truth) for any validation sample index:

```bash
# Visualize validation sample index 25 for Oya
./venv/bin/python scripts/visualize_downscaling.py --target oya --sample_idx 25 --device cuda

# Visualize validation sample index 25 for CHIRPS
./venv/bin/python scripts/visualize_downscaling.py --target chirps --sample_idx 25 --device cuda
```
*The resulting comparison plots are saved directly to `outputs/oya/downscaling_comparison_oya.png` and `outputs/chirps/downscaling_comparison_chirps.png` respectively.*

---

## 7. Complete Evaluation & Bias Correction (Phase 7)

The evaluation script performs the following actions:
1. Runs inference on the validation split to fit the Quantile Mapping thresholds, GPD extreme tail parameters, and residual noise spatial covariance.
2. Saves the fitted parameters to `checkpoints/{target}/bias_corrector_{target}.npz`.
3. Runs inference on the held-out test split, applying the deterministic bias correction and generating a 10-member stochastic ensemble.
4. Computes deterministic and probabilistic metrics (MAE, RMSE, correlation, KS statistic, CSI thresholds, and CRPS) country-wide and specifically over the **Sierra Madre Occidental** bounding box subregion (`[100:300, 220:340]`).
5. Saves comparative tables, plots, and a 2D **Spatial Relative Bias Map** under the `outputs/` folder.

### Run Full Evaluation Pipeline
```bash
# Evaluate Oya models
./venv/bin/python src/training/evaluate_experiment.py --target oya

# Evaluate CHIRPS models
./venv/bin/python src/training/evaluate_experiment.py --target chirps
```

### Generated Output Files
All evaluation outputs are saved in their respective directories:
* **Metrics Summary JSON:** `outputs/{target}/metrics_{target}.json`
* **Spatial Relative Bias Map:** `outputs/{target}/downscaling_bias_{target}.png`
* **Validation Stage Comparison Map:** `outputs/{target}/downscaling_comparison_{target}.png`

---

## 8. Command Cheat-Sheet

```bash
# 1. Environment Setup
source venv/bin/activate

# 2. Data Download & Preprocess (CHIRPS example)
./venv/bin/python src/data_extraction/pipeline_runner.py --start_year 2004 --end_year 2026 --target chirps
./venv/bin/python src/data_preprocessing/preprocess_pipeline.py --target chirps --calculate_stats
./venv/bin/python scripts/prepare_fast_dataset.py --target chirps

# 3. Train Model (Background Execution)
nohup ./venv/bin/python src/training/train_experiment.py --target chirps --epochs 50 --batch_size 16 --device cuda > train_chirps.log 2>&1 &

# 4. Check Checkpoints & Run Bias Correction
./venv/bin/python scripts/visualize_downscaling.py --target chirps --sample_idx 25 --device cuda
./venv/bin/python src/training/evaluate_experiment.py --target chirps
```
