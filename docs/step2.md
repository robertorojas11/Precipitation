### Data Preprocessing and Normalization Pipeline

**Description:**

The second step of the pipeline transforms the raw `.npz` archives produced in Step 1 into a fully normalized, patch-based `tf.data.Dataset` ready for GPU training. The pipeline operates in two sequential stages.

In the first stage, a normalization pass reads `norm_stats.json` from the dataset root — which stores the per-band mean and standard deviation computed exclusively over the training split (2004–2016) — and applies Z-score normalization to each of the 13 ERA5-Land input bands. The target Oya precipitation band is left in its original mm/hr units; loss scaling is handled at training time. Normalized values are never written back to disk; normalization is applied on-the-fly within the `tf.data` graph to avoid storage overhead and keep raw data fully reproducible.

In the second stage, the normalized spatial fields (1156 × 3796 for inputs, 1156 × 3796 for the target) are sliced into non-overlapping 128 × 128 pixel patches using a deterministic stride. Any edge tiles that cannot be filled to 128 × 128 are zero-padded and masked. A patch-level validity check is then applied: patches where the Oya target contains fewer than 5% non-null pixels are dropped via a dataset filter, consistent with the 0.95 non-null threshold established in `dataset_index.csv`. For the training split only, an additional precipitation filter discards patches where the target band is uniformly zero (i.e. fully dry patches), as a primary strategy for addressing the severe rain/no-rain class imbalance. Dry patches are retained in the validation and test splits to preserve realistic evaluation conditions.

A sample-weight scalar is attached to each surviving patch at yield time. Patches containing at least one pixel with precipitation ≥ 1 mm/hr receive a weight of `w_wet`; all others receive `w_dry = 1.0`. The ratio `w_wet / w_dry` is a tunable hyperparameter exposed in the pipeline configuration.

---
**Normalization:**
Z-score normalization is applied per band using statistics from `norm_stats.json`:

$$x' = \frac{x - \mu_b}{\sigma_b}$$

where μ_b and σ_b are the training-split mean and standard deviation for band b. Statistics are loaded once at pipeline initialization and broadcast as constants into the `tf.data` graph. If `norm_stats.json` is absent, the pipeline raises a `FileNotFoundError` at startup rather than silently computing statistics on-the-fly (which would risk train/val leakage).

|**Band**|**Physical range (approx.)**|**Normalization**|
|---|---|---|
|`temperature_2m`|250–320 K|Z-score|
|`dewpoint_temperature_2m`|230–310 K|Z-score|
|`surface_pressure`|50,000–105,000 Pa|Z-score|
|`u_component_of_wind_10m`|−30 to +30 m/s|Z-score|
|`v_component_of_wind_10m`|−30 to +30 m/s|Z-score|
|`total_precipitation_hourly`|0–0.05 m|Z-score|
|`runoff_hourly`|0–0.02 m|Z-score|
|`surface_runoff_hourly`|0–0.02 m|Z-score|
|`surface_solar_radiation_downwards_hourly`|0–4,000,000 J/m²|Z-score|
|`surface_net_solar_radiation_hourly`|−500,000–4,000,000 J/m²|Z-score|
|`surface_sensible_heat_flux_hourly`|−500,000–2,000,000 J/m²|Z-score|
|`surface_latent_heat_flux_hourly`|−500,000–2,000,000 J/m²|Z-score|
|`volumetric_soil_water_layer_1`|0.0–0.8 vol. fraction|Z-score|

> **Note on target normalization:** The Oya precipitation target (`mm/hr`) is not Z-score normalized. The U-Net's final activation is a ReLU ensuring non-negative output, and the loss function operates directly in mm/hr. This avoids the need to invert normalization during inference and preserves interpretability of validation metrics (e.g. MAE in mm/hr).

---
**Patching strategy:**

The 1156 × 3796 input and target grids are partitioned into non-overlapping 128 × 128 patches with a stride of 128 pixels. This yields a maximum of ⌈1156/128⌉ × ⌈3796/128⌉ = 10 × 30 = 300 candidate patches per hourly sample before filtering. Edge patches that extend beyond the spatial boundary are zero-padded; a binary validity mask of shape `(128, 128, 1)` is yielded alongside each patch, indicating padded pixels. The U-Net loss is computed only over unmasked pixels.

|**Dimension**|**Value**|
|---|---|
|Input patch shape|`(128, 128, 13)`|
|Target patch shape|`(128, 128, 1)`|
|Mask shape|`(128, 128, 1)`|
|Max patches per sample (pre-filter)|300|
|Stride|128 px (non-overlapping)|
  
---
**Rain/no-rain imbalance strategy:**

Dry pixels (0 mm/hr) constitute the large majority of all patches across the Mexico domain, particularly over arid northern regions and during the dry season. A two-pronged strategy is applied exclusively during training:

1. **Dry-patch filtering.** Patches where the entire 128 × 128 target tile is uniformly zero are removed from the training stream via a `tf.data.Dataset.filter` call. This is the primary imbalance correction. Patches with mixed wet/dry content are always retained, preserving meteorological spatial context around precipitation cells.

2. **Precipitation-weighted loss.** A per-patch sample weight scalar is passed to the model's `fit()` call. Wet patches (containing ≥ 1 pixel with precipitation ≥ 1 mm/hr) receive weight `w_wet` (default: 5.0); remaining mixed patches receive weight 1.0. This biases gradient updates toward precipitation events without discarding valid boundary-condition information. `w_wet` is exposed as a configurable hyperparameter in `pipeline_config.yaml`.


> **Validation and test behavior:** Neither dry-patch filtering nor sample weighting is applied during validation or test evaluation. All patches — including fully dry tiles — are passed through the pipeline unchanged so that evaluation metrics reflect operational distribution.

---
**Expectations:**

By the end of this step the following artifacts and runtime objects will be available:

- A `build_dataset(split, config)` function in `data/pipeline.py` that accepts a split name (`train`, `val`, or `test`) and a config object, and returns a fully prefetched `tf.data.Dataset` yielding batches of shape `(B, 128, 128, 13)` for inputs, `(B, 128, 128, 1)` for targets, `(B, 128, 128, 1)` for masks, and `(B,)` for sample weights.
- The dataset is sourced from `dataset_index.csv` at the root directory defined in Step 1 (`/data/era5_oya_mexico/`). Only samples with `valid_flag = True` (non-null pixel ratio ≥ 0.95) are loaded.
- Normalization statistics are read from `norm_stats.json` at pipeline initialization. The pipeline raises `FileNotFoundError` if this file is absent, rather than computing statistics on-the-fly.
- All preprocessing operations (normalization, patching, filtering, masking) execute within the `tf.data` graph using vectorized ops, ensuring the pipeline is not a training bottleneck on the dedicated GPU server.
- A `compute_norm_stats.py` script that reads training-split `.npz` files, computes per-band mean and standard deviation, and writes `norm_stats.json`. This script is idempotent and must be run once before pipeline initialization if `norm_stats.json` does not already exist.
- A `pipeline_config.yaml` file at the project root exposing all tunable pipeline parameters: `batch_size`, `w_wet`, `wet_threshold_mm` (the per-pixel precipitation threshold used to classify a patch as wet), `prefetch_buffer`, and `num_parallel_calls`.

---
**Glossary:**

- **Z-score normalization:** Linear rescaling of a variable by subtracting its mean and dividing by its standard deviation, so that the normalized output has zero mean and unit variance across the training distribution.
- **Patching:** Partitioning the full-resolution 1156 × 3796 spatial grid into non-overlapping 128 × 128 tiles that fit within GPU memory during training.
- **Dry-patch filter:** A `tf.data.Dataset.filter` operation that removes patches whose entire target tile is uniformly zero mm/hr, reducing the dominance of no-rain samples in the training stream.
- **Sample weight:** A scalar multiplier attached to each training patch that biases the gradient update toward precipitation events without discarding non-precipitating context patches.
- **Validity mask:** A binary `(128, 128, 1)` tensor indicating zero-padded pixels at spatial domain edges; loss is computed only over unmasked pixels.
- **`norm_stats.json`:** JSON file at the dataset root storing per-band mean and standard deviation computed from the training split (2004–2016). Produced by `compute_norm_stats.py`; consumed by `build_dataset()` at pipeline initialization.
- **`pipeline_config.yaml`:** YAML configuration file exposing all tunable pipeline hyperparameters, including `w_wet`, `wet_threshold_mm`, `batch_size`, and `prefetch_buffer`.
- **`tf.data.Dataset`:** TensorFlow's lazy evaluation dataset abstraction; all preprocessing ops execute in the graph to maximize GPU utilization and avoid Python-side bottlenecks.
- **`dataset_index.csv`:** The primary entry point for the pipeline, produced in Step 1. Lists every sample path, its split assignment, UTC timestamp, and validity flag. Only samples with `valid_flag = True` are loaded.