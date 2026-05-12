# Step 2: Data Preprocessing & Physics Models

**Description:**
Transforming raw `.npz` data into PyTorch-ready datasets for both Experimental Pipelines.

---

## 1. Branching the Pipeline
The preprocessing script must now handle two separate target distributions:
*   **Pipeline A:** ERA5 (0.25°) inputs paired with **CHIRPS** (5km) targets.
*   **Pipeline B:** ERA5 (0.25°) inputs paired with **Oya** (5km) targets.

## 2. Normalization Strategy
Standard Z-score normalization is applied to all input bands:
`x_norm = (x - μ) / σ`
*   Statistics are computed **independently** for the CHIRPS-aligned and Oya-aligned datasets to account for potential differences in pixel value distributions.
*   Normalization stats are saved to `norm_stats_chirps.json` and `norm_stats_oya.json`.

## 3. Physical Model Pre-computation
The physics-based channels (Upslope and Spectral) are computed once and stored. 
*   Since both targets are 5km, the physical models will be regridded to the 5km resolution to be used as input channels for the GANs.

## 4. PyTorch Dataset Configuration
The new scripts will implement a flexible `PrecipDataset` class that can load either CHIRPS or Oya targets based on a configuration flag.

| Config Parameter | Pipeline A | Pipeline B |
| ---------------- | ---------- | ---------- |
| `target_name`    | `chirps`   | `oya`      |
| `target_res`     | 5km        | 5km        |
| `temporal_scale` | Daily      | Daily      |

---

## Expected Artifacts
*   `data/processed/chirps_train.pt` / `data/processed/oya_train.pt`
*   `data/metadata/norm_stats_chirps.json`
*   `data/metadata/norm_stats_oya.json`