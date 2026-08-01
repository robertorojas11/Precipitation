# Project architecture

This repository contains one reproducible deterministic pipeline for daily
precipitation downscaling. CHIRPS and Oya are separate ground truths that share
the same data contract, model, training procedure, and evaluation protocol.

## Active modules

- `src/data_extraction/`: export Oya with valid-slot coverage and synchronize
  source rasters.
- `src/data_preprocessing/build_dataset.py`: align raw inputs, compute physical
  terrain features, apply quality control, and write versioned samples.
- `src/data_preprocessing/prepare_dataset.py`: calculate train-only
  normalization statistics, climatology, and local training tensors.
- `src/data_preprocessing/validate_dataset.py`: enforce raw, processed, and
  prepared-data acceptance gates.
- `src/models/multiscale_unet.py`: deterministic multiscale occurrence/amount
  U-Net.
- `src/training/train.py`: masked training, validation, checkpointing, and
  provenance capture.
- `src/training/search.py`: fixed validation-only model search.
- `src/training/evaluate.py`: held-out evaluation, baselines, event scores,
  yearly scores, and month-block confidence intervals.

## Data flow

```text
raw rasters -> QC/alignment -> versioned samples -> prepared tensors
                                               -> train/validation search
                                               -> frozen test evaluation
```

The active dataset version is `v2_clean`. The name remains stable so existing
clean artifacts and useful checkpoints retain provenance. Older checkpoints
are historical baselines only and are not load-compatible with the current
model.

## Scientific contract

- Invalid pixels never enter losses or metrics.
- Normalization statistics use the training split only.
- Model selection uses 2020–2022 validation data.
- The 2023–2025 test split is evaluated after configurations and seeds freeze.
- CHIRPS and Oya have independent statistics, manifests, checkpoints, and
  acceptance results.
- The target is pooled test R² >= 0.40 for each ground truth, while also beating
  ERA5 and climatology and passing yearly and bootstrap gates.

This project does not claim an ESRGAN, conditional Gaussian process, or
probabilistic model. Those legacy implementations and unsupported claims were
removed.
