# Data, software, and compute reference

This appendix defines resources shared by multiple stages. Dataset-provider
metadata and licensing remain authoritative; this document records how the
repository uses each source.

## Atmospheric predictor channels

Prepared samples expect 18 atmospheric channels in the source archive's fixed
order.

ERA5-Land surface channels:

1. total precipitation (exported in m, converted to mm);
2. 2 m temperature;
3. 2 m dew-point temperature;
4. surface pressure;
5. 10 m zonal wind;
6. 10 m meridional wind;
7. downward surface solar radiation;
8. surface sensible heat flux;
9. surface latent heat flux;
10. volumetric soil water, layer 1.

ERA5 pressure-level channels:

11. temperature at 500 hPa;
12. temperature at 850 hPa;
13. zonal wind at 500 hPa;
14. zonal wind at 850 hPa;
15. meridional wind at 500 hPa;
16. meridional wind at 850 hPa;
17. relative humidity at 500 hPa;
18. relative humidity at 850 hPa.

The physics implementation uses zero-based concatenated indices 13, 15, and 17
for 850 hPa zonal wind, meridional wind, and relative humidity respectively.
Changing export order without versioning the index would silently change the
physical features and is prohibited.

## Targets

### CHIRPS

CHIRPS is a station-blended satellite precipitation product used as daily
ground truth. Its source coverage also contributes the stable study/land mask.
NaN outside coverage is invalid, not zero precipitation.

### Oya

Oya is a half-hourly global precipitation-nowcast rate product. It is aggregated
to daily mm with valid-slot mean × 24 and requires at least 30 slots per pixel.
It remains a separate experimental truth because its retrieval/model error
differs from station-blended CHIRPS.

## Static data

NASADEM supplies elevation. The pipeline derives elevation gradients, upslope
moisture lifting, and a Fourier-domain spectral orographic response. Elevation
is also used with CHIRPS coverage to exclude ocean/out-of-study cells.

## Spatial and temporal organization

The final grid has 331,200 cells per day. Prepared scales are:

| Nominal scale | Shape | Content |
|---|---:|---|
| 25 km | 92 × 144 | 18 atmospheric channels |
| 10 km | 230 × 360 | 3 physical/topographic channels and auxiliary target |
| 5 km | 460 × 720 | final target, masks, and ERA5 precipitation baseline |

The source index owns the train/validation/test assignment. Current experiment
governance treats 2020–2022 as validation and 2023–2025 as test, while earlier
dates form training; the stored index is the executable authority. Temporal
three-day context never crosses split boundaries.

## Storage hierarchy

```text
$RAW_DATA_DIR/
├── era5/
├── era5_pl/
├── chirps/
├── dem/
└── v2_clean/oya/

$LOCAL_DATA_DIR/
├── metadata/                 # source indexes
└── v2_clean/
    ├── metadata/             # QC indexes, manifests, stats, climatology
    └── processed/<target>/   # aligned mask-aware NPZ files

data/v2_clean/<target>/       # prepared train/val/test tensors
outputs/v2_clean/<target>/    # searches, runs, metrics, reports
logs/pipeline/<run-id>/       # orchestration and child-process logs
```

Network-backed raw/processed paths are checked before every pipeline run.
Prepared tensors are local to reduce random network I/O during training.

## Software resources

Direct dependencies are recorded in `requirements.txt`:

- Earth Engine and Google API clients for acquisition;
- NumPy and pandas for arrays, indexes, statistics, and manifests;
- rasterio for geospatial reprojection;
- PyTorch for datasets, model training, AMP, and inference;
- Matplotlib for noninteractive reports;
- requests/urllib3/httplib2 and Google authentication libraries for resilient
  remote operations;
- python-dotenv and python-dateutil for configuration and dates.

The pipeline runs from the repository virtual environment and captures the Git
revision/dirty state in training metadata.

## Compute resources

- Storage validation, QC, build, and statistics are CPU/I/O workloads.
- Prepared-cache generation uses PyTorch interpolation but does not require a
  GPU.
- Search, candidate, final training, evaluation, and reporting benefit from
  CUDA. If CUDA is requested but unavailable, training/evaluation fall back to
  CPU.
- `--batch-size` controls accelerator memory; 4 is the documented starting
  value. Reduce it if CUDA runs out of memory.
- `--num-workers` controls DataLoader subprocesses; 8 is the documented
  starting value. Reduce it if RAM, file descriptors, or network/local storage
  contention becomes limiting.
- The 64-wide model uses materially more memory and computation than width 32.
  Three-day context triples atmospheric input channels from 18 to 54.

## Provenance and leakage controls

- Raw targets are content-hashed with SHA-256.
- Dataset and prepared-record collections receive deterministic manifest hashes.
- Checkpoints embed the prepared manifest hash and normalization statistics.
- Ensemble members must share the same hash.
- Train-only quantities: normalization and monthly climatology.
- Validation-only decisions: hyperparameters, context length, early stopping,
  and best checkpoint.
- Test-only purpose: one frozen final evaluation and reporting.

## Known limitations

The ERA5 precipitation baseline is interpolated bilinearly rather than
conservatively remapped. The spectral prior uses domain-mean winds and fixed
time constants. The land mask is based on CHIRPS coverage and an elevation
threshold rather than a formal political polygon. Pooled pixel metrics contain
spatial dependence, partly addressed by month-block bootstrap uncertainty.
Current reports do not include FSS, geographic boundaries, reliability
diagrams, or elevation/season-stratified figures. These limitations must remain
visible when interpreting an R² result.
