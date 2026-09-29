# Stage 0 — Storage preflight

Reused unchanged from the legacy pipeline: `src/utils/storage.py` has no
dependency on the daily 460×720 grid or on CHIRPS/Oya, so the v2-generative
framework calls the exact same module. Full documentation:
`docs/stages/00-storage-check.md`.

```bash
python -m src.utils.storage --minimum-free-gib 5
```

The only difference in practice is *which* directories matter for this
framework: `V2_RAW_DATA_DIR` (raw ERA5/IMERG GeoTIFFs,
`$RAW_DATA_DIR/v2_generative` by default) and `LOCAL_DATA_DIR` (prepared
tensors, under `<LOCAL_DATA_DIR>/v2_generative_hourly/`, see
[Build / alignment](02-build.md)) — both on the same network-mounted
`/mnt/data-r2` filesystem as the legacy pipeline's directories, so the same
preflight probe covers them.
