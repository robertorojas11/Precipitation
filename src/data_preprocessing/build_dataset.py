"""Build geospatially aligned, masked precipitation datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from src.data_preprocessing.physics import (
    compute_spectral_model,
    compute_terrain_gradients,
    compute_upslope_model,
    resample_dem,
)
from src.data_preprocessing.quality import (
    DEFAULT_MAX_DAILY_MM,
    DATASET_VERSION,
    ensure_finite_features,
    file_sha256,
    inspect_precipitation,
    manifest_sha256,
)
from src.utils.config import Config


HEIGHT, WIDTH = 460, 720
TARGET_CRS = "EPSG:4326"
TARGET_TRANSFORM = from_origin(-120.0, 35.0, 0.05, 0.05)
logger = Config.get_logger()


def _reproject_bands(path: str, resampling: Resampling) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as source:
        destination = np.full((source.count, HEIGHT, WIDTH), np.nan, dtype=np.float32)
        source_data = source.read(masked=True)
        for band in range(source.count):
            reproject(
                source=np.asarray(source_data[band].filled(np.nan), dtype=np.float32),
                destination=destination[band],
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=np.nan,
                dst_transform=TARGET_TRANSFORM,
                dst_crs=TARGET_CRS,
                dst_nodata=np.nan,
                resampling=resampling,
            )
    values = np.moveaxis(destination, 0, -1)
    return values, ensure_finite_features(values)


def _reproject_target(path: str) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as source:
        raw = source.read(1, masked=True)
        destination = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
        coverage = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        source_valid = (~np.ma.getmaskarray(raw)).astype(np.float32)
        source_values = np.asarray(raw.filled(np.nan), dtype=np.float32)
        reproject(
            source=source_values,
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=np.nan,
            dst_transform=TARGET_TRANSFORM,
            dst_crs=TARGET_CRS,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
        reproject(
            source=source_valid,
            destination=coverage,
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=TARGET_TRANSFORM,
            dst_crs=TARGET_CRS,
            resampling=Resampling.average,
        )
    return destination, coverage >= 0.999


def _load_slot_count(target_path: str) -> np.ndarray | None:
    path = Path(target_path)
    with rasterio.open(path) as source:
        if source.count >= 2:
            destination = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
            reproject(
                source=source.read(2),
                destination=destination,
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=source.nodata,
                dst_transform=TARGET_TRANSFORM,
                dst_crs=TARGET_CRS,
                dst_nodata=np.nan,
                resampling=Resampling.nearest,
            )
            return destination
    candidates = [
        path.with_name(path.stem + "_slot_count.tif"),
        path.with_name(path.stem.replace("oya_", "oya_slot_count_") + ".tif"),
    ]
    for candidate in candidates:
        if candidate.exists():
            values, _ = _reproject_bands(str(candidate), Resampling.nearest)
            return values[..., 0]
    return None


def _build_land_mask(elevation: np.ndarray, metadata_dir: Path) -> np.ndarray:
    """Use CHIRPS' stable land-only coverage rather than zero-valued DEM ocean."""
    chirps_index = pd.read_csv(metadata_dir / "dataset_index_chirps.csv")
    chirps_index = chirps_index[chirps_index["valid_flag"] == True].sort_values("date")
    for path in chirps_index["target_path"]:
        if isinstance(path, str) and Path(path).exists():
            _, coverage = _reproject_target(path)
            return coverage & np.isfinite(elevation) & (elevation > -100.0)
    raise RuntimeError("Cannot derive land mask: no valid CHIRPS source raster found")


def build_target(target_name: str, *, max_daily_mm: float, limit: int | None = None) -> dict:
    metadata_dir = Path(Config.LOCAL_DATA_DIR) / "metadata"
    source_index = pd.read_csv(metadata_dir / f"dataset_index_{target_name}.csv")
    source_index = source_index[source_index["valid_flag"] == True].sort_values("date")
    if limit is not None:
        source_index = source_index.head(limit)

    output_root = Path(Config.LOCAL_DATA_DIR) / DATASET_VERSION
    processed_root = output_root / "processed" / target_name
    output_metadata = output_root / "metadata"
    output_metadata.mkdir(parents=True, exist_ok=True)

    dem_path = Path(Config.RAW_DATA_DIR) / "dem" / "nasadem_mexico_1km.tif"
    elevation = resample_dem(str(dem_path), HEIGHT, WIDTH).astype(np.float32)
    land_mask = _build_land_mask(elevation, metadata_dir)
    dz_dx, dz_dy = compute_terrain_gradients(elevation)

    records: list[dict] = []
    total_records = len(source_index)
    logger.info("Building target=%s records=%d", target_name, total_records)
    for record_number, row in enumerate(source_index.itertuples(index=False), start=1):
        record = {"date": row.date, "split": row.split, "accepted": False}
        target_path = row.target_path
        if target_name == "oya":
            exported_candidate = (
                Path(Config.RAW_DATA_DIR) / DATASET_VERSION / "oya" /
                row.date[:4] / row.date[5:7] / f"oya_{row.date}.tif"
            )
            target_path = str(exported_candidate)
        required = [row.era5_path, row.era5_pl_path, target_path]
        missing = [path for path in required if not isinstance(path, str) or not Path(path).exists()]
        if missing:
            record["reject_reasons"] = ["missing_source_file"]
            records.append(record)
            continue

        target, source_target_valid = _reproject_target(target_path)
        slot_count = _load_slot_count(target_path) if target_name == "oya" else None
        if target_name == "oya" and slot_count is None:
            record["reject_reasons"] = ["missing_oya_slot_count"]
            records.append(record)
            continue
        if slot_count is not None:
            source_target_valid &= slot_count >= 30

        qc, precip_valid = inspect_precipitation(
            target,
            source_target_valid,
            max_daily_mm=max_daily_mm,
        )
        target_valid = precip_valid & land_mask
        record.update(qc.to_dict())
        if not qc.accepted or target_valid.sum() == 0:
            records.append(record)
            continue

        reused_features = False
        if isinstance(row.npz_path, str) and Path(row.npz_path).exists():
            with np.load(row.npz_path) as existing:
                if (
                    {"inputs", "upslope", "spectral"}.issubset(existing.files)
                    and existing["inputs"].shape == (HEIGHT, WIDTH, 18)
                ):
                    inputs = existing["inputs"].astype(np.float32)
                    upslope = existing["upslope"][..., 0].astype(np.float32)
                    spectral = existing["spectral"][..., 0].astype(np.float32)
                    input_valid = ensure_finite_features(inputs)
                    reused_features = True
        if not reused_features:
            surface, surface_valid = _reproject_bands(row.era5_path, Resampling.bilinear)
            pressure, pressure_valid = _reproject_bands(row.era5_pl_path, Resampling.bilinear)
            inputs = np.concatenate([surface, pressure], axis=-1).astype(np.float32)
            input_valid = surface_valid & pressure_valid
        if inputs.shape[-1] != 18:
            record["accepted"] = False
            record["reject_reasons"] = ["unexpected_input_channel_count"]
            records.append(record)
            continue

        # ERA5-Land total precipitation is exported in metres.
        inputs[..., 0] *= 1000.0
        if not reused_features:
            u_850, v_850, rh_850 = inputs[..., 13], inputs[..., 15], inputs[..., 17]
            upslope = compute_upslope_model(u_850, v_850, rh_850, dz_dx, dz_dy)
            spectral = compute_spectral_model(u_850, v_850, elevation)

        destination = processed_root / row.split / f"{row.date}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            target=np.where(target_valid, target, 0.0).astype(np.float32)[..., None],
            target_valid_mask=target_valid.astype(np.uint8)[..., None],
            input_valid_mask=input_valid.astype(np.uint8)[..., None],
            land_mask=land_mask.astype(np.uint8)[..., None],
            slot_count=(slot_count.astype(np.uint8)[..., None] if slot_count is not None else np.empty((0,), dtype=np.uint8)),
            date=row.date,
            crs=TARGET_CRS,
            transform=np.asarray(tuple(TARGET_TRANSFORM), dtype=np.float64),
            preprocessing_version=DATASET_VERSION,
        )
        if reused_features:
            payload["feature_source_npz"] = row.npz_path
        else:
            payload.update(
                inputs=inputs,
                upslope=upslope.astype(np.float32)[..., None],
                spectral=spectral.astype(np.float32)[..., None],
                elevation=elevation[..., None],
            )
        np.savez_compressed(destination, **payload)
        record["accepted"] = True
        record["npz_path"] = str(destination)
        record["target_sha256"] = file_sha256(target_path)
        records.append(record)
        if record_number % 100 == 0 or record_number == total_records:
            accepted_count = sum(bool(item.get("accepted")) for item in records)
            logger.info(
                "Build progress target=%s processed=%d/%d accepted=%d rejected=%d",
                target_name,
                record_number,
                total_records,
                accepted_count,
                len(records) - accepted_count,
            )

    frame = pd.DataFrame(records)
    index_path = output_metadata / f"dataset_index_{target_name}.csv"
    frame.to_csv(index_path, index=False)
    manifest_records = frame.fillna("").to_dict(orient="records")
    summary = {
        "preprocessing_version": DATASET_VERSION,
        "target": target_name,
        "source_index": str(metadata_dir / f"dataset_index_{target_name}.csv"),
        "records": len(frame),
        "accepted": int(frame.get("accepted", pd.Series(dtype=bool)).sum()),
        "rejected": int((~frame.get("accepted", pd.Series(dtype=bool))).sum()),
        "manifest_sha256": manifest_sha256(manifest_records),
        "max_daily_mm": max_daily_mm,
        "target_grid": {"height": HEIGHT, "width": WIDTH, "crs": TARGET_CRS, "transform": list(TARGET_TRANSFORM)},
    }
    with (output_metadata / f"manifest_{target_name}.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    logger.info("Build completed target=%s summary=%s", target_name, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=["chirps", "oya"])
    parser.add_argument("--max-daily-mm", type=float, default=DEFAULT_MAX_DAILY_MM)
    parser.add_argument("--limit", type=int, help="Build only the first N records for a smoke test")
    args = parser.parse_args()
    print(json.dumps(build_target(args.target, max_daily_mm=args.max_daily_mm, limit=args.limit), indent=2))


if __name__ == "__main__":
    main()
