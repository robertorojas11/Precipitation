"""Calculate masked statistics and build versioned training tensors."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data_preprocessing.quality import DATASET_VERSION, manifest_sha256
from src.utils.config import Config

logger = Config.get_logger()

def _paths(target: str) -> tuple[Path, Path, Path]:
    root = Path(Config.LOCAL_DATA_DIR) / DATASET_VERSION
    return (
        root,
        root / "metadata" / f"dataset_index_{target}.csv",
        Path("data") / DATASET_VERSION / target,
    )


def _accepted_index(target: str) -> pd.DataFrame:
    _, index_path, _ = _paths(target)
    frame = pd.read_csv(index_path)
    return frame[frame["accepted"] == True].sort_values("date").reset_index(drop=True)


def _load_features(data) -> np.ndarray:
    if "feature_source_npz" in data.files:
        source_path = str(np.asarray(data["feature_source_npz"]).item())
        with np.load(source_path) as source:
            features = np.concatenate(
                [source["inputs"], source["upslope"], source["spectral"], source["elevation"]],
                axis=-1,
            )
            features = features.copy()
            features[..., 0] *= 1000.0
            return features
    return np.concatenate(
        [data["inputs"], data["upslope"], data["spectral"], data["elevation"]],
        axis=-1,
    )


def _update_moments(values: np.ndarray, mask: np.ndarray, sums, squares, counts) -> None:
    for channel in range(values.shape[-1]):
        selected = values[..., channel][mask & np.isfinite(values[..., channel])]
        if selected.size:
            sums[channel] += selected.astype(np.float64).sum()
            squares[channel] += np.square(selected.astype(np.float64)).sum()
            counts[channel] += selected.size


def compute_stats(target: str) -> dict:
    root, _, _ = _paths(target)
    frame = _accepted_index(target)
    frame = frame[frame["split"] == "train"]
    input_sum = np.zeros(21, dtype=np.float64)
    input_sq = np.zeros(21, dtype=np.float64)
    input_count = np.zeros(21, dtype=np.int64)
    target_sum = target_sq = 0.0
    target_count = 0

    logger.info("Calculating statistics target=%s training_samples=%d", target, len(frame))
    for index, row in enumerate(frame.itertuples(index=False), start=1):
        with np.load(row.npz_path) as data:
            features = _load_features(data).astype(np.float64)
            target_values = data["target"][..., 0].astype(np.float64)
            target_mask = data["target_valid_mask"][..., 0].astype(bool)
            input_mask = data["input_valid_mask"][..., 0].astype(bool)
            land_mask = data["land_mask"][..., 0].astype(bool)

        features[..., 0] = np.log1p(np.maximum(features[..., 0], 0.0))
        transformed_target = np.log1p(np.maximum(target_values, 0.0))
        _update_moments(features, input_mask & land_mask, input_sum, input_sq, input_count)
        selected = transformed_target[target_mask]
        target_sum += selected.sum(dtype=np.float64)
        target_sq += np.square(selected, dtype=np.float64).sum(dtype=np.float64)
        target_count += selected.size
        if index % 100 == 0 or index == len(frame):
            logger.info("Statistics progress target=%s samples=%d/%d", target, index, len(frame))

    if not target_count or np.any(input_count == 0):
        raise RuntimeError("Cannot calculate statistics: one or more channels have no valid data")
    input_mean = input_sum / input_count
    input_std = np.sqrt(np.maximum(input_sq / input_count - np.square(input_mean), 0.0))
    input_std[input_std < 1e-8] = 1.0
    target_mean = target_sum / target_count
    target_std = float(np.sqrt(max(target_sq / target_count - target_mean**2, 0.0)))
    if target_std < 1e-8:
        target_std = 1.0

    stats = {
        "preprocessing_version": DATASET_VERSION,
        "target_name": target,
        "log_transform_precip": True,
        "processed_samples": len(frame),
        "input_mean": input_mean.tolist(),
        "input_std": input_std.tolist(),
        "target_mean": float(target_mean),
        "target_std": target_std,
    }
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    with (metadata / f"norm_stats_{target}.json").open("w") as stream:
        json.dump(stats, stream, indent=2)
    return stats


def _masked_area_downsample(values: torch.Tensor, mask: torch.Tensor, size) -> tuple[torch.Tensor, torch.Tensor]:
    numerator = F.interpolate(values * mask, size=size, mode="area")
    denominator = F.interpolate(mask, size=size, mode="area")
    result = numerator / denominator.clamp_min(1e-6)
    valid = denominator >= 0.999
    return torch.where(valid, result, torch.zeros_like(result)), valid


def build_fast_cache(target: str) -> dict:
    root, _, fast_root = _paths(target)
    frame = _accepted_index(target)
    with (root / "metadata" / f"norm_stats_{target}.json").open() as stream:
        stats = json.load(stream)
    input_mean = torch.tensor(stats["input_mean"], dtype=torch.float32).view(21, 1, 1)
    input_std = torch.tensor(stats["input_std"], dtype=torch.float32).view(21, 1, 1)
    target_mean = float(stats["target_mean"])
    target_std = float(stats["target_std"])
    records: list[dict] = []

    logger.info("Building prepared cache target=%s samples=%d", target, len(frame))
    for index, row in enumerate(frame.itertuples(index=False), start=1):
        with np.load(row.npz_path) as data:
            features = _load_features(data)
            target_values = data["target"][..., 0]
            target_mask_np = data["target_valid_mask"][..., 0].astype(bool)
            input_mask_np = data["input_valid_mask"][..., 0].astype(bool)
            land_mask_np = data["land_mask"][..., 0].astype(bool)

        features_t = torch.from_numpy(features.astype(np.float32)).permute(2, 0, 1)
        features_t[0] = torch.log1p(features_t[0].clamp_min(0.0))
        era5_log_precip = torch.where(
            torch.from_numpy(input_mask_np & land_mask_np).unsqueeze(0),
            features_t[0:1],
            torch.zeros_like(features_t[0:1]),
        )
        features_t = (features_t - input_mean) / input_std
        input_valid = torch.from_numpy((input_mask_np & land_mask_np)).unsqueeze(0)
        features_t = torch.where(input_valid, features_t, torch.zeros_like(features_t))

        target_t = torch.from_numpy(target_values.astype(np.float32)).unsqueeze(0)
        target_t = (torch.log1p(target_t.clamp_min(0.0)) - target_mean) / target_std
        target_valid = torch.from_numpy(target_mask_np).unsqueeze(0)
        target_t = torch.where(target_valid, target_t, torch.zeros_like(target_t))

        input_25 = F.interpolate(features_t[:18].unsqueeze(0), size=(92, 144), mode="bilinear", align_corners=False)[0]
        input_mask_25 = F.interpolate(input_valid.float().unsqueeze(0), size=(92, 144), mode="nearest")[0].bool()
        physics_10 = F.interpolate(features_t[18:].unsqueeze(0), size=(230, 360), mode="bilinear", align_corners=False)[0]
        target_10, mask_10 = _masked_area_downsample(
            target_t.unsqueeze(0), target_valid.float().unsqueeze(0), (230, 360)
        )
        era5_precip_5 = F.interpolate(
            era5_log_precip.unsqueeze(0), size=(460, 720), mode="bilinear", align_corners=False
        )[0]
        era5_precip_5_norm = (era5_precip_5 - target_mean) / target_std
        parsed_date = datetime.strptime(row.date, "%Y-%m-%d")
        phase = 2.0 * np.pi * (parsed_date.timetuple().tm_yday - 1) / 365.2425

        destination = fast_root / row.split / f"{row.date}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            inputs_25km=input_25.numpy(),
            input_valid_mask_25km=input_mask_25.numpy().astype(np.uint8),
            phys_dem_10km=physics_10.numpy(),
            real_10km=target_10[0].numpy(),
            target_valid_mask_10km=mask_10[0].numpy().astype(np.uint8),
            real_5km=target_t.numpy(),
            target_valid_mask_5km=target_valid.numpy().astype(np.uint8),
            land_mask_5km=torch.from_numpy(land_mask_np).unsqueeze(0).numpy().astype(np.uint8),
            era5_precip_5km_norm=era5_precip_5_norm.numpy(),
            season=np.asarray([np.sin(phase), np.cos(phase)], dtype=np.float32),
            date=row.date,
        )
        records.append({"date": row.date, "split": row.split, "path": str(destination)})
        if index % 100 == 0 or index == len(frame):
            logger.info("Cache progress target=%s samples=%d/%d", target, index, len(frame))

    manifest = {
        "preprocessing_version": DATASET_VERSION,
        "target": target,
        "records": len(records),
        "manifest_sha256": manifest_sha256(records),
        "split_counts": pd.Series([record["split"] for record in records]).value_counts().to_dict(),
    }
    with (root / "metadata" / f"fast_manifest_{target}.json").open("w") as stream:
        json.dump(manifest, stream, indent=2)
    return manifest


def build_monthly_climatology(target: str) -> dict:
    root, _, fast_root = _paths(target)
    with (root / "metadata" / f"norm_stats_{target}.json").open() as stream:
        stats = json.load(stream)
    sums = np.zeros((12, 460, 720), dtype=np.float64)
    counts = np.zeros((12, 460, 720), dtype=np.uint32)
    files = sorted((fast_root / "train").glob("*.npz"))
    logger.info("Building climatology target=%s training_samples=%d", target, len(files))
    for index, path in enumerate(files, start=1):
        month = int(path.stem[5:7]) - 1
        with np.load(path) as data:
            target_norm = data["real_5km"][0].astype(np.float64)
            valid = data["target_valid_mask_5km"][0].astype(bool)
        target_mm = np.maximum(
            np.expm1(target_norm * stats["target_std"] + stats["target_mean"]), 0.0
        )
        sums[month] += np.where(valid, target_mm, 0.0)
        counts[month] += valid
        if index % 100 == 0 or index == len(files):
            logger.info("Climatology progress target=%s samples=%d/%d", target, index, len(files))
    climatology = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0).astype(np.float32)
    output = root / "metadata" / f"monthly_climatology_{target}.npz"
    np.savez_compressed(output, precipitation_mm=climatology, valid_count=counts)
    return {"target": target, "training_files": len(files), "path": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=["chirps", "oya"])
    parser.add_argument("--stage", required=True, choices=["stats", "cache", "climatology", "all"])
    args = parser.parse_args()
    result = None
    if args.stage in {"stats", "all"}:
        result = compute_stats(args.target)
    if args.stage in {"cache", "all"}:
        result = build_fast_cache(args.target)
    if args.stage in {"climatology", "all"}:
        result = build_monthly_climatology(args.target)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
