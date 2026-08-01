"""Fail-fast validation for processed and prepared dataset artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from src.data_preprocessing.quality import DATASET_VERSION, FLOAT_FILL_THRESHOLD, inspect_precipitation
from src.utils.config import Config


PROCESSED_KEYS = {
    "target", "target_valid_mask", "input_valid_mask", "land_mask",
    "preprocessing_version",
}
FAST_KEYS = {
    "inputs_25km", "input_valid_mask_25km", "phys_dem_10km", "real_10km",
    "target_valid_mask_10km", "real_5km", "target_valid_mask_5km",
    "land_mask_5km", "era5_precip_5km_norm", "season",
}


def validate(target: str, stage: str) -> dict:
    root = Path(Config.LOCAL_DATA_DIR) / DATASET_VERSION
    index_path = root / "metadata" / f"dataset_index_{target}.csv"
    if not index_path.exists():
        return {
            "target": target,
            "stage": stage,
            "accepted": False,
            "counts": {"train": 0, "val": 0, "test": 0},
            "errors": [{"path": str(index_path), "error": "missing_dataset_index"}],
        }
    index = pd.read_csv(index_path)
    accepted = index[index["accepted"] == True]
    errors = []
    counts = {"train": 0, "val": 0, "test": 0}

    if stage == "raw":
        source = pd.read_csv(Path(Config.LOCAL_DATA_DIR) / "metadata" / f"dataset_index_{target}.csv")
        source = source[source["valid_flag"] == True]
        for row in source.itertuples(index=False):
            path = Path(row.target_path)
            if target == "oya":
                path = Path(Config.RAW_DATA_DIR) / DATASET_VERSION / "oya" / row.date[:4] / row.date[5:7] / f"oya_{row.date}.tif"
            if not path.exists():
                errors.append({"date": row.date, "path": str(path), "error": "missing_file"})
                continue
            with rasterio.open(path) as raster:
                values = raster.read(1, masked=True)
                qc, _ = inspect_precipitation(values.filled(np.nan), ~np.ma.getmaskarray(values))
                if target == "oya" and raster.count < 2:
                    errors.append({"date": row.date, "path": str(path), "error": "missing_slot_count_band"})
                elif not qc.accepted:
                    errors.append({"date": row.date, "path": str(path), "error": "qc_rejected", "reasons": list(qc.reject_reasons)})
                else:
                    counts[row.split] += 1
        expected_counts = source.groupby("split").size().to_dict()
        for split, expected in expected_counts.items():
            if counts.get(split, 0) != expected:
                errors.append({"split": split, "error": "index_artifact_count_mismatch", "expected": int(expected), "valid_artifacts": counts.get(split, 0)})
        return {"target": target, "stage": stage, "accepted": not errors, "counts": counts, "errors": errors}
    if stage == "processed":
        paths = [(row.split, Path(row.npz_path)) for row in accepted.itertuples(index=False)]
        expected_keys = PROCESSED_KEYS
    else:
        fast_root = Path("data") / DATASET_VERSION / target
        paths = [
            (split, path)
            for split in counts
            for path in sorted((fast_root / split).glob("*.npz"))
        ]
        expected_keys = FAST_KEYS

    for split, path in paths:
        if not path.exists():
            errors.append({"path": str(path), "error": "missing_file"})
            continue
        try:
            with np.load(path) as data:
                missing = expected_keys - set(data.files)
                if missing:
                    errors.append({"path": str(path), "error": "missing_keys", "keys": sorted(missing)})
                    continue
                if stage == "processed" and "feature_source_npz" not in data.files:
                    feature_keys = {"inputs", "upslope", "spectral", "elevation"}
                    if not feature_keys.issubset(data.files):
                        errors.append({"path": str(path), "error": "missing_feature_source"})
                        continue
                if stage == "processed" and "feature_source_npz" in data.files:
                    source_path = Path(str(np.asarray(data["feature_source_npz"]).item()))
                    if not source_path.exists():
                        errors.append({"path": str(path), "error": "missing_feature_source_file", "source": str(source_path)})
                        continue
                if stage == "processed":
                    values = data["target"][..., 0]
                    mask = data["target_valid_mask"][..., 0].astype(bool)
                else:
                    values = data["real_5km"][0]
                    mask = data["target_valid_mask_5km"][0].astype(bool)
                valid_values = values[mask]
                if not valid_values.size:
                    errors.append({"path": str(path), "error": "empty_valid_mask"})
                elif not np.all(np.isfinite(valid_values)):
                    errors.append({"path": str(path), "error": "nonfinite_valid_values"})
                elif np.any(np.abs(valid_values) >= FLOAT_FILL_THRESHOLD):
                    errors.append({"path": str(path), "error": "sentinel_valid_values"})
                else:
                    counts[split] += 1
        except Exception as error:
            errors.append({"path": str(path), "error": "unreadable", "detail": str(error)})

    expected_counts = accepted.groupby("split").size().to_dict()
    for split, expected in expected_counts.items():
        if counts.get(split, 0) != expected:
            errors.append({
                "split": split, "error": "index_artifact_count_mismatch",
                "expected": int(expected), "valid_artifacts": counts.get(split, 0),
            })
    return {
        "target": target,
        "stage": stage,
        "accepted": not errors,
        "counts": counts,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=["chirps", "oya"])
    parser.add_argument("--stage", required=True, choices=["raw", "processed", "fast"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.target, args.stage)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    raise SystemExit(0 if result["accepted"] else 1)


if __name__ == "__main__":
    main()
