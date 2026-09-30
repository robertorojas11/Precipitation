"""Fail-fast validation for the v2-generative hourly dataset (stage 3 --
docs/stages_v2/03-validate.md), mirroring the legacy pipeline's
``validate_dataset.py``: two gates, ``raw`` (before build) and
``processed`` (after build), each returning a structured
``{region, stage, accepted, counts, errors}`` report and exiting non-zero
on any error. Neither gate mutates data; both are safe to re-run at any
time as an integrity audit.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.data_preprocessing.build_hourly_dataset import HOURS_PER_DAY, HOURLY_DATASET_VERSION
from src.data_preprocessing.quality import FLOAT_FILL_THRESHOLD, file_sha256, manifest_sha256
from src.utils.config import Config
from src.utils.provenance import load_manifest

PROCESSED_KEYS = {
    "inputs", "target", "input_valid_mask", "target_valid_mask",
    "land_mask", "elevation", "basin_mask", "preprocessing_version",
}


def _v2_generative_root() -> Path:
    return Path(Config.LOCAL_DATA_DIR) / HOURLY_DATASET_VERSION


def validate_raw(region: str, *, v2_raw_root: Path | None = None, full_hash_check: bool = False) -> dict:
    """Cross-check the acquisition manifests (``era5_ee``, ``imerg``)
    against what is actually on disk for one region, and confirm both
    datasets cover the same set of days -- a day missing from only one of
    the two cannot be built (``build_day``'s ``missing_source_file``), so
    catching that gap here, before spending time on a build run, is the
    point of this gate. ``full_hash_check`` recomputes every file's SHA-256
    against its manifest record (slow, off by default -- existence-only is
    the fast path; a manifest hash mismatch is a tamper/corruption signal,
    not something that happens in normal operation).
    """
    errors: list[dict] = []
    counts = {"era5_ee": 0, "imerg": 0}
    root = Path(v2_raw_root or Config.V2_RAW_DATA_DIR)
    days_by_kind: dict[str, set[str]] = {}

    for kind in ("era5_ee", "imerg"):
        manifest_path = root / kind / region / f"manifest_{kind}_{region}.json"
        manifest = load_manifest(manifest_path)
        if manifest is None:
            errors.append({"kind": kind, "error": "missing_manifest", "path": str(manifest_path)})
            days_by_kind[kind] = set()
            continue
        days: set[str] = set()
        for record in manifest.get("records", []):
            file_path = root / record["path"]
            if not file_path.exists():
                errors.append({"kind": kind, "path": str(file_path), "error": "missing_file"})
                continue
            if full_hash_check:
                actual = file_sha256(file_path)
                if actual != record.get("sha256"):
                    errors.append({
                        "kind": kind, "path": str(file_path), "error": "hash_mismatch",
                        "expected": record.get("sha256"), "actual": actual,
                    })
                    continue
            days.add(record["day"])
            counts[kind] += 1
        days_by_kind[kind] = days

    era5_only = sorted(days_by_kind.get("era5_ee", set()) - days_by_kind.get("imerg", set()))
    imerg_only = sorted(days_by_kind.get("imerg", set()) - days_by_kind.get("era5_ee", set()))
    if era5_only:
        errors.append({"error": "era5_without_matching_imerg", "count": len(era5_only), "sample": era5_only[:5]})
    if imerg_only:
        errors.append({"error": "imerg_without_matching_era5", "count": len(imerg_only), "sample": imerg_only[:5]})

    return {"region": region, "stage": "raw", "accepted": not errors, "counts": counts, "errors": errors}


def _read_index(root: Path, region: str) -> list[dict]:
    index_path = root / "metadata" / f"dataset_index_hourly_{region}.csv"
    if not index_path.exists():
        return []
    with index_path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def validate_processed(region: str, *, root: Path | None = None) -> dict:
    """Open every NPZ the build stage marked accepted, verify its keys,
    shapes, and that no sentinel/non-finite value hides inside a pixel its
    own mask calls valid -- the manifest recording success is not proof by
    itself (same principle as the legacy pipeline's
    ``docs/stages/03-build.md``: "a manifest alone cannot authorize
    preparation").
    """
    root = Path(root or _v2_generative_root())
    errors: list[dict] = []
    counts = {"train": 0, "val": 0, "test": 0}
    index = _read_index(root, region)
    if not index:
        return {
            "region": region, "stage": "processed", "accepted": False, "counts": counts,
            "errors": [{"error": "missing_or_empty_dataset_index"}],
        }

    accepted_rows = [row for row in index if row.get("accepted") in ("True", "true", "1")]
    for row in accepted_rows:
        path = Path(row["npz_path"])
        split = row["split"]
        if not path.exists():
            errors.append({"path": str(path), "error": "missing_file"})
            continue
        try:
            with np.load(path) as data:
                missing_keys = PROCESSED_KEYS - set(data.files)
                if missing_keys:
                    errors.append({"path": str(path), "error": "missing_keys", "keys": sorted(missing_keys)})
                    continue
                inputs, target = data["inputs"], data["target"][..., 0]
                input_mask = data["input_valid_mask"].astype(bool)
                target_mask = data["target_valid_mask"].astype(bool)
                if inputs.shape[0] != HOURS_PER_DAY or target.shape[0] != HOURS_PER_DAY:
                    errors.append({"path": str(path), "error": "unexpected_hour_dimension", "inputs_shape": list(inputs.shape)})
                    continue

                def _bad(values: np.ndarray, mask: np.ndarray) -> str | None:
                    valid_values = values[mask]
                    if not valid_values.size:
                        return "empty_valid_mask"
                    if not np.all(np.isfinite(valid_values)):
                        return "nonfinite_valid_values"
                    if np.any(np.abs(valid_values) >= FLOAT_FILL_THRESHOLD):
                        return "sentinel_valid_values"
                    return None

                target_problem = _bad(target, target_mask)
                input_problem = _bad(inputs, np.broadcast_to(input_mask[..., None], inputs.shape))
                if target_problem:
                    errors.append({"path": str(path), "error": target_problem, "field": "target"})
                elif input_problem:
                    errors.append({"path": str(path), "error": input_problem, "field": "inputs"})
                else:
                    counts[split] = counts.get(split, 0) + 1
        except Exception as error:  # corrupted/unreadable NPZ
            errors.append({"path": str(path), "error": "unreadable", "detail": str(error)})

    expected_counts: dict[str, int] = {}
    for row in accepted_rows:
        expected_counts[row["split"]] = expected_counts.get(row["split"], 0) + 1
    for split, expected in expected_counts.items():
        if counts.get(split, 0) != expected:
            errors.append({
                "split": split, "error": "index_artifact_count_mismatch",
                "expected": expected, "valid_artifacts": counts.get(split, 0),
            })

    manifest_path = root / "metadata" / f"manifest_hourly_{region}.json"
    manifest = load_manifest(manifest_path)
    if manifest is None:
        errors.append({"error": "missing_manifest", "path": str(manifest_path)})
    else:
        # Recomputed from the manifest's *own* embedded records, not the
        # freshly re-read CSV: csv.DictReader stringifies every field (a
        # bool becomes the string "True"), so hashing those against a hash
        # computed over the original Python-typed records at build time
        # would mismatch unconditionally, even with zero tampering. This
        # instead checks that manifest.json is internally self-consistent
        # (records vs. its own manifest_sha256) -- exactly what would break
        # if only one of the two fields were hand-edited.
        recomputed = manifest_sha256(manifest.get("records", []))
        if recomputed != manifest.get("manifest_sha256"):
            errors.append({
                "error": "manifest_hash_mismatch",
                "expected": manifest.get("manifest_sha256"), "actual": recomputed,
            })

    return {"region": region, "stage": "processed", "accepted": not errors, "counts": counts, "errors": errors}


def validate(
    region: str, stage: str, *,
    v2_raw_root: Path | None = None, root: Path | None = None, full_hash_check: bool = False,
) -> dict:
    if stage == "raw":
        return validate_raw(region, v2_raw_root=v2_raw_root, full_hash_check=full_hash_check)
    if stage == "processed":
        return validate_processed(region, root=root)
    raise ValueError(f"Unknown stage {stage!r}; choose 'raw' or 'processed'")


def main() -> None:
    from src.data_preprocessing import regions as region_defs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, choices=region_defs.list_regions())
    parser.add_argument("--stage", required=True, choices=["raw", "processed"])
    parser.add_argument("--full-hash-check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.region, args.stage, full_hash_check=args.full_hash_check)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    raise SystemExit(0 if result["accepted"] else 1)


if __name__ == "__main__":
    main()
