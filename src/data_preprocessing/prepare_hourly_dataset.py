r"""Stage 4 -- Prepare: train-only normalization and final training tensors
for the hourly ERA5->IMERG dataset (docs/stages_v2/04-prepare.md has the
full narrative documentation).

Implements RainShift's own preprocessing recipe (matrix
``docs/audit/matriz_trazabilidad_metodologica.md`` M216-M219), reused
because `docs/decisions/0002-implementacion-v2.md` adopts RainShift (P2) as
the experimental skeleton:

* **M216 -- clip ERA5 by IMERG's range.** "Clip ERA5 precipitation using
  min/max of IMERG as thresholds (corrects overestimation of weak rain,
  false positives, unrealistic extremes)." Applied to the ERA5 input
  channels (cp, lsp) only, using the *pooled* IMERG train-split min/max
  across training regions (:func:`compute_imerg_range`) -- one shared
  clip range for the whole framework, not one per region, so a model sees
  the same input contract everywhere it runs (train or zero-shot eval).
* **M217 -- log transform.** "log(x*1000 + 1e-5)", applied to every
  precipitation-like channel (cp, lsp, target) and to orography
  (elevation). Land-sea mask is left untransformed in [0, 1] (M218).
  Elevation is clamped to >= 0 first (M217's formula assumes a
  non-negative quantity; this project's regions span ocean, where NASADEM
  reads small negative bathymetric noise -- clamping before the log
  matches the non-negativity clamp already used for precipitation
  elsewhere in this codebase, and is a documented, deliberate adaptation,
  not a silent one).
* **M219 -- pooled-variance normalization.** Z-score statistics computed
  **per training region**, then combined via RainShift's explicit pooled
  formula (not the naive combined-sample variance):

      mu_pooled = (1/n) * sum_i(mu_i)
      var_pooled = (1/n) * sum_i(sigma_i^2 + mu_i^2) - mu_pooled^2

  with ``n`` = number of training regions (2: `mexico_pacifico`,
  `mexico_atlantico_golfo`). The resulting single, shared set of
  normalization statistics is applied to **every** region, including the
  zero-shot evaluation region (`puerto_rico_caribe`), which never
  contributes to computing them -- exactly RainShift's own geographic-
  generalization setup.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data_preprocessing.build_hourly_dataset import HOURLY_DATASET_VERSION
from src.data_preprocessing.quality import manifest_sha256
from src.utils.config import Config
from src.utils.logging import configure_logging
from src.utils.provenance import write_manifest

logger = configure_logging(name="precipitation.prepare_hourly")

PREPARED_DATASET_VERSION = "v2_generative_hourly_prepared"
TRAINING_REGIONS = ("mexico_pacifico", "mexico_atlantico_golfo")
LOG_SCALE = 1000.0
LOG_EPS = 1e-5


def log_transform(x: np.ndarray) -> np.ndarray:
    """RainShift M217: ``log(max(x, 0) * 1000 + 1e-5)``."""
    return np.log(np.maximum(x, 0.0) * LOG_SCALE + LOG_EPS)


def inverse_log_transform(x_log: np.ndarray) -> np.ndarray:
    return np.maximum((np.exp(x_log) - LOG_EPS) / LOG_SCALE, 0.0)


def _build_root() -> Path:
    return Path(Config.LOCAL_DATA_DIR) / HOURLY_DATASET_VERSION


def _prepared_root() -> Path:
    return Path(Config.LOCAL_DATA_DIR) / PREPARED_DATASET_VERSION


def _train_rows(region: str, *, build_root: Path | None = None) -> list[dict]:
    root = build_root or _build_root()
    index_path = root / "metadata" / f"dataset_index_hourly_{region}.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"No build index for region {region!r}: {index_path}")
    with index_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [r for r in rows if r.get("accepted") in ("True", "true", "1") and r.get("split") == "train"]


def compute_imerg_range(regions: tuple[str, ...] = TRAINING_REGIONS, *, build_root: Path | None = None) -> tuple[float, float]:
    """Pooled IMERG train-split ``(min, max)`` across ``regions`` (M216)."""
    global_min, global_max = np.inf, -np.inf
    for region in regions:
        for row in _train_rows(region, build_root=build_root):
            with np.load(row["npz_path"]) as data:
                target = data["target"][..., 0]
                mask = data["target_valid_mask"].astype(bool)
            values = target[mask]
            if values.size:
                global_min = min(global_min, float(values.min()))
                global_max = max(global_max, float(values.max()))
    if not np.isfinite(global_min) or not np.isfinite(global_max):
        raise RuntimeError("No valid IMERG pixels found across the requested training regions")
    return global_min, global_max


@dataclass(frozen=True)
class ChannelMoments:
    mean: float
    std: float
    count: int


def _region_log_moments(
    region: str, imerg_range: tuple[float, float], *, build_root: Path | None = None,
) -> dict[str, ChannelMoments]:
    """Per-region mean/std of the log-transformed cp, lsp, target, and
    elevation channels, over valid training pixels only."""
    lo, hi = imerg_range
    sums = {"cp": 0.0, "lsp": 0.0, "target": 0.0, "elevation": 0.0}
    squares = dict(sums)
    counts = {k: 0 for k in sums}

    rows = _train_rows(region, build_root=build_root)
    logger.info("Computing log-moments region=%s train_days=%d", region, len(rows))
    elevation_done = False
    for index, row in enumerate(rows, start=1):
        with np.load(row["npz_path"]) as data:
            inputs = data["inputs"]  # (24, H, W, 2) = [cp, lsp]
            target = data["target"][..., 0]  # (24, H, W)
            input_mask = data["input_valid_mask"].astype(bool)  # (24, H, W)
            target_mask = data["target_valid_mask"].astype(bool)  # (24, H, W)
            elevation = data["elevation"]  # (H, W), static per region
            basin_mask = data["basin_mask"].astype(bool)  # (H, W), static per region

        cp = log_transform(np.clip(inputs[..., 0], lo, hi))[input_mask]
        lsp = log_transform(np.clip(inputs[..., 1], lo, hi))[input_mask]
        tgt = log_transform(target)[target_mask]
        for name, values in (("cp", cp), ("lsp", lsp), ("target", tgt)):
            if values.size:
                sums[name] += float(values.sum(dtype=np.float64))
                squares[name] += float(np.square(values, dtype=np.float64).sum())
                counts[name] += int(values.size)

        if not elevation_done:  # elevation is static per region: fold in once
            elev = log_transform(elevation)[basin_mask]
            if elev.size:
                sums["elevation"] = float(elev.sum(dtype=np.float64))
                squares["elevation"] = float(np.square(elev, dtype=np.float64).sum())
                counts["elevation"] = int(elev.size)
            elevation_done = True

        if index % 200 == 0 or index == len(rows):
            logger.info("log-moments progress region=%s days=%d/%d", region, index, len(rows))

    if any(counts[k] == 0 for k in sums):
        raise RuntimeError(f"Region {region!r} has a channel with zero valid training pixels: {counts}")
    moments = {}
    for name in sums:
        mean = sums[name] / counts[name]
        variance = max(squares[name] / counts[name] - mean ** 2, 0.0)
        moments[name] = ChannelMoments(mean=mean, std=max(variance ** 0.5, 1e-8), count=counts[name])
    return moments


def pooled_stats(per_region_moments: dict[str, dict[str, ChannelMoments]]) -> dict[str, ChannelMoments]:
    """RainShift M219 pooled-variance combination across regions (equal
    per-region weight -- ``n`` = number of *regions*, not of pixels, so a
    region with more accepted days does not dominate the statistics)."""
    n = len(per_region_moments)
    if n == 0:
        raise ValueError("pooled_stats requires at least one region's moments")
    channels = next(iter(per_region_moments.values())).keys()
    pooled: dict[str, ChannelMoments] = {}
    for channel in channels:
        means = [m[channel].mean for m in per_region_moments.values()]
        stds = [m[channel].std for m in per_region_moments.values()]
        total_count = sum(m[channel].count for m in per_region_moments.values())
        mu_pooled = sum(means) / n
        var_pooled = sum(s ** 2 + mu ** 2 for s, mu in zip(stds, means)) / n - mu_pooled ** 2
        pooled[channel] = ChannelMoments(mean=mu_pooled, std=max(var_pooled, 0.0) ** 0.5 or 1e-8, count=total_count)
    return pooled


def compute_stats(training_regions: tuple[str, ...] = TRAINING_REGIONS, *, build_root: Path | None = None) -> dict:
    """Full stats pipeline: IMERG clip range, per-region log moments, pooled
    combination. Writes ``norm_stats.json`` and returns it."""
    imerg_lo, imerg_hi = compute_imerg_range(training_regions, build_root=build_root)
    per_region = {
        region: _region_log_moments(region, (imerg_lo, imerg_hi), build_root=build_root)
        for region in training_regions
    }
    pooled = pooled_stats(per_region)

    stats = {
        "preprocessing_version": PREPARED_DATASET_VERSION,
        "training_regions": list(training_regions),
        "log_transform": {"scale": LOG_SCALE, "eps": LOG_EPS},
        "era5_clip_range_imerg_mm_per_hr": [imerg_lo, imerg_hi],
        "pooled": {name: {"mean": m.mean, "std": m.std, "count": m.count} for name, m in pooled.items()},
        "per_region": {
            region: {name: {"mean": m.mean, "std": m.std, "count": m.count} for name, m in moments.items()}
            for region, moments in per_region.items()
        },
    }
    root = _prepared_root()
    metadata_dir = root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    import json

    (metadata_dir / "norm_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def _normalize(values: np.ndarray, moments: ChannelMoments) -> np.ndarray:
    return (values - moments.mean) / moments.std


def prepare_region(region: str, stats: dict, *, build_root: Path | None = None, output_root: Path | None = None) -> dict:
    """Apply the (already-computed, shared) normalization stats to every
    accepted day of ``region`` -- train, val, test, or a never-trained-on
    zero-shot eval region alike -- and write one prepared NPZ per day."""
    build_root = build_root or _build_root()
    output_root = output_root or _prepared_root()
    lo, hi = stats["era5_clip_range_imerg_mm_per_hr"]
    pooled = {name: ChannelMoments(**m) for name, m in stats["pooled"].items()}

    index_path = build_root / "metadata" / f"dataset_index_hourly_{region}.csv"
    with index_path.open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r.get("accepted") in ("True", "true", "1")]

    records = []
    for index, row in enumerate(rows, start=1):
        with np.load(row["npz_path"]) as data:
            inputs = data["inputs"]
            target = data["target"][..., 0]
            input_mask = data["input_valid_mask"]
            target_mask = data["target_valid_mask"]
            land_mask = data["land_mask"]
            elevation = data["elevation"]
            basin_mask = data["basin_mask"]

        cp_norm = _normalize(log_transform(np.clip(inputs[..., 0], lo, hi)), pooled["cp"])
        lsp_norm = _normalize(log_transform(np.clip(inputs[..., 1], lo, hi)), pooled["lsp"])
        target_norm = _normalize(log_transform(target), pooled["target"])
        elevation_norm = _normalize(log_transform(elevation), pooled["elevation"])

        destination = output_root / region / row["split"] / f"{row['date']}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            inputs_norm=np.stack([cp_norm, lsp_norm], axis=-1).astype(np.float32),
            target_norm=target_norm.astype(np.float32)[..., None],
            invariants_norm=np.stack([land_mask.astype(np.float32), elevation_norm.astype(np.float32)], axis=-1),
            input_valid_mask=input_mask,
            target_valid_mask=target_mask,
            basin_mask=basin_mask,
            date=row["date"],
            region=region,
            split=row["split"],
            preprocessing_version=PREPARED_DATASET_VERSION,
        )
        records.append({"date": row["date"], "region": region, "split": row["split"], "path": str(destination)})
        if index % 200 == 0 or index == len(rows):
            logger.info("prepare progress region=%s days=%d/%d", region, index, len(rows))

    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_manifest(
        metadata_dir / f"manifest_prepared_{region}.json",
        kind="prepare_hourly",
        records=records,
        extra={"region": region, "preprocessing_version": PREPARED_DATASET_VERSION},
    )
    return manifest


def main() -> None:
    import argparse
    import json

    from src.data_preprocessing import regions as region_defs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["stats", "prepare"])
    parser.add_argument("--region", choices=region_defs.list_regions())
    args = parser.parse_args()

    if args.stage == "stats":
        result = compute_stats()
    else:
        if not args.region:
            raise SystemExit("--region is required for --stage prepare")
        metadata_dir = _prepared_root() / "metadata"
        stats = json.loads((metadata_dir / "norm_stats.json").read_text())
        result = prepare_region(args.region, stats)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
