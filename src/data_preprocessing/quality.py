"""Shared quality-control primitives for versioned precipitation datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


DATASET_VERSION = "v2_clean"
FLOAT_FILL_THRESHOLD = 1.0e10
DEFAULT_MAX_DAILY_MM = 1000.0
MIN_VALID_FRACTION = 0.05


@dataclass(frozen=True)
class ArrayQC:
    accepted: bool
    reject_reasons: tuple[str, ...]
    valid_pixels: int
    total_pixels: int
    valid_fraction: float
    nonfinite_count: int
    sentinel_count: int
    negative_count: int
    implausible_count: int
    minimum: float | None
    maximum: float | None
    p50: float | None
    p95: float | None
    p99: float | None
    p999: float | None

    def to_dict(self) -> dict:
        result = asdict(self)
        result["reject_reasons"] = list(self.reject_reasons)
        return result


def valid_precipitation_mask(
    values: np.ndarray,
    source_valid_mask: np.ndarray | None = None,
    *,
    max_daily_mm: float = DEFAULT_MAX_DAILY_MM,
) -> np.ndarray:
    """Return pixels that are finite, non-sentinel and physically plausible."""
    array = np.asarray(values)
    valid = np.isfinite(array)
    valid &= np.abs(array) < FLOAT_FILL_THRESHOLD
    valid &= array >= 0.0
    valid &= array <= max_daily_mm
    if source_valid_mask is not None:
        valid &= np.asarray(source_valid_mask, dtype=bool)
    return valid


def inspect_precipitation(
    values: np.ndarray,
    source_valid_mask: np.ndarray | None = None,
    *,
    max_daily_mm: float = DEFAULT_MAX_DAILY_MM,
    min_valid_fraction: float = MIN_VALID_FRACTION,
) -> tuple[ArrayQC, np.ndarray]:
    """Inspect an array without clipping or converting invalid pixels to rain."""
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    sentinel = finite & (np.abs(array) >= FLOAT_FILL_THRESHOLD)
    negative = finite & (array < 0.0) & ~sentinel
    implausible = finite & (array > max_daily_mm) & ~sentinel
    valid = valid_precipitation_mask(
        array, source_valid_mask, max_daily_mm=max_daily_mm
    )

    valid_values = array[valid]
    reasons: list[str] = []
    valid_fraction = float(valid.mean()) if valid.size else 0.0
    if sentinel.any():
        reasons.append("sentinel_values")
    if implausible.any():
        reasons.append("implausible_precipitation")
    if valid_fraction < min_valid_fraction:
        reasons.append("insufficient_valid_coverage")
    if valid_values.size == 0:
        reasons.append("no_valid_precipitation")

    quantiles: list[float | None]
    if valid_values.size:
        quantiles = [
            float(value)
            for value in np.quantile(valid_values, [0.0, 0.5, 0.95, 0.99, 0.999, 1.0])
        ]
    else:
        quantiles = [None] * 6

    qc = ArrayQC(
        accepted=not reasons,
        reject_reasons=tuple(dict.fromkeys(reasons)),
        valid_pixels=int(valid.sum()),
        total_pixels=int(valid.size),
        valid_fraction=valid_fraction,
        nonfinite_count=int((~finite).sum()),
        sentinel_count=int(sentinel.sum()),
        negative_count=int(negative.sum()),
        implausible_count=int(implausible.sum()),
        minimum=quantiles[0],
        p50=quantiles[1],
        p95=quantiles[2],
        p99=quantiles[3],
        p999=quantiles[4],
        maximum=quantiles[5],
    )
    return qc, valid


def aggregate_oya_slots_numpy(
    slots_mm_per_hour: np.ndarray,
    *,
    min_valid_slots: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate Oya rate slots to daily mm using valid-slot mean times 24.

    The first axis is time. Invalid and sentinel values do not contribute.
    Pixels below ``min_valid_slots`` remain NaN and false in the returned mask.
    """
    slots = np.asarray(slots_mm_per_hour, dtype=np.float64)
    if slots.ndim < 2:
        raise ValueError("Oya slots must have shape (time, ...spatial dimensions)")
    valid = np.isfinite(slots) & (np.abs(slots) < FLOAT_FILL_THRESHOLD) & (slots >= 0)
    slot_count = valid.sum(axis=0).astype(np.uint8)
    slot_sum = np.where(valid, slots, 0.0).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        daily = slot_sum / slot_count * 24.0
    coverage = slot_count >= min_valid_slots
    daily = np.where(coverage, daily, np.nan).astype(np.float32)
    return daily, slot_count, coverage


def masked_r2(prediction: np.ndarray, observation: np.ndarray, mask: np.ndarray) -> float:
    """Compute standard pooled R² over exactly the supplied valid pixels."""
    pred = np.asarray(prediction, dtype=np.float64)
    obs = np.asarray(observation, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(pred) & np.isfinite(obs)
    if valid.sum() < 2:
        raise ValueError("R² requires at least two valid pixels")
    pred_valid = pred[valid]
    obs_valid = obs[valid]
    denominator = np.square(obs_valid - obs_valid.mean()).sum()
    if denominator <= 0:
        raise ValueError("R² is undefined for constant observations")
    return float(1.0 - np.square(obs_valid - pred_valid).sum() / denominator)


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha256(records: Iterable[dict]) -> str:
    payload = json.dumps(list(records), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def ensure_finite_features(features: np.ndarray) -> np.ndarray:
    """Return a per-pixel mask requiring every feature channel to be finite."""
    values = np.asarray(features)
    if values.ndim < 3:
        raise ValueError("Features must end with a channel dimension")
    return np.all(np.isfinite(values) & (np.abs(values) < FLOAT_FILL_THRESHOLD), axis=-1)
