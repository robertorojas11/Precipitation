"""Compute training-only atmospheric normalization for every rolling fold."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from src.chirps_v3.config import FOLDS, ROOT, Fold
from src.chirps_v3.dataset import discover_samples


def compute_fold_stats(fold: Fold, output: Path | None = None) -> dict:
    samples = discover_samples(); total = np.zeros(18, np.float64); square = np.zeros(18, np.float64)
    count = np.zeros(18, np.int64); dates = sorted(d for d in samples if fold.contains(int(d[:4]), "train"))
    for date in dates:
        with np.load(samples[date]) as sample:
            values = sample["inputs_25km"].astype(np.float64)
            mask = sample["input_valid_mask_25km"].astype(bool)
        if mask.shape[0] == 1: mask = np.broadcast_to(mask, values.shape)
        elif mask.ndim == 2: mask = np.broadcast_to(mask[None], values.shape)
        for channel in range(18):
            valid = mask[channel] & np.isfinite(values[channel]); selected = values[channel][valid]
            total[channel] += selected.sum(); square[channel] += np.square(selected).sum(); count[channel] += selected.size
    if np.any(count == 0): raise RuntimeError(f"Fold {fold.name} has empty atmospheric channels")
    mean = total / count; std = np.sqrt(np.maximum(square / count - mean ** 2, 1e-8))
    payload = {"fold": fold.name, "training_dates": len(dates), "input_mean_in_v2_space": mean.tolist(),
               "input_std_in_v2_space": std.tolist(), "policy": "statistics use training years only"}
    output = output or ROOT / "fold_stats" / f"{fold.name}.json"; output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp"); temporary.write_text(json.dumps(payload, indent=2) + "\n"); temporary.replace(output)
    return payload


def main():
    p = argparse.ArgumentParser(); p.add_argument("--fold", choices=[f.name for f in FOLDS])
    a = p.parse_args(); folds = [next(f for f in FOLDS if f.name == a.fold)] if a.fold else FOLDS
    print(json.dumps([compute_fold_stats(fold) for fold in folds], indent=2))
if __name__ == "__main__": main()
