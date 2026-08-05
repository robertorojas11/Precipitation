"""Rolling-fold dataset backed by the immutable v2 prepared archive."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import json

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from src.chirps_v3.config import PREPARED_ROOT, ROOT, Fold, get_fold
from src.utils.config import Config


def source_stats() -> dict:
    path = Path(Config.LOCAL_DATA_DIR) / "v2_clean" / "metadata" / "norm_stats_chirps.json"
    if not path.exists():
        raise FileNotFoundError(f"Required v2 normalization statistics not found: {path}")
    return json.loads(path.read_text())


def discover_samples(root: Path = PREPARED_ROOT) -> dict[str, Path]:
    samples: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        for path in (root / split).glob("*.npz"):
            samples[path.stem] = path
    return samples


def context_offsets(days: int) -> range:
    if days not in {1, 3, 5, 7}:
        raise ValueError("context_days must be one of 1, 3, 5, 7")
    radius = days // 2
    return range(-radius, radius + 1)


def _terrain_features(physical_10km: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    elevation = physical_10km[2:3].unsqueeze(0)
    elevation = F.interpolate(elevation, size=output_size, mode="bilinear", align_corners=False)[0]
    padded = F.pad(elevation.unsqueeze(0), (1, 1, 1, 1), mode="replicate")[0]
    dx = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) * 0.5
    dy = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) * 0.5
    slope = torch.sqrt(dx.square() + dy.square() + 1e-8)
    aspect_sin = dy / slope.clamp_min(1e-4)
    aspect_cos = dx / slope.clamp_min(1e-4)
    curvature = (padded[:, 1:-1, 2:] + padded[:, 1:-1, :-2] +
                 padded[:, 2:, 1:-1] + padded[:, :-2, 1:-1] - 4 * elevation)
    relief_small = elevation - F.avg_pool2d(elevation, 5, stride=1, padding=2)
    relief_large = elevation - F.avg_pool2d(elevation, 21, stride=1, padding=10)
    return torch.cat((elevation, slope, aspect_sin, aspect_cos, curvature,
                      relief_small, relief_large), dim=0)


class RollingChirpsDataset(Dataset):
    """Samples a centered context without crossing a fold's split boundary."""

    def __init__(self, fold: str | Fold, split: str, context_days: int = 5,
                 root: Path = PREPARED_ROOT, include_static: bool = True) -> None:
        self.fold = get_fold(fold) if isinstance(fold, str) else fold
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        self.split, self.offsets = split, context_offsets(context_days)
        self.context_days, self.include_static = context_days, include_static
        self.available = discover_samples(root)
        candidates = [date for date in self.available
                      if self.fold.contains(int(date[:4]), split)]
        self.dates = sorted(date for date in candidates if self._context_is_valid(date))
        if not self.dates:
            raise RuntimeError(f"No usable {split} samples for {self.fold.name}")
        self.stats = source_stats()
        fold_stats_path = ROOT / "fold_stats" / f"{self.fold.name}.json"
        if not fold_stats_path.exists():
            raise FileNotFoundError(
                f"Training-only fold statistics not found: {fold_stats_path}. "
                "Run python -m src.chirps_v3.prepare first."
            )
        fold_stats = json.loads(fold_stats_path.read_text())
        self.input_mean = torch.tensor(fold_stats["input_mean_in_v2_space"]).view(18, 1, 1)
        self.input_std = torch.tensor(fold_stats["input_std_in_v2_space"]).view(18, 1, 1)

    def _context_is_valid(self, date: str) -> bool:
        center = datetime.strptime(date, "%Y-%m-%d")
        for offset in self.offsets:
            neighbor = (center + timedelta(days=offset)).strftime("%Y-%m-%d")
            if neighbor not in self.available or not self.fold.contains(int(neighbor[:4]), self.split):
                return False
        return True

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        date = self.dates[index]
        center = datetime.strptime(date, "%Y-%m-%d")
        context, context_masks = [], []
        for offset in self.offsets:
            key = (center + timedelta(days=offset)).strftime("%Y-%m-%d")
            with np.load(self.available[key]) as sample:
                atmospheric = torch.from_numpy(sample["inputs_25km"]).float()
                context.append((atmospheric - self.input_mean) / self.input_std)
                context_masks.append(torch.from_numpy(sample["input_valid_mask_25km"]).bool())
        with np.load(self.available[date]) as sample:
            target_norm = torch.from_numpy(sample["real_5km"]).float()
            target_mm = torch.expm1(target_norm * self.stats["target_std"] + self.stats["target_mean"]).clamp_min(0)
            physics = torch.from_numpy(sample["phys_dem_10km"]).float()
            baseline_norm = torch.from_numpy(sample["era5_precip_5km_norm"]).float()
            baseline_mm = torch.expm1(baseline_norm * self.stats["target_std"] + self.stats["target_mean"]).clamp_min(0)
            valid = torch.from_numpy(sample["target_valid_mask_5km"]).bool()
            land = torch.from_numpy(sample["land_mask_5km"]).bool()
            season = torch.from_numpy(sample["season"]).float()
        result = {
            "inputs_25km": torch.stack(context),
            "input_valid_mask_25km": torch.stack(context_masks),
            "physics_10km": physics,
            "target_mm": target_mm,
            "target_valid_mask": valid & land,
            "baseline_mm": baseline_mm,
            "season": season,
            "date": date,
        }
        if self.include_static:
            result["terrain_5km"] = _terrain_features(physics, target_mm.shape[-2:])
        return result
