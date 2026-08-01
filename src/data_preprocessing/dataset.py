"""PyTorch dataset for prepared precipitation downscaling tensors."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data_preprocessing.quality import DATASET_VERSION


class PrecipitationDataset(Dataset):
    """Load prepared tensors and explicit validity masks for one data split."""

    def __init__(
        self,
        target: str,
        split: str,
        context_days: int = 1,
        data_directory: Path | str = Path("data"),
    ) -> None:
        if target not in {"chirps", "oya"}:
            raise ValueError("target must be 'chirps' or 'oya'")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")
        if context_days not in {1, 3}:
            raise ValueError("context_days must be 1 or 3")

        self.target = target
        self.split = split
        self.context_days = context_days
        self.data_directory = Path(data_directory) / DATASET_VERSION / target / split
        if not self.data_directory.is_dir():
            raise FileNotFoundError(f"Prepared dataset not found: {self.data_directory}")

        paths = sorted(self.data_directory.glob("*.npz"))
        self.available = {path.stem: path for path in paths}
        self.files = paths if context_days == 1 else self._files_with_context(paths)
        if not self.files:
            raise RuntimeError(f"No usable samples found in {self.data_directory}")

    def _files_with_context(self, paths: list[Path]) -> list[Path]:
        usable = []
        for path in paths:
            date = datetime.strptime(path.stem, "%Y-%m-%d")
            neighbors = [
                (date + timedelta(days=offset)).strftime("%Y-%m-%d")
                for offset in (-1, 0, 1)
            ]
            if all(neighbor in self.available for neighbor in neighbors):
                usable.append(path)
        return usable

    def __len__(self) -> int:
        return len(self.files)

    def _atmospheric_inputs(self, center_path: Path) -> torch.Tensor:
        if self.context_days == 1:
            with np.load(center_path) as sample:
                return torch.from_numpy(sample["inputs_25km"]).float()

        date = datetime.strptime(center_path.stem, "%Y-%m-%d")
        context = []
        for offset in (-1, 0, 1):
            neighbor = (date + timedelta(days=offset)).strftime("%Y-%m-%d")
            with np.load(self.available[neighbor]) as sample:
                context.append(torch.from_numpy(sample["inputs_25km"]).float())
        return torch.cat(context, dim=0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.files[index]
        atmospheric_inputs = self._atmospheric_inputs(path)
        with np.load(path) as sample:
            return {
                "inputs_25km": atmospheric_inputs,
                "input_valid_mask_25km": torch.from_numpy(
                    sample["input_valid_mask_25km"]
                ).bool(),
                "phys_dem_10km": torch.from_numpy(sample["phys_dem_10km"]).float(),
                "target_10km": torch.from_numpy(sample["real_10km"]).float(),
                "target_valid_mask_10km": torch.from_numpy(
                    sample["target_valid_mask_10km"]
                ).bool(),
                "target_5km": torch.from_numpy(sample["real_5km"]).float(),
                "target_valid_mask_5km": torch.from_numpy(
                    sample["target_valid_mask_5km"]
                ).bool(),
                "land_mask_5km": torch.from_numpy(sample["land_mask_5km"]).bool(),
                "era5_precip_5km_norm": torch.from_numpy(
                    sample["era5_precip_5km_norm"]
                ).float(),
                "season": torch.from_numpy(sample["season"]).float(),
                "date": path.stem,
            }
