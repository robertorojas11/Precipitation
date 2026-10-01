"""PyTorch ``Dataset`` over stage 4/5's prepared, normalized hourly tensors
(stage 6 -- training, PASO 5 of docs/decisions/0002-implementacion-v2.md;
full narrative in docs/stages_v2/06-train.md).

One sample is one ``(day, hour)`` pair from a **single region's** prepared
split -- this project's training decision is "single-region start"
(docs/decisions/0002-implementacion-v2.md §1.1, reaffirming decision 0001;
``docs/architecture_v3.md`` B.4 marks the hierarchical region-union
mechanism DEFERRED). Each region has its own canonical grid shape
(``build_hourly_dataset.region_grid``), so pooling multiple regions into one
batch would require padding or a region-homogeneous batch sampler neither
of which is built yet -- training one region at a time sidesteps the issue
entirely and matches the deferred-union decision already on record.

Every tensor returned here matches the channel layout the three candidate
models (``src/models/{generator,discriminator,regression_stage,
deterministic_resnet}.py``) expect directly, with no further reshaping
needed by a training loop: ``inputs`` (lr_channels, H, W), ``invariants``
(2, H, W), ``target`` (1, H, W).

**NaN sanitization is a forward-pass concern only, never a loss/metric
concern.** Stage 4 deliberately leaves invalid pixels as ``NaN`` after the
log transform + normalization (docs/stages_v2/README.md's "Shared
experimental contract": "never silently converted to zero rain"). Feeding
a NaN into a convolution would poison every pixel in its receptive field,
not just the invalid one, so ``__getitem__`` replaces NaN with ``0.0``
(the distribution's own mean in normalized z-score space) **only** in the
tensors handed to the network; the original validity masks are returned
unchanged and must still be used to exclude those exact pixels from every
loss and metric -- the substitution never reaches a loss function.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data_preprocessing.prepare_hourly_dataset import _prepared_root
from src.utils.provenance import load_manifest

HOURS_PER_DAY = 24


def _nan_to_zero(array: np.ndarray) -> np.ndarray:
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


class PreparedHourlyDataset(Dataset):
    """``len()`` is ``accepted_days * 24``; ``__getitem__`` reads one day's
    NPZ per call and slices out the requested hour (no cross-hour caching --
    a known IO-performance limitation at full historical scale, see
    docs/stages_v2/06-train.md's "Known limitations").
    """

    def __init__(self, region: str, split: str, *, prepared_root: Path | None = None):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")
        self.region = region
        self.split = split
        self.prepared_root = Path(prepared_root) if prepared_root is not None else _prepared_root()
        manifest_path = self.prepared_root / "metadata" / f"manifest_prepared_{region}.json"
        manifest = load_manifest(manifest_path)
        if manifest is None:
            raise FileNotFoundError(f"No prepared manifest at {manifest_path}")
        self.records = [r for r in manifest.get("records", []) if r["split"] == split]
        if not self.records:
            raise ValueError(f"No prepared records for region={region!r} split={split!r} in {manifest_path}")

    def __len__(self) -> int:
        return len(self.records) * HOURS_PER_DAY

    def __getitem__(self, index: int) -> dict:
        day_index, hour = divmod(index, HOURS_PER_DAY)
        record = self.records[day_index]
        with np.load(record["path"]) as data:
            inputs = np.ascontiguousarray(data["inputs_norm"][hour])          # (H, W, 2)
            target = np.ascontiguousarray(data["target_norm"][hour, ..., 0])  # (H, W)
            invariants = np.ascontiguousarray(data["invariants_norm"])        # (H, W, 2)
            input_valid = np.ascontiguousarray(data["input_valid_mask"][hour])
            target_valid = np.ascontiguousarray(data["target_valid_mask"][hour])
            basin_mask = np.ascontiguousarray(data["basin_mask"])

        return {
            "inputs": torch.from_numpy(_nan_to_zero(inputs)).permute(2, 0, 1).float(),
            "invariants": torch.from_numpy(_nan_to_zero(invariants)).permute(2, 0, 1).float(),
            "target": torch.from_numpy(_nan_to_zero(target)).unsqueeze(0).float(),
            "input_valid_mask": torch.from_numpy(input_valid).bool(),
            "target_valid_mask": torch.from_numpy(target_valid).bool(),
            "basin_mask": torch.from_numpy(basin_mask).bool(),
            "region": self.region,
            "date": record["date"],
        }
