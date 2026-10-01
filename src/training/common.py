"""Shared training-loop utilities (stage 6) reused by all three candidate
training scripts (``train_baseline.py``, ``train_gan.py``,
``train_diffusion.py``) so the data-loading, physical-unit conversion,
masking, checkpointing, and run-provenance conventions are identical across
candidates -- the same "compare like with like" requirement that motivates
RainShift's own multi-model harness (docs/decisions/0002-implementacion-v2.md
§1.1). Full narrative: docs/stages_v2/06-train.md.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from src.data_preprocessing.prepare_hourly_dataset import (
    LOG_EPS,
    LOG_SCALE,
    ChannelMoments,
    _prepared_root,
)
from src.utils.provenance import source_revision


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_norm_stats(prepared_root: Path | None = None) -> dict:
    root = Path(prepared_root) if prepared_root is not None else _prepared_root()
    path = root / "metadata" / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No norm_stats.json at {path} -- run stage 4's compute_stats() first "
            "(docs/stages_v2/04-prepare.md)."
        )
    return json.loads(path.read_text())


def target_moments(stats: dict) -> ChannelMoments:
    return ChannelMoments(**stats["pooled"]["target"])


def denormalize_to_mm(target_norm: torch.Tensor, moments: ChannelMoments) -> torch.Tensor:
    """Inverse of stage 4's M217+M219: undo the z-score, then the log
    transform, returning physical precipitation in mm/hr, non-negative.
    Torch equivalent of ``prepare_hourly_dataset.inverse_log_transform``
    composed with de-standardization, kept in lockstep with that module so
    the two can never silently diverge (it imports ``LOG_SCALE``/``LOG_EPS``
    from it rather than redefining them).
    """
    log_value = target_norm * moments.std + moments.mean
    return ((torch.exp(log_value) - LOG_EPS) / LOG_SCALE).clamp_min(0.0)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def loss_mask(batch: dict) -> torch.Tensor:
    """The pixel set every candidate's loss/metric is restricted to: a
    target observation must be valid, every input channel at that pixel
    must be valid, and the pixel must be inside the basin polygon (mirrors
    the legacy pipeline's ``target_valid_mask & land_mask``-style
    combination, substituting ``basin_mask`` -- the v2-generative land/sea
    proxy, docs/stages_v2/02-build.md -- and the input validity mask, since
    stage 4 never required `input_valid == target_valid`).
    """
    return batch["target_valid_mask"] & batch["input_valid_mask"] & batch["basin_mask"]


class PooledMetrics:
    """Streaming R2/RMSE/MAE over every valid pixel seen across an entire
    validation pass (pooled, not averaged-per-batch, so small last batches
    don't get over-weighted) -- same construction as
    ``legacy/deterministic_unet/train.py``'s ``PooledMetrics``.
    """

    def __init__(self) -> None:
        self.count = 0
        self.obs_sum = 0.0
        self.obs_sq_sum = 0.0
        self.res_sq_sum = 0.0
        self.abs_sum = 0.0

    def update(self, prediction: torch.Tensor, observation: torch.Tensor, mask: torch.Tensor) -> None:
        if mask.dim() == prediction.dim() - 1:
            mask = mask.unsqueeze(1)
        mask = mask.expand_as(prediction)
        pred = prediction[mask].double()
        obs = observation[mask].double()
        self.count += obs.numel()
        self.obs_sum += obs.sum().item()
        self.obs_sq_sum += torch.square(obs).sum().item()
        self.res_sq_sum += torch.square(obs - pred).sum().item()
        self.abs_sum += torch.abs(obs - pred).sum().item()

    def compute(self) -> dict[str, float]:
        denominator = self.obs_sq_sum - self.obs_sum ** 2 / max(self.count, 1)
        return {
            "r2": 1.0 - self.res_sq_sum / denominator if denominator > 0 else float("nan"),
            "rmse": math.sqrt(self.res_sq_sum / max(self.count, 1)),
            "mae": self.abs_sum / max(self.count, 1),
            "valid_pixels": self.count,
        }


def init_run_dir(
    model_name: str, region: str, *, run_name: str | None = None, output_root: Path | None = None,
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name or f"{model_name}_{stamp}"
    root = Path(output_root) if output_root is not None else Path("outputs") / "v2_generative"
    run_dir = root / region / model_name / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _config_to_dict(config) -> dict:
    return asdict(config) if is_dataclass(config) else dict(config)


def write_run_metadata(run_dir: Path, config, *, stats: dict, extra: dict | None = None) -> None:
    payload = {
        "config": _config_to_dict(config),
        "source": source_revision(),
        "norm_stats": stats,
    }
    if extra:
        payload.update(extra)
    (run_dir / "run.json").write_text(json.dumps(payload, indent=2, default=str))


def append_history(run_dir: Path, row: dict) -> list[dict]:
    history_path = run_dir / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(row)
    history_path.write_text(json.dumps(history, indent=2, default=str))
    return history
