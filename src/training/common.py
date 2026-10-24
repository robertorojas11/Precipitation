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


def input_moments(stats: dict) -> tuple[ChannelMoments, ChannelMoments]:
    """``(cp_moments, lsp_moments)`` -- mirrors ``target_moments``, for
    denormalizing the two ``inputs_norm`` channels (``[cp, lsp]``, see
    ``prepare_hourly_dataset.prepare_region``) back to physical mm.
    """
    return ChannelMoments(**stats["pooled"]["cp"]), ChannelMoments(**stats["pooled"]["lsp"])


def total_precip_input_mm(
    inputs_norm: torch.Tensor, cp_moments: ChannelMoments, lsp_moments: ChannelMoments,
) -> torch.Tensor:
    """Physical-mm ERA5 total precipitation (cp + lsp) on the HR grid, from
    the normalized ``inputs`` tensor's first two channels -- the
    ``total_precip_input`` that ``DeterministicResNet.forward``'s
    residual-over-input branch expects (``src/models/deterministic_resnet.py``).
    """
    cp_mm = denormalize_to_mm(inputs_norm[:, 0:1], cp_moments)
    lsp_mm = denormalize_to_mm(inputs_norm[:, 1:2], lsp_moments)
    return cp_mm + lsp_mm


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
    """Mean of ``values`` over pixels where ``mask`` is truthy.

    **Shape bug, found and fixed 2026-10-19.** ``mask`` is usually (B, H, W)
    (``loss_mask``'s native shape) while ``values`` is (B, 1, H, W). Without
    this guard, ``values * mask`` silently broadcasts the size-1 channel
    axis against the *batch* axis instead (both are non-1 after padding),
    producing a (B, B, H, W) cross-batch outer product -- confirmed
    numerically (a 3x4x4 synthetic case gave 1.145 instead of the correct
    0.363). This was live in ``train_baseline.py``'s and diffusion stage-1's
    training loss (not validation -- ``PooledMetrics.update`` already had
    this same guard), so every gradient step was computed over a corrupted
    mix of unrelated batch elements' residuals and masks instead of each
    sample's own pixels.
    """
    if mask.dim() == values.dim() - 1:
        mask = mask.unsqueeze(1)
    weights = mask.to(values.dtype).expand_as(values)
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

    **Residual winsorizing, and why.** Measured on real diffusion-stage1/2
    training (run4): a single pathological sample in a validation pass (one
    pixel, one batch) can drive a squared-error-based ``r2`` to something
    like ``-4e18`` or ``-738682`` -- not genuine model badness, but one
    numerically-unstable prediction swamping a sum over otherwise-sane
    residuals. ERA5's own clip ceiling for this dataset is ~130mm/hr
    (``norm_stats.era5_clip_range_imerg_mm_per_hr``), so no physically
    plausible residual should ever exceed a few hundred mm -- ``
    max_abs_residual`` (default 500.0mm, generous on purpose) clips the
    residual *before* squaring, so one numerical blow-up no longer renders
    the whole epoch's r2 unreadable while leaving every ordinary residual
    untouched. ``clipped_count`` in ``compute()`` surfaces exactly how often
    this fires -- if it's ever more than a handful of pixels, that is itself
    a signal worth investigating, not something to silently absorb.
    """

    def __init__(self, *, max_abs_residual: float | None = 500.0) -> None:
        self.count = 0
        self.obs_sum = 0.0
        self.obs_sq_sum = 0.0
        self.res_sq_sum = 0.0
        self.abs_sum = 0.0
        self.log_abs_sum = 0.0
        self.clipped_count = 0
        self.max_abs_residual = max_abs_residual

    def update(self, prediction: torch.Tensor, observation: torch.Tensor, mask: torch.Tensor) -> None:
        if mask.dim() == prediction.dim() - 1:
            mask = mask.unsqueeze(1)
        mask = mask.expand_as(prediction)
        pred = prediction[mask].double()
        obs = observation[mask].double()
        self.count += obs.numel()
        self.obs_sum += obs.sum().item()
        self.obs_sq_sum += torch.square(obs).sum().item()

        residual = obs - pred
        if self.max_abs_residual is not None:
            clipped = residual.clamp(-self.max_abs_residual, self.max_abs_residual)
            self.clipped_count += int((residual != clipped).sum().item())
            residual = clipped
        self.res_sq_sum += torch.square(residual).sum().item()
        self.abs_sum += torch.abs(residual).sum().item()

        # log-space MAE: scale-robust (a 1mm vs 2mm miss and a 50mm vs 100mm
        # miss both read as log1p-diff ~= 0.69), so one extreme-rain pixel
        # can't dominate it the way it dominates a squared-error metric --
        # a second, differently-robust read on the same validation pass.
        log_residual = torch.log1p(obs.clamp_min(0.0)) - torch.log1p(pred.clamp_min(0.0))
        self.log_abs_sum += torch.abs(log_residual).sum().item()

    def compute(self) -> dict[str, float]:
        denominator = self.obs_sq_sum - self.obs_sum ** 2 / max(self.count, 1)
        return {
            "r2": 1.0 - self.res_sq_sum / denominator if denominator > 0 else float("nan"),
            "rmse": math.sqrt(self.res_sq_sum / max(self.count, 1)),
            "mae": self.abs_sum / max(self.count, 1),
            "log_mae": self.log_abs_sum / max(self.count, 1),
            "clipped_count": self.clipped_count,
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
