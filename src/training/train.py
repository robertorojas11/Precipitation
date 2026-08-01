"""Train the precipitation model with masked objectives and reproducible checkpoints."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import subprocess

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data_preprocessing.dataset import PrecipitationDataset
from src.data_preprocessing.quality import DATASET_VERSION
from src.models.multiscale_unet import MultiscalePrecipUNet
from src.utils.config import Config


@dataclass
class TrainingConfig:
    target: str
    epochs: int = 80
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    base_width: int = 32
    dropout: float = 0.0
    event_weight: float = 2.0
    patience: int = 10
    warmup_epochs: int = 3
    seed: int = 42
    num_workers: int = 8
    device: str = "cuda"
    run_name: str | None = None
    context_days: int = 1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _neighbor_gradient_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x_mask = mask[..., :, 1:] & mask[..., :, :-1]
    y_mask = mask[..., 1:, :] & mask[..., :-1, :]
    pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    return _masked_mean(torch.abs(pred_x - target_x), x_mask) + _masked_mean(
        torch.abs(pred_y - target_y), y_mask
    )


def _multiscale_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    total = prediction.new_tensor(0.0)
    for factor in (2, 5, 10):
        size = (prediction.shape[-2] // factor, prediction.shape[-1] // factor)
        weights = F.interpolate(mask.float(), size=size, mode="area")
        pred = F.interpolate(prediction * mask, size=size, mode="area") / weights.clamp_min(1e-6)
        obs = F.interpolate(target * mask, size=size, mode="area") / weights.clamp_min(1e-6)
        total = total + _masked_mean(torch.abs(pred - obs), weights >= 0.999)
    return total / 3.0


def compute_losses(outputs: dict, target_norm: torch.Tensor, mask: torch.Tensor, mean: float, std: float, event_weight: float):
    target_log = target_norm * std + mean
    target_mm = torch.expm1(target_log).clamp_min(0.0)
    prediction_mm = outputs["prediction_mm"]
    wet = target_mm >= 1.0
    valid_wet = wet & mask

    valid_count = mask.sum().clamp_min(1)
    wet_count = valid_wet.sum().clamp_min(1)
    dry_count = (mask & ~wet).sum().clamp_min(1)
    pos_weight = (dry_count.float() / wet_count.float()).clamp(1.0, 20.0)
    occurrence_raw = F.binary_cross_entropy_with_logits(
        outputs["occurrence_logits"], wet.float(), reduction="none", pos_weight=pos_weight
    )
    occurrence = _masked_mean(occurrence_raw, mask)

    amount_raw = F.smooth_l1_loss(outputs["positive_log_amount"], target_log, reduction="none")
    amount = _masked_mean(amount_raw, valid_wet)

    pixel_weights = torch.ones_like(target_mm)
    pixel_weights = pixel_weights + event_weight * (target_mm >= 10.0)
    pixel_weights = pixel_weights + (2.0 * event_weight) * (target_mm >= 25.0)
    physical = (torch.abs(prediction_mm - target_mm) * pixel_weights * mask).sum()
    physical = physical / (pixel_weights * mask).sum().clamp_min(1.0)
    gradient = _neighbor_gradient_loss(prediction_mm, target_mm, mask)
    multiscale = _multiscale_loss(prediction_mm, target_mm, mask.float())
    total = occurrence + amount + physical + 0.1 * gradient + 0.25 * multiscale
    return total, {
        "total": total.detach(),
        "occurrence": occurrence.detach(),
        "amount": amount.detach(),
        "physical": physical.detach(),
        "gradient": gradient.detach(),
        "multiscale": multiscale.detach(),
    }


class PooledMetrics:
    def __init__(self):
        self.count = 0
        self.obs_sum = 0.0
        self.obs_sq_sum = 0.0
        self.res_sq_sum = 0.0
        self.abs_sum = 0.0

    def update(self, prediction: torch.Tensor, observation: torch.Tensor, mask: torch.Tensor):
        pred = prediction[mask].double()
        obs = observation[mask].double()
        self.count += obs.numel()
        self.obs_sum += obs.sum().item()
        self.obs_sq_sum += torch.square(obs).sum().item()
        self.res_sq_sum += torch.square(obs - pred).sum().item()
        self.abs_sum += torch.abs(obs - pred).sum().item()

    def compute(self) -> dict[str, float]:
        denominator = self.obs_sq_sum - self.obs_sum**2 / max(self.count, 1)
        return {
            "r2": 1.0 - self.res_sq_sum / denominator if denominator > 0 else float("nan"),
            "rmse": math.sqrt(self.res_sq_sum / max(self.count, 1)),
            "mae": self.abs_sum / max(self.count, 1),
            "valid_pixels": self.count,
        }


def _to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _forward(model, batch):
    return model(
        batch["inputs_25km"],
        batch["phys_dem_10km"],
        batch["era5_precip_5km_norm"],
        batch["season"],
    )


def validate(model, loader, device, mean, std) -> dict[str, float]:
    model.eval()
    metrics = PooledMetrics()
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            outputs = _forward(model, batch)
            target_mm = torch.expm1(batch["target_5km"] * std + mean).clamp_min(0.0)
            mask = batch["target_valid_mask_5km"] & batch["land_mask_5km"]
            metrics.update(outputs["prediction_mm"], target_mm, mask)
    return metrics.compute()


def _source_revision() -> dict:
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return {"commit": revision, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def train(config: TrainingConfig) -> Path:
    seed_everything(config.seed)
    requested_device = config.device if config.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    root = Path(Config.LOCAL_DATA_DIR) / DATASET_VERSION
    with (root / "metadata" / f"norm_stats_{config.target}.json").open() as stream:
        stats = json.load(stream)
    with (root / "metadata" / f"fast_manifest_{config.target}.json").open() as stream:
        manifest = json.load(stream)

    train_dataset = PrecipitationDataset(config.target, "train", config.context_days)
    val_dataset = PrecipitationDataset(config.target, "val", config.context_days)
    generator = torch.Generator().manual_seed(config.seed)
    common = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)

    model = MultiscalePrecipUNet(
        base_width=config.base_width,
        dropout=config.dropout,
        target_mean=stats["target_mean"],
        target_std=stats["target_std"],
        atmospheric_channels=18 * config.context_days,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.epochs - config.warmup_epochs, 1), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = config.run_name or f"unet_{stamp}_seed{config.seed}"
    run_dir = Path("outputs") / DATASET_VERSION / config.target / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    run_metadata = {
        "config": asdict(config),
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "source": _source_revision(),
        "stats": stats,
    }
    with (run_dir / "run.json").open("w") as stream:
        json.dump(run_metadata, stream, indent=2)

    best_r2 = -float("inf")
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        if epoch <= config.warmup_epochs:
            warmup_scale = epoch / max(config.warmup_epochs, 1)
            for group in optimizer.param_groups:
                group["lr"] = config.learning_rate * warmup_scale

        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            autocast = torch.amp.autocast("cuda", enabled=device.type == "cuda") if device.type == "cuda" else nullcontext()
            with autocast:
                outputs = _forward(model, batch)
                mask = batch["target_valid_mask_5km"] & batch["land_mask_5km"]
                loss, _ = compute_losses(
                    outputs,
                    batch["target_5km"],
                    mask,
                    stats["target_mean"],
                    stats["target_std"],
                    config.event_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            batches += 1

        if epoch > config.warmup_epochs:
            cosine.step()
        validation = validate(
            model, val_loader, device, stats["target_mean"], stats["target_std"]
        )
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(batches, 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"val_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": cosine.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": epoch,
            "best_validation_r2": max(best_r2, validation["r2"]),
            "config": asdict(config),
            "dataset_manifest_sha256": manifest["manifest_sha256"],
            "stats": stats,
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if validation["r2"] > best_r2:
            best_r2 = validation["r2"]
            epochs_without_improvement = 0
            torch.save(checkpoint, run_dir / "best.pt")
        else:
            epochs_without_improvement += 1
        with (run_dir / "history.json").open("w") as stream:
            json.dump(history, stream, indent=2)
        if epochs_without_improvement >= config.patience:
            break
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=["chirps", "oya"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--base-width", type=int, default=32, choices=[32, 64])
    parser.add_argument("--dropout", type=float, default=0.0, choices=[0.0, 0.1])
    parser.add_argument("--event-weight", type=float, default=2.0, choices=[2.0, 4.0])
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name")
    parser.add_argument("--context-days", type=int, default=1, choices=[1, 3])
    args = parser.parse_args()
    config = TrainingConfig(
        target=args.target,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        base_width=args.base_width,
        dropout=args.dropout,
        event_weight=args.event_weight,
        patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        run_name=args.run_name,
        context_days=args.context_days,
    )
    print(train(config))


if __name__ == "__main__":
    main()
