"""Losses, EMA, and a single rolling-fold training run."""

from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.chirps_v3.config import ROOT
from src.chirps_v3.dataset import RollingChirpsDataset
from src.chirps_v3.metrics import PooledMetrics
from src.chirps_v3.model import TemporalTerrainNet
from src.utils.config import Config

logger = Config.get_logger()


@dataclass
class TrainConfig:
    fold: str
    run_name: str
    context_days: int = 5
    width: int = 48
    dropout: float = 0.1
    fusion: str = "attention"
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 40
    batch_size: int = 2
    accumulation_steps: int = 2
    patience: int = 8
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4


def masked_mean(values, mask):
    mask = mask.expand_as(values)
    return values[mask].mean() if mask.any() else values.sum() * 0


def r2_aligned_loss(outputs, target_mm, mask):
    prediction = outputs["prediction_mm"]
    variance = masked_mean((target_mm - masked_mean(target_mm, mask)).square(), mask).clamp_min(1.0)
    physical_mse = masked_mean((prediction - target_mm).square(), mask) / variance
    log_huber = masked_mean(F.smooth_l1_loss(torch.log1p(prediction), torch.log1p(target_mm),
                                             reduction="none"), mask)
    # Logit-form BCE is numerically stable and explicitly safe under CUDA AMP.
    occurrence = masked_mean(F.binary_cross_entropy_with_logits(
        outputs["occurrence_logits"], (target_mm >= 1).float(), reduction="none"
    ), mask)
    gradient = masked_mean(torch.abs(prediction[..., 1:, :] - prediction[..., :-1, :] -
                                     target_mm[..., 1:, :] + target_mm[..., :-1, :]), mask[..., 1:, :])
    total = physical_mse + 0.2 * log_huber + 0.1 * occurrence + 0.03 * gradient
    return total, {"physical_mse": float(physical_mse.detach()), "log_huber": float(log_huber.detach()),
                   "occurrence": float(occurrence.detach()), "gradient": float(gradient.detach())}


class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    def update(self, model):
        for key, value in model.state_dict().items():
            self.state[key].mul_(self.decay).add_(value.detach(), alpha=1 - self.decay)


def _evaluate(model, loader, device):
    metrics = PooledMetrics(); model.eval()
    with torch.no_grad():
        for batch in loader:
            tensors = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            output = model(tensors["inputs_25km"], tensors["physics_10km"], tensors["terrain_5km"],
                           tensors["baseline_mm"], tensors["season"])
            metrics.update(output["prediction_mm"].cpu(), tensors["target_mm"].cpu(),
                           tensors["target_valid_mask"].cpu())
    return metrics.result()


def train(config: TrainConfig) -> dict:
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    device = torch.device(config.device if config.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    train_data = RollingChirpsDataset(config.fold, "train", config.context_days)
    validation_data = RollingChirpsDataset(config.fold, "validation", config.context_days)
    logger.info("v3 training start fold=%s run=%s train_samples=%d validation_samples=%d context=%d device=%s",
                config.fold, config.run_name, len(train_data), len(validation_data), config.context_days, device)
    loaders = {name: DataLoader(data, batch_size=config.batch_size, shuffle=name == "train",
               num_workers=config.num_workers, pin_memory=device.type == "cuda")
               for name, data in (("train", train_data), ("validation", validation_data))}
    model = TemporalTerrainNet(width=config.width, dropout=config.dropout, fusion=config.fusion).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = EMA(model); run_dir = ROOT / "runs" / config.fold / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_r2, stale, history = -float("inf"), 0, []
    for epoch in range(1, config.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(loaders["train"], 1):
            tensors = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(tensors["inputs_25km"], tensors["physics_10km"], tensors["terrain_5km"],
                               tensors["baseline_mm"], tensors["season"])
                loss, _ = r2_aligned_loss(output, tensors["target_mm"], tensors["target_valid_mask"])
                loss = loss / config.accumulation_steps
            scaler.scale(loss).backward(); losses.append(float(loss.detach()) * config.accumulation_steps)
            if step % config.accumulation_steps == 0 or step == len(loaders["train"]):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); ema.update(model)
        scheduler.step()
        live = {k: v.detach().clone() for k, v in model.state_dict().items()}; model.load_state_dict(ema.state)
        validation = _evaluate(model, loaders["validation"], device); model.load_state_dict(live)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), **validation}; history.append(record)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if validation["r2"] is not None and validation["r2"] > best_r2:
            best_r2, stale = validation["r2"], 0
            torch.save({"model_state": ema.state, "config": asdict(config), "metrics": validation}, run_dir / "best.pt")
        else:
            stale += 1
        logger.info("v3 epoch fold=%s run=%s epoch=%d/%d train_loss=%.6f val_r2=%s val_rmse=%.4f best_r2=%.6f stale=%d/%d lr=%.3e",
                    config.fold, config.run_name, epoch, config.epochs, record["train_loss"],
                    "null" if validation["r2"] is None else f'{validation["r2"]:.6f}',
                    validation["rmse"], best_r2, stale, config.patience, optimizer.param_groups[0]["lr"])
        if stale >= config.patience:
            logger.info("v3 early stop fold=%s run=%s epoch=%d", config.fold, config.run_name, epoch)
            break
    result = {"run_dir": str(run_dir), "best_r2": best_r2, "epochs": len(history), "config": asdict(config)}
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    logger.info("v3 training completed fold=%s run=%s best_r2=%.6f epochs=%d",
                config.fold, config.run_name, best_r2, len(history))
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--fold", required=True); parser.add_argument("--run-name", required=True)
    parser.add_argument("--context-days", type=int, default=5); parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=.1); parser.add_argument("--fusion", default="attention")
    parser.add_argument("--learning-rate", type=float, default=2e-4); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=40); parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=2); parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--device", default="cuda"); parser.add_argument("--num-workers", type=int, default=4)
    print(json.dumps(train(TrainConfig(**vars(parser.parse_args()))), indent=2))


if __name__ == "__main__": main()
