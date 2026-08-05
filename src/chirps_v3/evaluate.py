"""Evaluate a v3 checkpoint on its chronological validation fold."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.chirps_v3.dataset import RollingChirpsDataset
from src.chirps_v3.metrics import PooledMetrics
from src.chirps_v3.model import TemporalTerrainNet


def evaluate(checkpoint_path: Path, batch_size=2, device_name="cuda", num_workers=4):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False); config = checkpoint["config"]
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = TemporalTerrainNet(width=config["width"], dropout=config["dropout"], fusion=config["fusion"]).to(device)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    dataset = RollingChirpsDataset(config["fold"], "validation", config["context_days"])
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=device.type == "cuda")
    metric, baseline, years = PooledMetrics(), PooledMetrics(), {}
    with torch.no_grad():
        for batch in loader:
            tensors = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            output = model(tensors["inputs_25km"], tensors["physics_10km"], tensors["terrain_5km"], tensors["baseline_mm"], tensors["season"])
            pred, obs, mask = output["prediction_mm"].cpu(), tensors["target_mm"].cpu(), tensors["target_valid_mask"].cpu()
            metric.update(pred, obs, mask); baseline.update(tensors["baseline_mm"].cpu(), obs, mask)
            for i, date in enumerate(batch["date"]):
                years.setdefault(date[:4], PooledMetrics()).update(pred[i], obs[i], mask[i])
    result = {"checkpoint": str(checkpoint_path), "fold": config["fold"], "model": metric.result(),
              "era5": baseline.result(), "by_year": {k: v.result() for k, v in sorted(years.items())}}
    output = checkpoint_path.parent / "metrics_validation.json"; output.write_text(json.dumps(result, indent=2) + "\n"); return result


def main():
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--batch-size", type=int, default=2); p.add_argument("--device", default="cuda"); p.add_argument("--num-workers", type=int, default=4)
    a = p.parse_args(); print(json.dumps(evaluate(a.checkpoint, a.batch_size, a.device, a.num_workers), indent=2))
if __name__ == "__main__": main()

