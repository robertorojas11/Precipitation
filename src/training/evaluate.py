"""Evaluate trained models and fixed baselines on explicit masks."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_preprocessing.dataset import PrecipitationDataset
from src.data_preprocessing.quality import DATASET_VERSION
from src.models.multiscale_unet import MultiscalePrecipUNet
from src.utils.config import Config

logger = Config.get_logger()


def _json_default(value):
    """Convert NumPy scalar results to JSON-native Python values."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class VerificationAccumulator:
    def __init__(self):
        self.n = 0
        self.obs_sum = self.obs_sq = self.sse = self.sae = 0.0
        self.events = {threshold: [0, 0, 0] for threshold in (1.0, 10.0, 25.0)}

    def update(self, pred, obs, mask):
        pred = np.asarray(pred)[mask].astype(np.float64)
        obs = np.asarray(obs)[mask].astype(np.float64)
        self.n += obs.size
        self.obs_sum += obs.sum()
        self.obs_sq += np.square(obs).sum()
        self.sse += np.square(obs - pred).sum()
        self.sae += np.abs(obs - pred).sum()
        for threshold, counts in self.events.items():
            pred_event, obs_event = pred >= threshold, obs >= threshold
            counts[0] += int(np.sum(pred_event & obs_event))
            counts[1] += int(np.sum(pred_event & ~obs_event))
            counts[2] += int(np.sum(~pred_event & obs_event))

    def merge(self, other):
        self.n += other.n
        self.obs_sum += other.obs_sum
        self.obs_sq += other.obs_sq
        self.sse += other.sse
        self.sae += other.sae
        for threshold in self.events:
            for index in range(3):
                self.events[threshold][index] += other.events[threshold][index]
        return self

    def metrics(self):
        denominator = self.obs_sq - self.obs_sum**2 / max(self.n, 1)
        result = {
            "r2": 1.0 - self.sse / denominator if denominator > 0 else None,
            "rmse": math.sqrt(self.sse / max(self.n, 1)),
            "mae": self.sae / max(self.n, 1),
            "valid_pixels": self.n,
        }
        for threshold, (tp, fp, fn) in self.events.items():
            suffix = str(int(threshold))
            result[f"csi_{suffix}"] = tp / max(tp + fp + fn, 1)
            result[f"pod_{suffix}"] = tp / max(tp + fn, 1)
            result[f"far_{suffix}"] = fp / max(tp + fp, 1)
            result[f"frequency_bias_{suffix}"] = (tp + fp) / max(tp + fn, 1)
        return result


def _bootstrap_months(months, draws=2000, seed=42):
    keys = sorted(months)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        merged = VerificationAccumulator()
        for key in rng.choice(keys, size=len(keys), replace=True):
            merged.merge(months[key])
        value = merged.metrics()["r2"]
        if value is not None:
            values.append(value)
    return {
        "draws": len(values),
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def evaluate(
    run_dirs: list[Path],
    split: str = "test",
    batch_size: int = 2,
    device_name: str = "cuda",
    num_workers: int = 4,
):
    if not run_dirs:
        raise ValueError("At least one run directory is required")
    checkpoints = [
        torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        for run_dir in run_dirs
    ]
    checkpoint = checkpoints[0]
    config, stats = checkpoint["config"], checkpoint["stats"]
    target = config["target"]
    for candidate in checkpoints[1:]:
        if candidate["config"]["target"] != target:
            raise ValueError("All ensemble runs must use the same target")
        if candidate["dataset_manifest_sha256"] != checkpoint["dataset_manifest_sha256"]:
            raise ValueError("All ensemble runs must use the same dataset manifest")
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    dataset = PrecipitationDataset(target, split, config.get("context_days", 1))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    models = []
    for candidate in checkpoints:
        candidate_config = candidate["config"]
        model = MultiscalePrecipUNet(
            base_width=candidate_config["base_width"], dropout=candidate_config["dropout"],
            target_mean=stats["target_mean"], target_std=stats["target_std"],
            atmospheric_channels=18 * candidate_config.get("context_days", 1),
        ).to(device)
        model.load_state_dict(candidate["model_state"])
        model.eval()
        models.append(model)
    climatology_path = Path(Config.LOCAL_DATA_DIR) / DATASET_VERSION / "metadata" / f"monthly_climatology_{target}.npz"
    with np.load(climatology_path) as values:
        climatology = values["precipitation_mm"]

    accumulators = {name: VerificationAccumulator() for name in ("model", "era5", "climatology", "dry")}
    years = defaultdict(VerificationAccumulator)
    months = defaultdict(VerificationAccumulator)
    logger.info(
        "Evaluating target=%s split=%s samples=%d ensemble=%d device=%s",
        target, split, len(dataset), len(models), device,
    )
    with torch.no_grad():
        for batch_number, batch in enumerate(loader, start=1):
            tensor_batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            member_predictions = []
            for model in models:
                output = model(
                    tensor_batch["inputs_25km"], tensor_batch["phys_dem_10km"],
                    tensor_batch["era5_precip_5km_norm"], tensor_batch["season"],
                )
                member_predictions.append(output["prediction_mm"])
            prediction = torch.stack(member_predictions).mean(dim=0).cpu().numpy()
            observation = torch.expm1(
                tensor_batch["target_5km"] * stats["target_std"] + stats["target_mean"]
            ).clamp_min(0.0).cpu().numpy()
            era5 = torch.expm1(
                tensor_batch["era5_precip_5km_norm"] * stats["target_std"] + stats["target_mean"]
            ).clamp_min(0.0).cpu().numpy()
            mask = (tensor_batch["target_valid_mask_5km"] & tensor_batch["land_mask_5km"]).cpu().numpy()
            for index, date in enumerate(batch["date"]):
                month_index = int(date[5:7]) - 1
                item_mask = mask[index]
                item_obs = observation[index]
                item_pred = prediction[index]
                accumulators["model"].update(item_pred, item_obs, item_mask)
                accumulators["era5"].update(era5[index], item_obs, item_mask)
                accumulators["climatology"].update(climatology[month_index][None], item_obs, item_mask)
                accumulators["dry"].update(np.zeros_like(item_obs), item_obs, item_mask)
                item_accumulator = VerificationAccumulator()
                item_accumulator.update(item_pred, item_obs, item_mask)
                years[date[:4]].merge(item_accumulator)
                months[date[:7]].merge(item_accumulator)
            if batch_number % 25 == 0 or batch_number == len(loader):
                logger.info("Evaluation progress batches=%d/%d", batch_number, len(loader))

    results = {
        "target": target,
        "split": split,
        "checkpoints": [str(run_dir / "best.pt") for run_dir in run_dirs],
        "dataset_manifest_sha256": checkpoint["dataset_manifest_sha256"],
        "metrics": {name: accumulator.metrics() for name, accumulator in accumulators.items()},
        "model_by_year": {key: value.metrics() for key, value in sorted(years.items())},
        "model_r2_month_bootstrap": _bootstrap_months(months),
    }
    model_metrics = results["metrics"]["model"]
    results["acceptance"] = {
        "pooled_r2_at_least_0_40": bool(model_metrics["r2"] >= 0.40),
        "every_year_r2_at_least_0_20": bool(all(
            value["r2"] >= 0.20 for value in results["model_by_year"].values()
        )),
        "bootstrap_lower_at_least_0_35": bool(
            results["model_r2_month_bootstrap"]["lower_95"] >= 0.35
        ),
        "beats_era5_r2": bool(
            model_metrics["r2"] > results["metrics"]["era5"]["r2"]
        ),
        "beats_climatology_r2": bool(
            model_metrics["r2"] > results["metrics"]["climatology"]["r2"]
        ),
    }
    output = run_dirs[0] / f"metrics_{split}.json"
    temporary_output = output.with_suffix(".json.tmp")
    with temporary_output.open("w") as stream:
        json.dump(results, stream, indent=2, default=_json_default)
        stream.write("\n")
    temporary_output.replace(output)
    logger.info("Evaluation completed output=%s model_metrics=%s", output, model_metrics)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True, nargs="+")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(
        evaluate(
            args.run_dir,
            args.split,
            args.batch_size,
            args.device,
            args.num_workers,
        ),
        indent=2,
        default=_json_default,
    ))


if __name__ == "__main__":
    main()
