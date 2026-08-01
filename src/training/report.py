"""Generate reproducible diagnostic figures from frozen model checkpoints."""

from __future__ import annotations

import argparse
import heapq
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/precipitation-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_preprocessing.dataset import PrecipitationDataset
from src.data_preprocessing.quality import DATASET_VERSION
from src.models.multiscale_unet import MultiscalePrecipUNet
from src.utils.config import Config

logger = Config.get_logger()


def _load_models(run_directories: list[Path], device: torch.device):
    checkpoints = [
        torch.load(path / "best.pt", map_location="cpu", weights_only=False)
        for path in run_directories
    ]
    reference = checkpoints[0]
    target = reference["config"]["target"]
    fingerprint = reference["dataset_manifest_sha256"]
    for checkpoint in checkpoints:
        if checkpoint["config"]["target"] != target:
            raise ValueError("All checkpoints must use the same target")
        if checkpoint["dataset_manifest_sha256"] != fingerprint:
            raise ValueError("All checkpoints must use the same dataset manifest")
    models = []
    for checkpoint in checkpoints:
        config = checkpoint["config"]
        stats = checkpoint["stats"]
        model = MultiscalePrecipUNet(
            base_width=config["base_width"],
            dropout=config["dropout"],
            target_mean=stats["target_mean"],
            target_std=stats["target_std"],
            atmospheric_channels=18 * config.get("context_days", 1),
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models.append(model)
    return models, reference


def _plot_map(values, title: str, label: str, output: Path, *, cmap="viridis", symmetric=False):
    finite = np.isfinite(values)
    if not finite.any():
        return
    kwargs = {}
    if symmetric:
        bound = float(np.nanpercentile(np.abs(values), 98))
        kwargs = {"vmin": -bound, "vmax": bound}
    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    image = axis.imshow(values, origin="upper", cmap=cmap, **kwargs)
    axis.set_title(title)
    axis.set_xlabel("Grid column")
    axis.set_ylabel("Grid row")
    figure.colorbar(image, ax=axis, label=label, shrink=0.8)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_example(example: dict, output: Path) -> None:
    fields = (
        ("observation", "Observed"),
        ("era5", "ERA5"),
        ("prediction", "Model ensemble"),
        ("error", "Model − observed"),
    )
    positive = np.concatenate([
        example["observation"][example["mask"]],
        example["prediction"][example["mask"]],
        example["era5"][example["mask"]],
    ])
    maximum = max(float(np.quantile(positive, 0.99)), 1.0)
    figure, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    for axis, (key, title) in zip(axes, fields):
        values = np.where(example["mask"], example[key], np.nan)
        if key == "error":
            bound = max(float(np.nanpercentile(np.abs(values), 98)), 1.0)
            image = axis.imshow(values, cmap="RdBu_r", vmin=-bound, vmax=bound)
        else:
            image = axis.imshow(values, cmap="Blues", vmin=0.0, vmax=maximum)
        axis.set_title(title)
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, shrink=0.7)
    figure.suptitle(f"{example['date']} | observed mean={example['score']:.2f} mm/day")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _plot_histories(run_directories: list[Path], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for run_directory in run_directories:
        history = json.loads((run_directory / "history.json").read_text())
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["train_loss"] for row in history], label=run_directory.name)
        axes[1].plot(epochs, [row["val_r2"] for row in history], label=run_directory.name)
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Validation R²", xlabel="Epoch", ylabel="R²")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def generate_report(
    run_directories: list[Path],
    split: str,
    batch_size: int,
    device_name: str,
    example_count: int,
    num_workers: int,
) -> Path:
    device = torch.device(
        device_name
        if device_name.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    models, checkpoint = _load_models(run_directories, device)
    config = checkpoint["config"]
    stats = checkpoint["stats"]
    target = config["target"]
    dataset = PrecipitationDataset(target, split, config.get("context_days", 1))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    shape = (460, 720)
    prediction_sum = np.zeros(shape, dtype=np.float64)
    observation_sum = np.zeros(shape, dtype=np.float64)
    squared_error_sum = np.zeros(shape, dtype=np.float64)
    valid_count = np.zeros(shape, dtype=np.uint32)
    examples = []

    logger.info(
        "Generating report target=%s split=%s samples=%d device=%s",
        target, split, len(dataset), device,
    )
    with torch.no_grad():
        for batch_number, batch in enumerate(loader, start=1):
            tensors = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            predictions = []
            for model in models:
                predictions.append(model(
                    tensors["inputs_25km"],
                    tensors["phys_dem_10km"],
                    tensors["era5_precip_5km_norm"],
                    tensors["season"],
                )["prediction_mm"])
            prediction = torch.stack(predictions).mean(0).cpu().numpy()[:, 0]
            observation = torch.expm1(
                tensors["target_5km"] * stats["target_std"] + stats["target_mean"]
            ).clamp_min(0).cpu().numpy()[:, 0]
            era5 = torch.expm1(
                tensors["era5_precip_5km_norm"] * stats["target_std"]
                + stats["target_mean"]
            ).clamp_min(0).cpu().numpy()[:, 0]
            mask = (
                tensors["target_valid_mask_5km"] & tensors["land_mask_5km"]
            ).cpu().numpy()[:, 0]
            for index, date in enumerate(batch["date"]):
                valid = mask[index]
                prediction_sum[valid] += prediction[index][valid]
                observation_sum[valid] += observation[index][valid]
                squared_error_sum[valid] += np.square(
                    prediction[index][valid] - observation[index][valid]
                )
                valid_count[valid] += 1
                score = float(observation[index][valid].mean()) if valid.any() else 0.0
                record = {
                    "score": score,
                    "date": date,
                    "observation": observation[index],
                    "prediction": prediction[index],
                    "era5": era5[index],
                    "error": prediction[index] - observation[index],
                    "mask": valid,
                }
                if len(examples) < example_count:
                    heapq.heappush(examples, (score, date, record))
                elif score > examples[0][0]:
                    heapq.heapreplace(examples, (score, date, record))
            if batch_number % 25 == 0 or batch_number == len(loader):
                logger.info("Report progress batches=%d/%d", batch_number, len(loader))

    valid = valid_count > 0
    observation_mean = np.divide(
        observation_sum, valid_count, out=np.full(shape, np.nan), where=valid
    )
    prediction_mean = np.divide(
        prediction_sum, valid_count, out=np.full(shape, np.nan), where=valid
    )
    bias = prediction_mean - observation_mean
    rmse = np.sqrt(np.divide(
        squared_error_sum, valid_count, out=np.full(shape, np.nan), where=valid
    ))
    relative_bias = np.divide(
        bias * 100.0,
        observation_mean,
        out=np.full(shape, np.nan),
        where=valid & (observation_mean >= 0.1),
    )

    report_directory = (
        Path("outputs") / DATASET_VERSION / target / "final_report"
    )
    figures = report_directory / "figures"
    examples_directory = figures / "selected_days"
    examples_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        report_directory / f"spatial_diagnostics_{split}.npz",
        observation_mean=observation_mean.astype(np.float32),
        prediction_mean=prediction_mean.astype(np.float32),
        bias=bias.astype(np.float32),
        relative_bias=relative_bias.astype(np.float32),
        rmse=rmse.astype(np.float32),
        valid_count=valid_count,
    )
    _plot_map(bias, f"{target.upper()} mean bias", "mm/day", figures / "mean_bias_mm_day.png", cmap="RdBu_r", symmetric=True)
    _plot_map(relative_bias, f"{target.upper()} relative bias", "%", figures / "relative_bias_percent.png", cmap="RdBu_r", symmetric=True)
    _plot_map(rmse, f"{target.upper()} RMSE", "mm/day", figures / "rmse_map.png")
    _plot_map(valid_count.astype(float), f"{target.upper()} valid sample count", "days", figures / "valid_sample_count.png")
    _plot_histories(run_directories, figures / "training_history.png")
    selected = sorted((record for _, _, record in examples), key=lambda item: item["score"], reverse=True)
    for record in selected:
        _plot_example(record, examples_directory / f"{record['date']}.png")

    metrics_path = run_directories[0] / f"metrics_{split}.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    report_path = report_directory / "report.md"
    model_metrics = metrics.get("metrics", {}).get("model", {})
    acceptance = metrics.get("acceptance", {})
    lines = [
        f"# {target.upper()} final report",
        "",
        f"- Split: `{split}`",
        f"- Ensemble members: {len(run_directories)}",
        f"- R²: {model_metrics.get('r2', 'not evaluated')}",
        f"- RMSE: {model_metrics.get('rmse', 'not evaluated')}",
        f"- MAE: {model_metrics.get('mae', 'not evaluated')}",
        f"- R² ≥ 0.40: {acceptance.get('pooled_r2_at_least_0_40', 'not evaluated')}",
        "",
        "Figures and spatial arrays use only target-valid land pixels.",
    ]
    report_path.write_text("\n".join(lines) + "\n")
    logger.info("Report completed output=%s", report_path)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True, nargs="+")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--example-count", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    output = generate_report(
        args.run_dir,
        args.split,
        args.batch_size,
        args.device,
        args.example_count,
        args.num_workers,
    )
    print(output)


if __name__ == "__main__":
    main()
