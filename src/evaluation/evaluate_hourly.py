r"""Stage 7 -- evaluate a trained checkpoint (from stage 6) against the
metrics stage 8 of docs/decisions/0002-implementacion-v2.md asks for:
CRPS, percentile-MAE, (m/me)FSS, RAPSD, rank histogram, the continuity
metrics (R2/RMSE/MAE/BIAS + block bootstrap), and full-field RMSE/SSIM.
Full narrative: docs/stages_v2/07-evaluate.md.

This module is **wiring, not new metrics**: every formula already lives in
``src/evaluation/{probabilistic_metrics,deterministic_metrics,baselines,
model_comparison}.py`` (built generically, ahead of any trainable dataset --
``docs/architecture_v3.md``'s own B.5 rows say "DONE ... no real training
run yet to feed it"). What was missing, and what this stage adds, is the
part that turns a stage-6 checkpoint plus a prepared split into the actual
numbers those functions expect.

```bash
python -m src.evaluation.evaluate_hourly --model baseline  --region mexico_pacifico --split test \
    --checkpoint outputs/v2_generative/mexico_pacifico/deterministic_resnet/<run>/best.pt
python -m src.evaluation.evaluate_hourly --model gan       --region mexico_pacifico --split test \
    --checkpoint outputs/v2_generative/mexico_pacifico/gan/<run>/best.pt
python -m src.evaluation.evaluate_hourly --model diffusion --region mexico_pacifico --split test \
    --stage1-checkpoint .../diffusion_stage1_regression/<run>/best.pt \
    --stage2-checkpoint .../diffusion_stage2_residual/<run>/best.pt
python -m src.evaluation.evaluate_hourly --model bilinear  --region puerto_rico_caribe --split test
```

**Ensemble convention** (shared with ``src/evaluation/probabilistic_metrics
.py``): every model here is evaluated as an ``(M, B, 1, H, W)`` ensemble of
*physical* mm/hr predictions. The deterministic baseline and the bilinear
baseline use ``M=1`` (CRPS correctly reduces to MAE, P1 M181/P2 M247,
verified by `test_diffusion_residual`/`test_probabilistic_metrics`'s own
tests, not re-verified here). The GAN reuses ``ConditionalGenerator.sample()``
directly (frozen per-member dropout seed, P1 M160). Candidate B draws ``M``
independent Heun-sampled residuals and adds each to the frozen stage 1 mean.

**The bilinear-interpolation baseline needs physical, pre-clip ERA5 -- not
the prepared split.** P2 M225 / P1 M199's baseline is "ERA5 total
precipitation bilinearly interpolated to the target grid". Stage 2
(`build_hourly_dataset.py`) already reprojects ERA5 cp/lsp onto the
canonical HR grid with ``Resampling.bilinear`` (confirmed by reading that
module), so the **build-stage** NPZ's ``inputs`` field, summed over its two
channels, *is already* this baseline by construction -- no second bilinear
resample is needed, and ``src.evaluation.baselines.bilinear_interpolation``
is not called here. The *prepared* (stage 4) tensors cannot be used for
this instead: M216's IMERG-range clip is lossy (values outside the clipped
range are not recoverable), so this baseline is computed straight from the
build-stage index/NPZs (`_iter_build_physical_frames`), independent of
stage 4 entirely.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_preprocessing.prepare_hourly_dataset import _build_root as _default_build_root
from src.evaluation import deterministic_metrics as det
from src.evaluation import probabilistic_metrics as prob
from src.evaluation.model_comparison import ModelRunResult
from src.models.deterministic_resnet import DeterministicResNet, ResNetConfig
from src.models.diffusion_residual import EDMPrecond, HeunSamplerConfig, heun_sample
from src.models.generator import ConditionalGenerator, GeneratorConfig
from src.models.regression_stage import RegressionStageConfig, RegressionUNet
from src.training.common import denormalize_to_mm, load_norm_stats, loss_mask, target_moments, to_device
from src.training.dataset import HOURS_PER_DAY, PreparedHourlyDataset
from src.utils.logging import configure_logging

logger = configure_logging(name="precipitation.evaluate_hourly")

# IMERG's 0.1 deg grid in km, nominal equatorial conversion (1 deg ~= 111 km)
# -- not latitude-corrected per region, a documented known limitation
# (docs/stages_v2/07-evaluate.md), since FSS's own scale buckets (P1 M184,
# tens to hundreds of km) are coarse enough that this approximation does not
# change which bucket a scale falls into for this project's three regions.
IMERG_PIXEL_KM = 0.1 * 111.0
DEFAULT_ENSEMBLE_MEMBERS = 3
MAX_SPATIAL_FRAMES = 20  # FSS/RAPSD/SSIM/rank-histogram are per-frame; capped for cost


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    tmp.replace(path)


def _build_root(build_root: Path | None) -> Path:
    return Path(build_root) if build_root is not None else _default_build_root()


def run_evaluation(
    region: str,
    split: str,
    predict_fn,
    *,
    prepared_root: Path | None = None,
    batch_size: int = 8,
    device: str = "cpu",
    n_members: int = 1,
    max_spatial_frames: int = MAX_SPATIAL_FRAMES,
    pixel_km: float = IMERG_PIXEL_KM,
) -> dict:
    """Shared evaluation core for every model here.

    ``predict_fn(inputs, invariants) -> (M, B, 1, H, W)`` physical mm/hr
    ensemble -- the one thing that differs between candidates; everything
    else (masking, pooling, metric calls) is identical, so every candidate
    is scored by literally the same code.

    Pooled metrics (R2/RMSE/MAE/BIAS, percentile-MAE) are accumulated over
    **every** valid pixel in the split, concatenated into flat arrays --
    correct (percentiles and R2 need the global distribution, not a
    per-batch average of them), but means the whole split's valid pixels
    are held in memory at once, a known limitation at full historical scale
    (docs/stages_v2/07-evaluate.md). CRPS is pooled as a running
    valid-pixel-weighted average of per-batch CRPS instead, since CRPS *is*
    already a plain mean over pixels and the two are mathematically
    identical -- this one metric does not need the flat-array accumulation.
    The inherently 2D metrics (FSS, RAPSD, full-field SSIM, rank histogram)
    only ever operate on one frame at a time, so they are computed on an
    evenly-spaced sample of at most ``max_spatial_frames`` frames across the
    split rather than all of them.
    """
    device_t = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    stats = load_norm_stats(prepared_root)
    moments = target_moments(stats)
    dataset = PreparedHourlyDataset(region, split, prepared_root=prepared_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total = len(dataset)
    spatial_stride = max(1, total // max(max_spatial_frames, 1))

    pooled_pred: list[np.ndarray] = []
    pooled_obs: list[np.ndarray] = []
    crps_weighted_sum = 0.0
    crps_weight = 0.0
    rank_counts = np.zeros(n_members + 1, dtype=np.int64)
    fss_scores: list[float] = []
    ssim_scores: list[float] = []
    rmse_full_scores: list[float] = []
    pred_spectra: list[np.ndarray] = []
    obs_spectra: list[np.ndarray] = []

    global_index = 0
    with torch.no_grad():
        for batch in loader:
            batch_t = to_device(batch, device_t)
            ensemble = predict_fn(batch_t["inputs"], batch_t["invariants"])  # (M, B, 1, H, W)
            target_mm = denormalize_to_mm(batch_t["target"], moments)
            mask = loss_mask(batch_t)
            mean_pred = ensemble.mean(dim=0)

            mask_np = mask.cpu().numpy()
            mean_pred_np = mean_pred.squeeze(1).cpu().numpy()
            target_np = target_mm.squeeze(1).cpu().numpy()
            ensemble_np = ensemble.squeeze(2).cpu().numpy()  # (M, B, H, W)

            pooled_pred.append(mean_pred_np[mask_np])
            pooled_obs.append(target_np[mask_np])

            batch_crps = prob.crps_ensemble(ensemble_np, target_np, mask_np)
            weight = float(mask_np.sum())
            if weight > 0 and np.isfinite(batch_crps):
                crps_weighted_sum += batch_crps * weight
                crps_weight += weight

            batch_size_actual = mean_pred_np.shape[0]
            for item in range(batch_size_actual):
                if global_index % spatial_stride == 0:
                    frame_mask = mask_np[item]
                    frame_pred = mean_pred_np[item]
                    frame_obs = target_np[item]
                    frame_ensemble = ensemble_np[:, item]
                    rank_counts += prob.rank_histogram(frame_ensemble, frame_obs, frame_mask)
                    fss_scores.append(prob.mean_fss(frame_pred, frame_obs, pixel_km, frame_mask))
                    full = det.full_field_rmse_ssim(frame_pred, frame_obs, frame_mask)
                    rmse_full_scores.append(full["rmse"])
                    ssim_scores.append(full["ssim"])
                    pred_spectra.append(prob.rapsd(np.where(frame_mask, frame_pred, 0.0)))
                    obs_spectra.append(prob.rapsd(np.where(frame_mask, frame_obs, 0.0)))
                global_index += 1

    pred_flat = np.concatenate(pooled_pred) if pooled_pred else np.array([])
    obs_flat = np.concatenate(pooled_obs) if pooled_obs else np.array([])

    result: dict = {
        "region": region, "split": split, "sample_count": total,
        "pooled": det.pooled_scores(pred_flat, obs_flat),
        "percentile_mae": prob.percentile_mae(pred_flat, obs_flat),
        "crps": crps_weighted_sum / crps_weight if crps_weight > 0 else float("nan"),
        "spatial_frames_sampled": len(fss_scores),
        "mean_fss": float(np.mean(fss_scores)) if fss_scores else float("nan"),
        "full_field_rmse": float(np.mean(rmse_full_scores)) if rmse_full_scores else float("nan"),
        "full_field_ssim": float(np.mean(ssim_scores)) if ssim_scores else float("nan"),
        "rank_histogram": rank_counts.tolist(),
    }
    if pred_spectra and obs_spectra:
        length = min(min(len(s) for s in pred_spectra), min(len(s) for s in obs_spectra))
        if length > 0:
            pred_spectrum = np.mean([s[:length] for s in pred_spectra], axis=0)
            obs_spectrum = np.mean([s[:length] for s in obs_spectra], axis=0)
            result["rapsd_log_distance"] = float(
                np.mean(np.abs(np.log1p(pred_spectrum) - np.log1p(obs_spectrum)))
            )
    return result


def _wrap_deterministic(model) -> "callable":
    def predict(inputs: torch.Tensor, invariants: torch.Tensor) -> torch.Tensor:
        return model(inputs, invariants).unsqueeze(0)

    return predict


def evaluate_baseline(
    region: str, split: str, checkpoint_path: str | Path, *,
    prepared_root: Path | None = None, device: str = "cpu", **kwargs,
) -> dict:
    device_t = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location=device_t, weights_only=False)
    model = DeterministicResNet(ResNetConfig(**payload["model_config"])).to(device_t)
    model.load_state_dict(payload["model_state"])
    model.eval()
    result = run_evaluation(
        region, split, _wrap_deterministic(model),
        prepared_root=prepared_root, device=device, n_members=1, **kwargs,
    )
    result["model_name"] = "deterministic"
    result["checkpoint"] = str(checkpoint_path)
    return result


def evaluate_gan(
    region: str, split: str, checkpoint_path: str | Path, *,
    prepared_root: Path | None = None, device: str = "cpu",
    n_members: int = DEFAULT_ENSEMBLE_MEMBERS, seed: int = 0, **kwargs,
) -> dict:
    device_t = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location=device_t, weights_only=False)
    generator = ConditionalGenerator(GeneratorConfig(**payload["generator_config"])).to(device_t)
    generator.load_state_dict(payload["generator_state"])
    generator.eval()

    def predict(inputs: torch.Tensor, invariants: torch.Tensor) -> torch.Tensor:
        return generator.sample(inputs, invariants, n_members=n_members, seed=seed)

    result = run_evaluation(
        region, split, predict, prepared_root=prepared_root, device=device, n_members=n_members, **kwargs,
    )
    result["model_name"] = "gan"
    result["checkpoint"] = str(checkpoint_path)
    return result


def evaluate_diffusion(
    region: str, split: str, stage1_checkpoint: str | Path, stage2_checkpoint: str | Path, *,
    prepared_root: Path | None = None, device: str = "cpu",
    n_members: int = DEFAULT_ENSEMBLE_MEMBERS, seed: int = 0,
    sampler_config: HeunSamplerConfig | None = None, **kwargs,
) -> dict:
    device_t = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    sampler_config = sampler_config or HeunSamplerConfig()

    stage1_payload = torch.load(stage1_checkpoint, map_location=device_t, weights_only=False)
    stage1 = RegressionUNet(RegressionStageConfig(**stage1_payload["model_config"])).to(device_t)
    stage1.load_state_dict(stage1_payload["model_state"])
    stage1.eval()

    from src.models.diffusion_residual import SimpleNoiseConditionedUNet

    stage2_payload = torch.load(stage2_checkpoint, map_location=device_t, weights_only=False)
    raw = SimpleNoiseConditionedUNet(RegressionStageConfig(**stage2_payload["denoiser_config"]))
    denoiser = EDMPrecond(raw).to(device_t)
    denoiser.load_state_dict(stage2_payload["denoiser_state"])
    denoiser.eval()

    stats = load_norm_stats(prepared_root)
    moments = target_moments(stats)

    def predict(inputs: torch.Tensor, invariants: torch.Tensor) -> torch.Tensor:
        stage1_mean = stage1(torch.cat([inputs, invariants], dim=1))
        conditioning = torch.cat([inputs, invariants, stage1_mean], dim=1)
        members = []
        for member in range(n_members):
            generator = torch.Generator(device=stage1_mean.device.type).manual_seed(seed + member)
            residual = heun_sample(denoiser, conditioning, stage1_mean.shape, sampler_config, generator=generator)
            members.append(denormalize_to_mm(stage1_mean + residual, moments))
        return torch.stack(members, dim=0)

    result = run_evaluation(
        region, split, predict, prepared_root=prepared_root, device=device, n_members=n_members, **kwargs,
    )
    result["model_name"] = "regression_diffusion"
    result["checkpoint"] = {"stage1": str(stage1_checkpoint), "stage2": str(stage2_checkpoint)}
    return result


def _read_accepted_rows(build_root: Path, region: str, split: str) -> list[dict]:
    index_path = build_root / "metadata" / f"dataset_index_hourly_{region}.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"No build index for region {region!r}: {index_path}")
    with index_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [r for r in rows if r.get("accepted") in ("True", "true", "1") and r.get("split") == split]


def evaluate_bilinear_baseline(
    region: str, split: str, *, build_root: Path | None = None,
    max_spatial_frames: int = MAX_SPATIAL_FRAMES, pixel_km: float = IMERG_PIXEL_KM,
) -> dict:
    """P2 M225 / P1 M199 lower bound, computed directly from stage 2's
    build-stage NPZs (see module docstring for why -- stage 4's clip is
    lossy). No model, no checkpoint, M=1 (CRPS reduces to MAE)."""
    root = _build_root(build_root)
    rows = _read_accepted_rows(root, region, split)
    if not rows:
        raise ValueError(f"No accepted {split!r} rows for region={region!r} in {root}")

    total = len(rows) * HOURS_PER_DAY
    spatial_stride = max(1, total // max(max_spatial_frames, 1))

    pooled_pred: list[np.ndarray] = []
    pooled_obs: list[np.ndarray] = []
    crps_weighted_sum = 0.0
    crps_weight = 0.0
    fss_scores: list[float] = []
    ssim_scores: list[float] = []
    rmse_full_scores: list[float] = []

    global_index = 0
    for row in rows:
        with np.load(row["npz_path"]) as data:
            era5_total = data["inputs"][..., 0] + data["inputs"][..., 1]  # (24, H, W), mm/hr
            target = data["target"][..., 0]
            input_valid = data["input_valid_mask"]
            target_valid = data["target_valid_mask"]
            basin_mask = data["basin_mask"]
        mask = input_valid & target_valid & basin_mask[None, :, :]
        for hour in range(HOURS_PER_DAY):
            frame_pred, frame_obs, frame_mask = era5_total[hour], target[hour], mask[hour]
            pooled_pred.append(frame_pred[frame_mask])
            pooled_obs.append(frame_obs[frame_mask])
            weight = float(frame_mask.sum())
            if weight > 0:
                batch_crps = prob.crps_ensemble(frame_pred[None], frame_obs, frame_mask)
                crps_weighted_sum += batch_crps * weight
                crps_weight += weight
            if global_index % spatial_stride == 0:
                fss_scores.append(prob.mean_fss(frame_pred, frame_obs, pixel_km, frame_mask))
                full = det.full_field_rmse_ssim(frame_pred, frame_obs, frame_mask)
                rmse_full_scores.append(full["rmse"])
                ssim_scores.append(full["ssim"])
            global_index += 1

    pred_flat = np.concatenate(pooled_pred) if pooled_pred else np.array([])
    obs_flat = np.concatenate(pooled_obs) if pooled_obs else np.array([])
    return {
        "region": region, "split": split, "model_name": "bilinear_era5_baseline",
        "sample_count": total,
        "pooled": det.pooled_scores(pred_flat, obs_flat),
        "percentile_mae": prob.percentile_mae(pred_flat, obs_flat),
        "crps": crps_weighted_sum / crps_weight if crps_weight > 0 else float("nan"),
        "spatial_frames_sampled": len(fss_scores),
        "mean_fss": float(np.mean(fss_scores)) if fss_scores else float("nan"),
        "full_field_rmse": float(np.mean(rmse_full_scores)) if rmse_full_scores else float("nan"),
        "full_field_ssim": float(np.mean(ssim_scores)) if ssim_scores else float("nan"),
    }


def to_model_run_result(result: dict, *, eta_enabled: bool = False) -> ModelRunResult:
    """Flattens one evaluation report into the (model, eta_enabled) ->
    metrics shape ``src.evaluation.model_comparison`` aggregates."""
    metrics = {
        "r2": result["pooled"]["r2"], "rmse": result["pooled"]["rmse"],
        "mae": result["pooled"]["mae"], "bias": result["pooled"]["bias"],
        "crps": result["crps"], "fss": result["mean_fss"],
        "percentile_mae_95": result["percentile_mae"].get(95.0, float("nan")),
        "percentile_mae_99": result["percentile_mae"].get(99.0, float("nan")),
        "full_field_rmse": result["full_field_rmse"], "full_field_ssim": result["full_field_ssim"],
    }
    return ModelRunResult(model_name=result["model_name"], eta_enabled=eta_enabled, metrics=metrics)


def write_report(output_path: Path, result: dict) -> Path:
    payload = {**result, "evaluated_utc": datetime.now(timezone.utc).isoformat()}
    _atomic_write_json(output_path, payload)
    return output_path


def main() -> None:
    import argparse

    from src.data_preprocessing import regions as region_defs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["baseline", "gan", "diffusion", "bilinear"])
    parser.add_argument("--region", required=True, choices=region_defs.list_regions())
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--checkpoint")
    parser.add_argument("--stage1-checkpoint")
    parser.add_argument("--stage2-checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.model == "baseline":
        result = evaluate_baseline(args.region, args.split, args.checkpoint, device=args.device)
    elif args.model == "gan":
        result = evaluate_gan(args.region, args.split, args.checkpoint, device=args.device)
    elif args.model == "diffusion":
        result = evaluate_diffusion(
            args.region, args.split, args.stage1_checkpoint, args.stage2_checkpoint, device=args.device,
        )
    else:
        result = evaluate_bilinear_baseline(args.region, args.split)

    if args.output:
        write_report(Path(args.output), result)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
