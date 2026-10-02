r"""Stage 8 -- interpretability and calibrated products: wires a stage 6
checkpoint into Shapley channel attribution (PASO 6) and per-pixel
calibrated distribution fitting (PASO 7) of
docs/decisions/0002-implementacion-v2.md. Full narrative:
docs/stages_v2/08-interpretability.md.

Like stage 7, this module adds **no new method**: every formula already
exists in `src/evaluation/{shapley_attribution,pixel_distributions}.py`,
built generically ahead of any trainable dataset. What's new here is the
wiring -- turning a checkpoint plus a prepared split into the specific
inputs those two modules expect (a scalar-valued ``predict_fn`` over input
channels for Shapley; a pooled per-pixel ensemble-sample array for the
distribution fit) -- and it reuses stage 7's own checkpoint loaders
(`evaluate_hourly.load_{baseline,gan,diffusion}_predict_fn`) rather than
re-deriving how to load a model from a checkpoint a second time.

```bash
python -m src.evaluation.interpretability_hourly --task shapley \
    --model gan --region mexico_pacifico --split test --checkpoint .../best.pt
python -m src.evaluation.interpretability_hourly --task pixel-distribution \
    --model gan --region mexico_pacifico --split test --checkpoint .../best.pt \
    --pixel-row 50 --pixel-col 80
```
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.evaluation.evaluate_hourly import (
    DEFAULT_ENSEMBLE_MEMBERS,
    load_baseline_predict_fn,
    load_diffusion_predict_fn,
    load_gan_predict_fn,
)
from src.evaluation.pixel_distributions import ALL_FAMILIES, fit_pixel_distribution, quantile_map
from src.evaluation.shapley_attribution import channel_attribution
from src.training.common import to_device
from src.training.dataset import PreparedHourlyDataset
from src.training.eta_learning import spatial_maximum_observable
from src.utils.logging import configure_logging

logger = configure_logging(name="precipitation.interpretability_hourly")

CHANNEL_NAMES = ("era5_cp", "era5_lsp", "land_mask", "elevation")
DEFAULT_SHAPLEY_SAMPLES = 5
DEFAULT_QUANTILES = (0.90, 0.95, 0.99)
DEFAULT_EXCEEDANCE_THRESHOLD_MM = 25.0  # spateGAN-ERA5's severe-rain reference, OQ-5


def _predict_fn_for_model(model_name: str, device: str, **kwargs):
    if model_name == "baseline":
        return load_baseline_predict_fn(kwargs["checkpoint"], device=device)
    if model_name == "gan":
        return load_gan_predict_fn(
            kwargs["checkpoint"], device=device,
            n_members=kwargs.get("n_members", DEFAULT_ENSEMBLE_MEMBERS), seed=kwargs.get("seed", 0),
        )
    if model_name == "diffusion":
        return load_diffusion_predict_fn(
            kwargs["stage1_checkpoint"], kwargs["stage2_checkpoint"], prepared_root=kwargs.get("prepared_root"),
            device=device, n_members=kwargs.get("n_members", DEFAULT_ENSEMBLE_MEMBERS), seed=kwargs.get("seed", 0),
            sampler_config=kwargs.get("sampler_config"),
        )
    raise ValueError(f"Unknown model_name {model_name!r}; choose 'baseline', 'gan', or 'diffusion'")


def _sample_indices(total: int, n_samples: int) -> list[int]:
    if total <= 0:
        return []
    n_samples = min(n_samples, total)
    return sorted({int(i) for i in np.linspace(0, total - 1, n_samples)})


def shapley_attribution_for_sample(
    predict_fn, inputs: torch.Tensor, invariants: torch.Tensor,
) -> dict:
    """Shapley attribution of the **predicted peak precipitation** (the
    observable `src.training.eta_learning.spatial_maximum_observable` also
    uses, for consistency across this project's interpretability and
    eta-learning machinery) over the model's 4 input channels, for one
    ``(inputs, invariants)`` sample (each already batch-of-1, ``(1, c, H, W)``).

    Baseline: an all-zero field per channel -- "uninformative" in the exact
    representation these channels are normalized into (z-score space, where
    0.0 is each channel's own training-distribution mean), the same
    convention `src.training.dataset.PreparedHourlyDataset` already uses to
    sanitize invalid pixels for the forward pass. 4 channels is comfortably
    inside `shapley_attribution.DEFAULT_MAX_EXACT_CHANNELS` (14), so this is
    always the exact enumeration (16 subsets), never the sampling fallback.
    """
    channels = [inputs[:, 0:1], inputs[:, 1:2], invariants[:, 0:1], invariants[:, 1:2]]
    sample = [c.clone() for c in channels]
    baseline = [torch.zeros_like(c) for c in channels]

    def predict(composed) -> float:
        composed_inputs = torch.cat([composed[0], composed[1]], dim=1)
        composed_invariants = torch.cat([composed[2], composed[3]], dim=1)
        with torch.no_grad():
            ensemble = predict_fn(composed_inputs, composed_invariants)
        mean_prediction = ensemble.mean(dim=0)
        return float(spatial_maximum_observable(mean_prediction).reshape(-1)[0].item())

    return channel_attribution(predict, sample, baseline, CHANNEL_NAMES)


def compute_shapley_attribution(
    region: str, split: str, model_name: str, *,
    prepared_root: Path | None = None, device: str = "cpu",
    n_samples: int = DEFAULT_SHAPLEY_SAMPLES, **model_kwargs,
) -> dict:
    """Runs `shapley_attribution_for_sample` over an evenly-spaced sample of
    at most ``n_samples`` (day, hour) instances from the split (exact
    Shapley enumeration is cheap per sample -- 16 forward passes -- but
    still real cost multiplied by however many samples are requested;
    docs/stages_v2/08-interpretability.md documents why this stays small by
    default rather than covering the whole split).
    """
    predict_fn = _predict_fn_for_model(model_name, device, prepared_root=prepared_root, **model_kwargs)
    dataset = PreparedHourlyDataset(region, split, prepared_root=prepared_root)
    device_t = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    per_sample = []
    for index in _sample_indices(len(dataset), n_samples):
        item = to_device(
            {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in dataset[index].items()}, device_t,
        )
        attribution = shapley_attribution_for_sample(predict_fn, item["inputs"], item["invariants"])
        attribution["date"] = item["date"]
        per_sample.append(attribution)

    mean_values = (
        np.mean([s["shapley_values"] for s in per_sample], axis=0).tolist() if per_sample else []
    )
    return {
        "region": region, "split": split, "model_name": model_name,
        "channel_names": list(CHANNEL_NAMES), "samples_evaluated": len(per_sample),
        "per_sample": per_sample, "mean_shapley_values": mean_values,
    }


def pooled_ensemble_samples_at_pixel(
    predict_fn, dataset: PreparedHourlyDataset, row: int, col: int, *,
    device: torch.device, max_frames: int,
) -> np.ndarray:
    """Pools every ensemble member's prediction at one fixed pixel across an
    evenly-spaced sample of frames -- "for each pixel, all ensemble-
    generative samples ... are pooled into a value set"
    (`pixel_distributions.py`'s own docstring). ``max_frames`` keeps this
    tractable: `fit_pixel_distribution` bootstraps every candidate family
    (200 refits each by default), so the cost is per call, not per pixel --
    the frame cap controls how many *samples* feed one such call, not how
    expensive fitting the resulting pool is.
    """
    values: list[float] = []
    with torch.no_grad():
        for index in _sample_indices(len(dataset), max_frames):
            item = dataset[index]
            inputs = item["inputs"].unsqueeze(0).to(device)
            invariants = item["invariants"].unsqueeze(0).to(device)
            ensemble = predict_fn(inputs, invariants)  # (M, 1, 1, H, W)
            pixel_values = ensemble[:, 0, 0, row, col].cpu().numpy()
            values.extend(float(v) for v in pixel_values)
    return np.asarray(values, dtype=np.float64)


def fit_pixel_distributions(
    region: str, split: str, model_name: str, *,
    pixels: list[tuple[int, int]],
    prepared_root: Path | None = None, device: str = "cpu",
    max_frames: int = 20, n_bootstrap: int = 100, quantiles=DEFAULT_QUANTILES,
    exceedance_threshold_mm: float = DEFAULT_EXCEEDANCE_THRESHOLD_MM,
    families=ALL_FAMILIES, random_state: int = 0, **model_kwargs,
) -> dict:
    """Fits the zero-inflated per-pixel model (PASO 7) at each of ``pixels``
    (``(row, col)`` pairs -- an explicit list, not "every pixel": at
    `n_bootstrap=100` refits per family x 5 families, one pixel's fit is
    already real compute; a full HxW grid is a deliberately separate,
    much more expensive, user-triggered run, see docs/stages_v2/
    08-interpretability.md's "Known limitations").
    """
    predict_fn = _predict_fn_for_model(model_name, device, prepared_root=prepared_root, **model_kwargs)
    dataset = PreparedHourlyDataset(region, split, prepared_root=prepared_root)
    device_t = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    per_pixel = []
    for row, col in pixels:
        pixel_samples_mm = pooled_ensemble_samples_at_pixel(
            predict_fn, dataset, row, col, device=device_t, max_frames=max_frames,
        )
        try:
            fit = fit_pixel_distribution(
                pixel_samples_mm, families=families, n_bootstrap=n_bootstrap, random_state=random_state,
            )
        except ValueError as error:
            per_pixel.append(
                {"row": row, "col": col, "error": str(error), "sample_count": int(pixel_samples_mm.size)}
            )
            continue
        per_pixel.append({
            "row": row, "col": col, "sample_count": int(pixel_samples_mm.size),
            "p_zero": fit.p_zero, "family_name": fit.family_name,
            "goodness_of_fit_passed": fit.goodness_of_fit_passed,
            "quantile_map_mm": quantile_map(fit, quantiles),
            "exceedance_probability": fit.exceedance_probability(exceedance_threshold_mm),
            "exceedance_threshold_mm": exceedance_threshold_mm,
        })
    return {
        "region": region, "split": split, "model_name": model_name,
        "max_frames_pooled": max_frames, "n_bootstrap": n_bootstrap, "families": list(families),
        "pixels": per_pixel,
    }


def main() -> None:
    import argparse

    from src.data_preprocessing import regions as region_defs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["shapley", "pixel-distribution"])
    parser.add_argument("--model", required=True, choices=["baseline", "gan", "diffusion"])
    parser.add_argument("--region", required=True, choices=region_defs.list_regions())
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--checkpoint")
    parser.add_argument("--stage1-checkpoint")
    parser.add_argument("--stage2-checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pixel-row", type=int, action="append", default=[])
    parser.add_argument("--pixel-col", type=int, action="append", default=[])
    args = parser.parse_args()

    model_kwargs = (
        {"checkpoint": args.checkpoint} if args.model != "diffusion"
        else {"stage1_checkpoint": args.stage1_checkpoint, "stage2_checkpoint": args.stage2_checkpoint}
    )

    if args.task == "shapley":
        result = compute_shapley_attribution(args.region, args.split, args.model, device=args.device, **model_kwargs)
    else:
        pixels = list(zip(args.pixel_row, args.pixel_col))
        if not pixels:
            raise SystemExit("--pixel-row/--pixel-col (at least one pair) are required for --task pixel-distribution")
        result = fit_pixel_distributions(
            args.region, args.split, args.model, pixels=pixels, device=args.device, **model_kwargs,
        )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
