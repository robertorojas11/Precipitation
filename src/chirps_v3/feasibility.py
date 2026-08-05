"""Measure information ceilings before spending GPU time."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from src.chirps_v3.config import PREPARED_ROOT, ROOT, write_contract
from src.chirps_v3.dataset import discover_samples, source_stats
from src.chirps_v3.metrics import PooledMetrics


def masked_coarse_oracle(target, mask, factor):
    numerator = F.avg_pool2d(target * mask, factor, factor) * factor * factor
    denominator = F.avg_pool2d(mask, factor, factor) * factor * factor
    coarse = numerator / denominator.clamp_min(1)
    return F.interpolate(coarse, size=target.shape[-2:], mode="bilinear", align_corners=False)


def run(max_samples: int | None = None, output: Path = ROOT / "feasibility.json") -> dict:
    stats = source_stats(); available = discover_samples(PREPARED_ROOT)
    metrics = {"era5": PooledMetrics(), "oracle_10km": PooledMetrics(), "oracle_25km": PooledMetrics()}
    dates = sorted(available)
    if max_samples: dates = dates[::max(1, len(dates) // max_samples)][:max_samples]
    for date in dates:
        with np.load(available[date]) as sample:
            normalized = torch.from_numpy(sample["real_5km"]).float().unsqueeze(0)
            target = torch.expm1(normalized * stats["target_std"] + stats["target_mean"]).clamp_min(0)
            baseline_n = torch.from_numpy(sample["era5_precip_5km_norm"]).float().unsqueeze(0)
            baseline = torch.expm1(baseline_n * stats["target_std"] + stats["target_mean"]).clamp_min(0)
            mask = (torch.from_numpy(sample["target_valid_mask_5km"]) &
                    torch.from_numpy(sample["land_mask_5km"])).unsqueeze(0)
        metrics["era5"].update(baseline.numpy(), target.numpy(), mask.numpy())
        for factor, name in ((2, "oracle_10km"), (5, "oracle_25km")):
            prediction = masked_coarse_oracle(target, mask.float(), factor)
            metrics[name].update(prediction.numpy(), target.numpy(), mask.numpy())
    results = {name: value.result() for name, value in metrics.items()}
    payload = {"samples": len(dates), "metrics": results,
               "r2_0_8_credible_from_25km_information": bool(results["oracle_25km"]["r2"] >= .8),
               "warning": None if results["oracle_25km"]["r2"] >= .8 else
               "The target-derived 25 km oracle is below 0.8; architecture search alone cannot credibly guarantee the goal."}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(payload, indent=2) + "\n")
    write_contract(); return payload


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--max-samples", type=int); parser.add_argument("--output", type=Path, default=ROOT / "feasibility.json")
    args = parser.parse_args()
    print(json.dumps(run(args.max_samples, args.output), indent=2))


if __name__ == "__main__": main()
