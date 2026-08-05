"""Freeze the best robust configuration and guard the untouched 2026 holdout."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from src.chirps_v3.config import FOLDS, ROOT


def freeze(results_root: Path = ROOT / "runs", minimum_r2: float = .8) -> dict:
    grouped = {}
    for path in results_root.glob("*/*/result.json"):
        result = json.loads(path.read_text()); config = result["config"]
        signature = tuple((key, config[key]) for key in ("context_days", "width", "dropout", "fusion", "learning_rate", "weight_decay"))
        grouped.setdefault(signature, []).append(result)
    eligible = []
    for signature, results in grouped.items():
        folds = {r["config"]["fold"] for r in results}; values = [r["best_r2"] for r in results]
        if folds == {f.name for f in FOLDS}:
            eligible.append((float(np.mean(values) - .25 * np.std(values)), signature, results))
    if not eligible: raise RuntimeError("No configuration has completed all five rolling folds")
    score, signature, results = max(eligible, key=lambda item: item[0]); mean_r2 = float(np.mean([r["best_r2"] for r in results]))
    payload = {"frozen": True, "selection_score": score, "mean_fold_r2": mean_r2,
               "meets_r2_goal": bool(mean_r2 >= minimum_r2), "configuration": dict(signature),
               "fold_results": results, "holdout_2026_status": "locked_until_365_complete_days"}
    output = ROOT / "frozen_experiment.json"; output.write_text(json.dumps(payload, indent=2) + "\n"); return payload


def verify_holdout(index_path: Path) -> dict:
    import pandas as pd
    frame = pd.read_csv(index_path); frame["date"] = pd.to_datetime(frame["date"])
    valid = frame[(frame.date.dt.year == 2026) & (frame.get("accepted", True) == True)]
    unique = valid.date.dt.date.nunique()
    if unique < 365: raise RuntimeError(f"2026 holdout remains locked: found {unique}/365 complete accepted dates")
    return {"accepted_days": unique, "unlocked": True}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--results-root", type=Path, default=ROOT / "runs"); p.add_argument("--verify-2026-index", type=Path)
    a = p.parse_args(); result = verify_holdout(a.verify_2026_index) if a.verify_2026_index else freeze(a.results_root)
    print(json.dumps(result, indent=2))
if __name__ == "__main__": main()
