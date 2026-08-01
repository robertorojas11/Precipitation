"""Run the fixed validation-only hyperparameter search."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import subprocess
import sys


GRID = {
    "learning_rate": [1e-4, 3e-4],
    "base_width": [32, 64],
    "event_weight": [2.0, 4.0],
    "dropout": [0.0, 0.1],
}


def _run(target: str, name: str, epochs: int, parameters: dict, seed: int, context_days: int):
    command = [
        sys.executable, "-m", "src.training.train",
        "--target", target, "--epochs", str(epochs), "--run-name", name,
        "--learning-rate", str(parameters["learning_rate"]),
        "--base-width", str(parameters["base_width"]),
        "--event-weight", str(parameters["event_weight"]),
        "--dropout", str(parameters["dropout"]),
        "--seed", str(seed), "--context-days", str(context_days),
    ]
    subprocess.run(command, check=True)
    run_dir = Path("outputs/v2_clean") / target / name
    history = json.loads((run_dir / "history.json").read_text())
    return {"run_dir": str(run_dir), "best_val_r2": max(row["val_r2"] for row in history), **parameters}


def short_search(target: str):
    results = []
    keys = list(GRID)
    for index, values in enumerate(product(*(GRID[key] for key in keys))):
        parameters = dict(zip(keys, values))
        results.append(_run(target, f"search_{index:02d}", 15, parameters, 42, 1))
    results.sort(key=lambda row: row["best_val_r2"], reverse=True)
    output = Path("outputs/v2_clean") / target / "search_results.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    return results


def full_candidates(target: str):
    search_path = Path("outputs/v2_clean") / target / "search_results.json"
    results = json.loads(search_path.read_text())[:2]
    completed = []
    for index, row in enumerate(results):
        parameters = {key: row[key] for key in GRID}
        completed.append(_run(target, f"candidate_{index}", 80, parameters, 42, 1))
    completed.sort(key=lambda row: row["best_val_r2"], reverse=True)
    if completed[0]["best_val_r2"] < 0.40:
        parameters = {key: completed[0][key] for key in GRID}
        completed.append(_run(target, "candidate_context3", 80, parameters, 42, 3))
        completed.sort(key=lambda row: row["best_val_r2"], reverse=True)
    output = Path("outputs/v2_clean") / target / "candidate_results.json"
    output.write_text(json.dumps(completed, indent=2) + "\n")
    return completed


def final_seeds(target: str):
    candidates = json.loads(
        (Path("outputs/v2_clean") / target / "candidate_results.json").read_text()
    )
    winner = candidates[0]
    parameters = {key: winner[key] for key in GRID}
    context_days = 3 if winner["run_dir"].endswith("context3") else 1
    results = []
    for seed in (17, 42, 73):
        results.append(_run(target, f"final_seed{seed}", 80, parameters, seed, context_days))
    output = Path("outputs/v2_clean") / target / "final_runs.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=["chirps", "oya"])
    parser.add_argument("--stage", required=True, choices=["search", "candidates", "final"])
    args = parser.parse_args()
    function = {"search": short_search, "candidates": full_candidates, "final": final_seeds}[args.stage]
    print(json.dumps(function(args.target), indent=2))


if __name__ == "__main__":
    main()
