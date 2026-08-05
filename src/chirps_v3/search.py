"""Generate and execute the bounded 120-run, multi-phase search."""

from __future__ import annotations
import argparse
from itertools import product
import json
from pathlib import Path
import subprocess
import sys

from src.chirps_v3.config import FOLDS, ROOT
from src.utils.config import Config

logger = Config.get_logger()


PHASES = {
    "representation": [dict(context_days=c, fusion=f, width=48, dropout=.1, learning_rate=2e-4)
                       for c, f in product((1, 3, 5, 7), ("mean", "attention"))],
    "architecture": [dict(context_days=5, fusion="attention", width=w, dropout=d, learning_rate=2e-4)
                     for w, d in product((32, 48, 64, 80), (0., .1, .2))],
    "optimization": [dict(context_days=5, fusion="attention", width=64, dropout=.1, learning_rate=lr,
                           weight_decay=wd) for lr, wd in product((1e-4, 2e-4, 4e-4), (1e-5, 1e-4, 1e-3))],
}


def manifest(budget: int = 120) -> list[dict]:
    jobs = []
    # Screening uses two distant folds; rolling confirmation uses all five.
    screening_folds = (FOLDS[0], FOLDS[-1])
    for phase, configs in PHASES.items():
        for index, config in enumerate(configs):
            for fold in screening_folds:
                jobs.append({"phase": phase, "fold": fold.name,
                             "run_name": f"{phase}_{index:02d}", **config})
    # Reserve remaining budget for all-fold confirmation of a strong default.
    seed = 0
    while len(jobs) < budget:
        fold = FOLDS[seed % len(FOLDS)]
        jobs.append({"phase": "rolling", "fold": fold.name, "run_name": f"rolling_{seed:03d}",
                     "context_days": 5, "fusion": "attention", "width": 64,
                     "dropout": .1, "learning_rate": 2e-4, "seed": (17, 42, 73)[seed % 3]})
        seed += 1
    return jobs[:budget]


def run(budget, phase, execute, runtime):
    jobs = [job for job in manifest(budget) if phase == "all" or job["phase"] == phase]
    path = ROOT / "search_manifest.json"; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jobs, indent=2) + "\n")
    if execute:
        for job_number, job in enumerate(jobs, 1):
            result_path = ROOT / "runs" / job["fold"] / job["run_name"] / "result.json"
            if result_path.exists():
                logger.info("search job=%d/%d fold=%s run=%s status=skipped result_exists=true",
                            job_number, len(jobs), job["fold"], job["run_name"])
                continue
            command = [sys.executable, "-m", "src.chirps_v3.training"]
            for key, value in {**job, **runtime}.items():
                if key != "phase": command.extend((f"--{key.replace('_', '-')}", str(value)))
            logger.info("search job=%d/%d fold=%s run=%s status=running",
                        job_number, len(jobs), job["fold"], job["run_name"])
            subprocess.run(command, check=True)
            logger.info("search job=%d/%d fold=%s run=%s status=completed",
                        job_number, len(jobs), job["fold"], job["run_name"])
    return {"manifest": str(path), "jobs": len(jobs), "executed": execute}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--budget", type=int, default=120); parser.add_argument("--phase", choices=(*PHASES, "rolling", "all"), default="all")
    parser.add_argument("--execute", action="store_true"); parser.add_argument("--epochs", type=int, default=40); parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=2); parser.add_argument("--device", default="cuda"); parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args(); runtime = {key: getattr(args, key) for key in ("epochs", "batch_size", "accumulation_steps", "device", "num_workers")}
    print(json.dumps(run(args.budget, args.phase, args.execute, runtime), indent=2))


if __name__ == "__main__": main()
