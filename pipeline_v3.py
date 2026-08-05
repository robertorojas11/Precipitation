"""Resumable, stage-selectable CHIRPS v3 pipeline with durable child logs."""

from __future__ import annotations
import argparse, json, logging, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

from src.chirps_v3.config import ROOT, write_contract
from src.utils.logging import configure_logging

STAGES = ("storage_check", "contract", "prepare_folds", "feasibility", "search_manifest", "train_search", "freeze", "verify_2026")
DEFAULT_STAGES = STAGES[:-1]
DEPENDENCIES = {
    "prepare_folds": {"storage_check", "contract"},
    "feasibility": {"storage_check", "contract"},
    "train_search": {"storage_check", "contract", "prepare_folds", "feasibility", "search_manifest"},
    "freeze": {"contract"},
    "verify_2026": {"storage_check", "contract"},
}


def selected_stages(requested):
    """Resolve transitive prerequisites while preserving canonical stage order."""
    wanted = set(DEFAULT_STAGES if requested == ["all"] else requested)
    changed = True
    while changed:
        before = len(wanted)
        for stage in tuple(wanted):
            wanted.update(DEPENDENCIES.get(stage, ()))
        changed = len(wanted) != before
    return [stage for stage in STAGES if stage in wanted]


def command(stage, args):
    py = sys.executable
    if stage == "storage_check":
        return [py, "-m", "src.utils.storage", "--minimum-free-gib", str(args.minimum_free_gib), "--output", str(ROOT / "validation" / "storage.json")]
    if stage == "feasibility": return [py, "-m", "src.chirps_v3.feasibility"]
    if stage == "prepare_folds": return [py, "-m", "src.chirps_v3.prepare"]
    if stage == "search_manifest": return [py, "-m", "src.chirps_v3.search", "--budget", str(args.budget)]
    if stage == "train_search":
        return [py, "-m", "src.chirps_v3.search", "--budget", str(args.budget), "--phase", args.phase, "--execute",
                "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--accumulation-steps", str(args.accumulation_steps),
                "--device", args.device, "--num-workers", str(args.num_workers)]
    if stage == "freeze": return [py, "-m", "src.chirps_v3.freeze"]
    if stage == "verify_2026":
        if not args.holdout_index: raise ValueError("--holdout-index is required for verify_2026")
        return [py, "-m", "src.chirps_v3.freeze", "--verify-2026-index", str(args.holdout_index)]
    return None


def complete(stage):
    artifacts = {"storage_check": ROOT / "validation" / "storage.json", "contract": ROOT / "experiment_contract.json",
                 "prepare_folds": ROOT / "fold_stats" / "fold_2024_2025.json",
                 "feasibility": ROOT / "feasibility.json", "search_manifest": ROOT / "search_manifest.json",
                 "freeze": ROOT / "frozen_experiment.json"}
    if stage == "prepare_folds":
        return all((ROOT / "fold_stats" / f"fold_{start}_{end}.json").exists()
                   for start, end in ((2016, 2017), (2018, 2019), (2020, 2021), (2022, 2023), (2024, 2025)))
    return artifacts.get(stage, Path("/__never__")).exists()


def run(args):
    ROOT.mkdir(parents=True, exist_ok=True); log_dir = ROOT / "logs"; log_dir.mkdir(exist_ok=True)
    logger = configure_logging(name="precipitation.v3", log_file=log_dir / f"pipeline_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.log")
    selected = selected_stages(args.stages)
    logger.info("requested_stages=%s resolved_stages=%s", args.stages, selected)
    state_path = ROOT / "pipeline_state.json"; state = json.loads(state_path.read_text()) if state_path.exists() else {}
    for stage in selected:
        # Network storage is probed on every invocation that resolves it; an old
        # JSON report is not evidence that a network mount is healthy now.
        if stage != "storage_check" and args.resume and complete(stage):
            logger.info("stage=%s status=skipped artifact_complete=true", stage); continue
        logger.info("stage=%s status=running", stage); started = time.monotonic()
        try:
            if stage == "contract": write_contract()
            else:
                cmd = command(stage, args); logger.info("command=%s", " ".join(cmd))
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in process.stdout: logger.info("child | %s", line.rstrip())
                if process.wait(): raise subprocess.CalledProcessError(process.returncode, cmd)
            state[stage] = {"status": "completed", "finished_at": datetime.now(timezone.utc).isoformat(), "seconds": round(time.monotonic() - started, 2)}
        except Exception as error:
            state[stage] = {"status": "failed", "error": str(error), "finished_at": datetime.now(timezone.utc).isoformat()}
            state_path.write_text(json.dumps(state, indent=2) + "\n"); logger.exception("stage=%s status=failed", stage); raise
        state_path.write_text(json.dumps(state, indent=2) + "\n"); logger.info("stage=%s status=completed", stage)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--stages", nargs="+", default=["all"], choices=(*STAGES, "all")); p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--budget", type=int, default=120); p.add_argument("--phase", choices=("representation", "architecture", "optimization", "rolling", "all"), default="all")
    p.add_argument("--epochs", type=int, default=40); p.add_argument("--batch-size", type=int, default=2); p.add_argument("--accumulation-steps", type=int, default=2)
    p.add_argument("--device", default="cuda"); p.add_argument("--num-workers", type=int, default=4); p.add_argument("--minimum-free-gib", type=float, default=100); p.add_argument("--holdout-index", type=Path)
    run(p.parse_args())
if __name__ == "__main__": main()
