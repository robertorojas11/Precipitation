"""Run the precipitation workflow by target and stage with durable logs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from src.data_preprocessing.quality import DATASET_VERSION
from src.utils.config import Config
from src.utils.logging import configure_logging


STAGE_ORDER = (
    "storage_check",
    "acquire",
    "validate_raw",
    "build",
    "validate_processed",
    "prepare",
    "validate_prepared",
    "search",
    "candidates",
    "train_final",
    "evaluate_validation",
    "evaluate_test",
    "report",
)


@dataclass(frozen=True)
class PipelineOptions:
    targets: tuple[str, ...]
    stages: tuple[str, ...]
    start_year: int
    end_year: int
    device: str
    batch_size: int
    num_workers: int
    resume: bool
    dry_run: bool
    continue_on_error: bool
    minimum_free_gib: float


def _final_run_directories(target: str) -> list[Path]:
    root = Path("outputs") / DATASET_VERSION / target
    manifest = root / "final_runs.json"
    if manifest.exists():
        return [Path(record["run_dir"]) for record in json.loads(manifest.read_text())]
    return [root / f"final_seed{seed}" for seed in (17, 42, 73)]


def _commands(target: str, stage: str, options: PipelineOptions) -> list[list[str]]:
    python = sys.executable
    module = lambda name, *arguments: [python, "-m", name, *map(str, arguments)]
    validation_dir = Path("outputs") / DATASET_VERSION / target / "validation"
    final_runs = _final_run_directories(target)

    if stage == "storage_check":
        return [module(
            "src.utils.storage",
            "--minimum-free-gib", options.minimum_free_gib,
            "--output", validation_dir / "storage.json",
        )]
    if stage == "acquire":
        if target == "chirps":
            return []
        return [module(
            "src.data_extraction.export_oya",
            "--start-year", options.start_year,
            "--end-year", options.end_year,
        )]
    if stage == "validate_raw":
        return [module(
            "src.data_preprocessing.validate_dataset",
            "--target", target, "--stage", "raw",
            "--output", validation_dir / "raw.json",
        )]
    if stage == "build":
        return [module("src.data_preprocessing.build_dataset", "--target", target)]
    if stage == "validate_processed":
        return [module(
            "src.data_preprocessing.validate_dataset",
            "--target", target, "--stage", "processed",
            "--output", validation_dir / "processed.json",
        )]
    if stage == "prepare":
        return [module(
            "src.data_preprocessing.prepare_dataset",
            "--target", target, "--stage", "all",
        )]
    if stage == "validate_prepared":
        return [module(
            "src.data_preprocessing.validate_dataset",
            "--target", target, "--stage", "fast",
            "--output", validation_dir / "prepared.json",
        )]
    if stage in {"search", "candidates"}:
        return [module(
            "src.training.search", "--target", target, "--stage", stage,
            "--batch-size", options.batch_size,
            "--num-workers", options.num_workers,
            "--device", options.device,
        )]
    if stage == "train_final":
        return [module(
            "src.training.search", "--target", target, "--stage", "final",
            "--batch-size", options.batch_size,
            "--num-workers", options.num_workers,
            "--device", options.device,
        )]
    if stage in {"evaluate_validation", "evaluate_test"}:
        split = "val" if stage == "evaluate_validation" else "test"
        return [module(
            "src.training.evaluate",
            "--run-dir", *final_runs,
            "--split", split,
            "--batch-size", options.batch_size,
            "--device", options.device,
            "--num-workers", options.num_workers,
        )]
    if stage == "report":
        return [module(
            "src.training.report",
            "--run-dir", *final_runs,
            "--split", "test",
            "--batch-size", options.batch_size,
            "--device", options.device,
            "--num-workers", options.num_workers,
        )]
    raise ValueError(f"Unknown stage: {stage}")


def _artifact(target: str, stage: str) -> Path | None:
    local = Path(Config.LOCAL_DATA_DIR) / DATASET_VERSION
    output = Path("outputs") / DATASET_VERSION / target
    final_runs = _final_run_directories(target)
    first_final_run = final_runs[0]
    artifacts = {
        "validate_raw": output / "validation" / "raw.json",
        "build": local / "metadata" / f"manifest_{target}.json",
        "validate_processed": output / "validation" / "processed.json",
        "prepare": local / "metadata" / f"fast_manifest_{target}.json",
        "validate_prepared": output / "validation" / "prepared.json",
        "search": output / "search_results.json",
        "candidates": output / "candidate_results.json",
        "train_final": output / "final_runs.json",
        "evaluate_validation": first_final_run / "metrics_val.json",
        "evaluate_test": first_final_run / "metrics_test.json",
        "report": output / "final_report" / "report.md",
    }
    return artifacts.get(stage)


def _write_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, default=str) + "\n")


def _run_command(command: list[str], log_path: Path, logger) -> int:
    logger.info("command=%s", shlex.join(command))
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"\nCOMMAND {shlex.join(command)}\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parent,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            message = line.rstrip()
            stream.write(line)
            stream.flush()
            logger.info("child | %s", message)
        return_code = process.wait()
    logger.info(
        "return_code=%d duration_seconds=%.1f",
        return_code,
        time.monotonic() - started,
    )
    return return_code


def run_pipeline(options: PipelineOptions, run_directory: Path) -> int:
    logger = configure_logging(
        name="precipitation.pipeline",
        log_file=run_directory / "pipeline.log",
    )
    events_path = run_directory / "events.jsonl"
    (run_directory / "options.json").write_text(
        json.dumps(asdict(options), indent=2) + "\n"
    )
    failures = 0
    storage_checked = False
    for target in options.targets:
        for stage in options.stages:
            if not storage_checked and stage != "storage_check":
                storage_event = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "target": target,
                    "stage": "storage_check",
                    "status": "started",
                }
                _write_event(events_path, storage_event)
                storage_log = run_directory / target / "storage_check.log"
                storage_log.parent.mkdir(parents=True, exist_ok=True)
                command = _commands(target, "storage_check", options)[0]
                if options.dry_run:
                    logger.info("target=%s stage=storage_check dry_run=%s", target, shlex.join(command))
                    _write_event(events_path, {**storage_event, "status": "dry_run"})
                else:
                    result = _run_command(command, storage_log, logger)
                    status = "completed" if result == 0 else "failed"
                    _write_event(events_path, {
                        **storage_event,
                        "status": status,
                        "return_code": result,
                    })
                    if result:
                        logger.error("Storage preflight failed; no pipeline process was started")
                        return 1
                storage_checked = True
            artifact = _artifact(target, stage)
            log_path = run_directory / target / f"{stage}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "target": target,
                "stage": stage,
                "status": "started",
                "artifact": str(artifact) if artifact else None,
            }
            _write_event(events_path, event)

            if options.resume and artifact is not None and artifact.exists():
                logger.info("target=%s stage=%s status=skipped artifact=%s", target, stage, artifact)
                _write_event(events_path, {**event, "status": "skipped"})
                continue

            commands = _commands(target, stage, options)
            if not commands:
                logger.info("target=%s stage=%s status=not_applicable", target, stage)
                _write_event(events_path, {**event, "status": "not_applicable"})
                continue
            if options.dry_run:
                for command in commands:
                    logger.info("target=%s stage=%s dry_run=%s", target, stage, shlex.join(command))
                _write_event(events_path, {**event, "status": "dry_run"})
                continue

            logger.info("target=%s stage=%s status=running", target, stage)
            started = time.monotonic()
            return_code = 0
            try:
                for command in commands:
                    return_code = _run_command(command, log_path, logger)
                    if return_code:
                        break
            except Exception:
                logger.exception("target=%s stage=%s status=failed", target, stage)
                return_code = 1
            status = "completed" if return_code == 0 else "failed"
            _write_event(events_path, {
                **event,
                "status": status,
                "return_code": return_code,
                "duration_seconds": round(time.monotonic() - started, 3),
            })
            logger.info("target=%s stage=%s status=%s", target, stage, status)
            if stage == "storage_check" and return_code == 0:
                storage_checked = True
            if return_code:
                failures += 1
                if not options.continue_on_error:
                    return 1
    return 1 if failures else 0


def _selected_stages(arguments) -> tuple[str, ...]:
    if arguments.stages:
        return STAGE_ORDER if "all" in arguments.stages else tuple(arguments.stages)
    start = STAGE_ORDER.index(arguments.from_stage) if arguments.from_stage else 0
    end = STAGE_ORDER.index(arguments.to_stage) + 1 if arguments.to_stage else len(STAGE_ORDER)
    if start >= end:
        raise ValueError("--from-stage must precede or equal --to-stage")
    return STAGE_ORDER[start:end]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run selected precipitation pipeline stages with durable logs."
    )
    parser.add_argument("--target", choices=["chirps", "oya", "both"], default="both")
    parser.add_argument("--stages", nargs="+", choices=[*STAGE_ORDER, "all"])
    parser.add_argument("--from-stage", choices=STAGE_ORDER)
    parser.add_argument("--to-stage", choices=STAGE_ORDER)
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--minimum-free-gib", type=float, default=5.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.stages and (args.from_stage or args.to_stage):
        parser.error("--stages cannot be combined with --from-stage/--to-stage")
    if args.start_year > args.end_year:
        parser.error("--start-year cannot exceed --end-year")
    try:
        stages = _selected_stages(args)
    except ValueError as error:
        parser.error(str(error))
    targets = ("chirps", "oya") if args.target == "both" else (args.target,)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_directory = Path("logs") / "pipeline" / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    options = PipelineOptions(
        targets=targets,
        stages=stages,
        start_year=args.start_year,
        end_year=args.end_year,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resume=args.resume,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
        minimum_free_gib=args.minimum_free_gib,
    )
    raise SystemExit(run_pipeline(options, run_directory))


if __name__ == "__main__":
    main()
