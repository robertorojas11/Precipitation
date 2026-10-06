r"""Orchestrates the v2-generative hourly ERA5->IMERG pipeline end to end --
validate raw -> build -> validate processed -> compute stats -> prepare ->
validate prepared -- closing the "no orchestrator yet" gap every one of
`docs/stages_v2/{02,03,04,05}-*.md` documents explicitly. Mirrors the
legacy daily pipeline's own orchestrator (`pipeline.py`: resumable,
durably-logged, stage-ordered) but scoped to exactly the stages that exist
for this framework so far.

**Training is deliberately not a stage here.** Stage 6
(`docs/stages_v2/06-train.md`) is a separate, model-specific, explicitly
user-triggered step for each of the three candidates -- this script builds
*a trainable dataset*, the same scope `docs/decisions/0002-implementacion
-v2.md` gives stages 1-5, and stops there.

Each validation stage is a **hard gate**: a failing `accepted: false`
halts the whole run immediately (unless `--continue-on-error` is passed),
the same "validation gates block, they don't just inform" posture
`docs/stages_v2/03-validate.md` and `05-validate-prepared.md` already
describe -- this script just actually enforces it end to end, instead of
leaving that to whoever runs each stage by hand in order.

**Resumability reuses the validation gates themselves**, rather than a
separate completeness heuristic: "build is done" for a region means
`validate_processed(region)["accepted"]` is already `True`; "prepare is
done" means `validate_prepared(region)["accepted"]` is already `True`;
"stats is done" means `norm_stats.json` already exists. Re-running the
real gate as the resume check means there is only ever one definition of
"this stage already succeeded" -- at the cost of the resume check itself
being as thorough (and therefore sometimes as slow, at full historical
scale) as the gate it reuses, since `validate_processed`/`validate_
prepared` open and check every accepted file, not just a cheap marker.
Still far cheaper than actually re-running the stage it is checking.

**Every root is injectable** (`--v2-raw-root`/`--build-root`/`--prepared-
root`/`--results-root`, or the matching `PipelineOptions` fields), the same
convention every stage module here already follows, so tests can redirect
the whole pipeline into a temporary directory instead of touching
`Config.LOCAL_DATA_DIR`.

```bash
python -m src.data_preprocessing.pipeline_hourly
python -m src.data_preprocessing.pipeline_hourly --from-stage stats --dry-run
# A zero-shot region (never contributes to stats) must skip that stage
# explicitly -- --to-stage/--from-stage are contiguous ranges and cannot
# express "every stage except stats", so name them all:
python -m src.data_preprocessing.pipeline_hourly --regions puerto_rico_caribe \
    --stages validate_raw build validate_processed prepare validate_prepared
```

**Pooling stats over the wrong regions is a real trap, not a hypothetical
one.** `--regions` doubles as "which regions does `stats` pool" (see
`_run_stats`) *and* "which regions does `build`/`prepare` run for" -- the
right choice for the two training regions, where both meanings coincide,
but wrong for a zero-shot region, which must *never* contribute to stats
(`docs/stages_v2/04-prepare.md`). Selecting `stats` while `--regions`
includes a zero-shot region silently computes contaminated statistics;
this module only warns about it (see `run_pipeline`), it does not block it,
since a future experiment might deliberately want different training
regions. Read the warning.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from src.data_preprocessing import build_hourly_dataset as build_mod
from src.data_preprocessing import prepare_hourly_dataset as prepare_mod
from src.data_preprocessing import regions as region_defs
from src.data_preprocessing import validate_hourly_dataset as validate_mod
from src.utils.logging import configure_logging

STAGE_ORDER = ("validate_raw", "build", "validate_processed", "stats", "prepare", "validate_prepared")
GLOBAL_STAGES = frozenset({"stats"})  # run once, not once per region
_GATE_STAGES = frozenset({"validate_raw", "validate_processed", "validate_prepared"})
_ARTIFACT_FILENAME = {"validate_raw": "raw.json", "validate_processed": "processed.json", "validate_prepared": "prepared.json"}


@dataclass(frozen=True)
class PipelineOptions:
    regions: tuple[str, ...]
    stages: tuple[str, ...]
    resume: bool = True
    dry_run: bool = False
    continue_on_error: bool = False
    full_hash_check: bool = False
    v2_raw_root: Path | None = None
    build_root: Path | None = None
    prepared_root: Path | None = None
    results_root: Path | None = None


def _results_root(options: PipelineOptions) -> Path:
    return Path(options.results_root) if options.results_root is not None else Path("outputs") / "v2_generative"


def _validation_dir(region: str, options: PipelineOptions) -> Path:
    return _results_root(options) / region / "validation"


def _write_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, default=str) + "\n")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _norm_stats_path(options: PipelineOptions) -> Path:
    root = Path(options.prepared_root) if options.prepared_root is not None else prepare_mod._prepared_root()
    return root / "metadata" / "norm_stats.json"


def _already_done(stage: str, region: str | None, options: PipelineOptions) -> bool:
    """The resume check: re-queries the real gate/artifact for ``stage``,
    never a separate heuristic (see module docstring)."""
    if not options.resume:
        return False
    if stage == "build":
        return bool(validate_mod.validate_processed(region, root=options.build_root)["accepted"])
    if stage == "stats":
        return _norm_stats_path(options).exists()
    if stage == "prepare":
        return bool(
            validate_mod.validate_prepared(region, prepared_root=options.prepared_root, build_root=options.build_root)[
                "accepted"
            ]
        )
    return False  # validate_raw/validate_processed/validate_prepared always re-run: they're the gates themselves


def _run_validate_raw(region: str, options: PipelineOptions) -> dict:
    return validate_mod.validate_raw(region, v2_raw_root=options.v2_raw_root, full_hash_check=options.full_hash_check)


def _run_build(region: str, options: PipelineOptions) -> dict:
    output_root = Path(options.build_root) / "processed" if options.build_root is not None else None
    manifest = build_mod.build_region(region, v2_raw_root=options.v2_raw_root, output_root=output_root)
    return {"accepted": True, "manifest_sha256": manifest.get("manifest_sha256")}


def _run_validate_processed(region: str, options: PipelineOptions) -> dict:
    return validate_mod.validate_processed(region, root=options.build_root)


def _run_stats(options: PipelineOptions) -> dict:
    # `options.regions` *is* the training-region set pooled into stats --
    # not `prepare_hourly_dataset.TRAINING_REGIONS`'s own default, which
    # would silently ignore `--regions` (and break pooling stats over a
    # single region, e.g. in a test). Real usage with the default
    # `--regions mexico_pacifico mexico_atlantico_golfo` pools exactly the
    # same two regions as before, so production behavior is unchanged.
    return prepare_mod.compute_stats(options.regions, build_root=options.build_root, output_root=options.prepared_root)


def _run_prepare(region: str, options: PipelineOptions) -> dict:
    stats = json.loads(_norm_stats_path(options).read_text())
    manifest = prepare_mod.prepare_region(region, stats, build_root=options.build_root, output_root=options.prepared_root)
    return {"accepted": True, "manifest_sha256": manifest.get("manifest_sha256")}


def _run_validate_prepared(region: str, options: PipelineOptions) -> dict:
    return validate_mod.validate_prepared(region, prepared_root=options.prepared_root, build_root=options.build_root)


_RUNNERS = {
    "validate_raw": _run_validate_raw,
    "build": _run_build,
    "validate_processed": _run_validate_processed,
    "prepare": _run_prepare,
    "validate_prepared": _run_validate_prepared,
}


def _run_one_stage(stage: str, region: str | None, options: PipelineOptions, logger, events_path: Path) -> int:
    label = region or "<all training regions>"
    event = {"timestamp": datetime.now(timezone.utc).isoformat(), "stage": stage, "region": region, "status": "started"}
    _write_event(events_path, event)

    if _already_done(stage, region, options):
        logger.info("stage=%s region=%s status=skipped reason=already_done", stage, label)
        _write_event(events_path, {**event, "status": "skipped"})
        return 0

    if options.dry_run:
        logger.info("stage=%s region=%s status=dry_run", stage, label)
        _write_event(events_path, {**event, "status": "dry_run"})
        return 0

    logger.info("stage=%s region=%s status=running", stage, label)
    started = time.monotonic()
    try:
        result = _run_stats(options) if stage == "stats" else _RUNNERS[stage](region, options)
    except Exception:
        logger.exception("stage=%s region=%s status=failed", stage, label)
        _write_event(events_path, {**event, "status": "failed", "error": "exception"})
        return 1
    duration = round(time.monotonic() - started, 3)

    if stage in _GATE_STAGES and region is not None:
        _write_json(_validation_dir(region, options) / _ARTIFACT_FILENAME[stage], result)

    accepted = bool(result.get("accepted", True))
    status = "completed" if accepted else "failed"
    logger.info("stage=%s region=%s status=%s accepted=%s duration_seconds=%.1f", stage, label, status, accepted, duration)
    _write_event(events_path, {**event, "status": status, "accepted": accepted, "duration_seconds": duration})
    if not accepted:
        logger.error("stage=%s region=%s rejected -- errors=%s", stage, region, result.get("errors"))
        return 1
    return 0


def run_pipeline(options: PipelineOptions, run_directory: Path) -> int:
    logger = configure_logging(name="precipitation.pipeline_hourly")
    events_path = run_directory / "events.jsonl"
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "options.json").write_text(json.dumps(asdict(options), indent=2, default=str) + "\n")

    if "stats" in options.stages:
        logger.warning(
            "stage=stats will pool normalization statistics over exactly regions=%s -- "
            "this must never include a zero-shot evaluation region (docs/stages_v2/04-prepare.md)",
            options.regions,
        )

    failures = 0
    for stage in options.stages:
        if stage in GLOBAL_STAGES:
            code = _run_one_stage(stage, None, options, logger, events_path)
        else:
            code = 0
            for region in options.regions:
                code = _run_one_stage(stage, region, options, logger, events_path)
                if code:
                    break
        if code:
            failures += 1
            if not options.continue_on_error:
                logger.error("Halting pipeline: stage=%s failed and --continue-on-error was not set", stage)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regions", nargs="+", choices=region_defs.list_regions(), default=list(prepare_mod.TRAINING_REGIONS),
    )
    parser.add_argument("--stages", nargs="+", choices=[*STAGE_ORDER, "all"])
    parser.add_argument("--from-stage", choices=STAGE_ORDER)
    parser.add_argument("--to-stage", choices=STAGE_ORDER)
    parser.add_argument("--full-hash-check", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--v2-raw-root", type=Path)
    parser.add_argument("--build-root", type=Path)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.stages and (args.from_stage or args.to_stage):
        parser.error("--stages cannot be combined with --from-stage/--to-stage")
    try:
        stages = _selected_stages(args)
    except ValueError as error:
        parser.error(str(error))

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_directory = Path("outputs") / "logs" / "pipeline_hourly" / run_id
    options = PipelineOptions(
        regions=tuple(args.regions), stages=stages, resume=args.resume, dry_run=args.dry_run,
        continue_on_error=args.continue_on_error, full_hash_check=args.full_hash_check,
        v2_raw_root=args.v2_raw_root, build_root=args.build_root, prepared_root=args.prepared_root,
        results_root=args.results_root,
    )
    raise SystemExit(run_pipeline(options, run_directory))


if __name__ == "__main__":
    main()
