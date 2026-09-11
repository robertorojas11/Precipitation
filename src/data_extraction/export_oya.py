"""Export Oya daily precipitation with explicit coverage metadata."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

import ee
from dateutil.relativedelta import relativedelta

from src.data_extraction import drive, earth_engine
from src.utils.config import Config
from src.utils.retries import get_task_status_with_retry, start_task_with_retry


TASK_BATCH_SIZE = 20
logger = Config.get_logger()


def _wait_for_tasks(tasks, poll_interval: int = 60) -> None:
    pending = {task.id for task in tasks}
    logger.info("Waiting for %d Earth Engine tasks", len(pending))
    while pending:
        for task_id in list(pending):
            status = get_task_status_with_retry(task_id)
            if status["state"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                pending.remove(task_id)
                if status["state"] != "COMPLETED":
                    raise RuntimeError(
                        f"Earth Engine task {task_id} ended as {status['state']}: "
                        f"{status.get('error_message', 'no details')}"
                    )
        if pending:
            logger.info("%d Earth Engine tasks remain active", len(pending))
            time.sleep(poll_interval)


def _submit_in_batches(task_factories) -> None:
    for offset in range(0, len(task_factories), TASK_BATCH_SIZE):
        tasks = [
            task
            for factory in task_factories[offset : offset + TASK_BATCH_SIZE]
            for task in factory()
        ]
        _wait_for_tasks(tasks)


def _submit(date: datetime, region_name: str | None = None):
    date_str = date.strftime("%Y-%m-%d")
    start = ee.Date(date_str)
    geometry = (
        earth_engine.region_geometry(region_name)
        if region_name is not None
        else earth_engine.DOMAIN_POLYGON
    )
    collection = (
        ee.ImageCollection("projects/global-precipitation-nowcast/assets/global_estimation")
        .filterDate(start, start.advance(1, "day"))
        .filterBounds(geometry)
        .select(["precipitation"])
    )
    image = earth_engine.aggregate_oya_day(collection, date_str).clip(
        geometry
    ).unmask(-9999.0)
    # Region-scoped exports (PASO 3.2, secondary/refinement target of the
    # ERA5->IMERG->Oya cascade) use a v2_generative/ prefix, distinct from
    # the legacy unscoped v2_clean/oya/ path the archived deterministic
    # pipeline still reads.
    if region_name is not None:
        description = f"v2_generative_oya_{region_name}_{date_str}"
        prefix = f"v2_generative/oya/{region_name}/{date.year}/{date.month:02d}/oya_{region_name}_{date_str}"
    else:
        description = f"v2_clean_oya_{date_str}"
        prefix = f"v2_clean/oya/{date.year}/{date.month:02d}/oya_{date_str}"
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=Config.GEE_DRIVE_FOLDER,
        fileNamePrefix=prefix,
        region=geometry,
        scale=5566,
        crs="EPSG:4326",
        maxPixels=1e13,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True, "noData": -9999},
    )
    start_task_with_retry(task)
    return task


def _local_target(destination_root: Path, date: datetime, region_name: str | None) -> Path:
    if region_name is not None:
        return (
            destination_root / "oya" / region_name / f"{date.year}"
            / f"{date.month:02d}" / f"oya_{region_name}_{date:%Y-%m-%d}.tif"
        )
    return destination_root / "oya" / f"{date.year}" / f"{date.month:02d}" / f"oya_{date:%Y-%m-%d}.tif"


def export_date_range(start_year: int, end_year: int, region_name: str | None = None):
    logger.info(
        "Starting Oya acquisition for %d-%d%s",
        start_year, end_year, f" (region={region_name})" if region_name else "",
    )
    earth_engine.initialize()
    destination_root = (
        Path(Config.V2_RAW_DATA_DIR) if region_name is not None else Path(Config.RAW_DATA_DIR) / "v2_clean"
    )
    month = datetime(start_year, 1, 1)
    final_month = datetime(end_year, 12, 1)
    while month <= final_month:
        next_month = month + relativedelta(months=1)
        functions = []
        date = month
        while date < next_month:
            local = _local_target(destination_root, date, region_name)
            if not local.exists():
                functions.append(lambda value=date: [_submit(value, region_name)])
            date += relativedelta(days=1)
        if functions:
            logger.info(
                "Submitting %d missing Oya days for %s",
                len(functions),
                month.strftime("%Y-%m"),
            )
            _submit_in_batches(functions)
            downloaded = drive.sync_oya(destination_root)
            logger.info("Downloaded %d Oya files for %s", downloaded, month.strftime("%Y-%m"))
        else:
            logger.info("Oya month already complete: %s", month.strftime("%Y-%m"))
        month = next_month
    logger.info("Oya acquisition completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--region",
        default=None,
        help="Named region (src.data_preprocessing.regions). Omit to reproduce "
        "the legacy unscoped export used by the archived deterministic pipeline.",
    )
    args = parser.parse_args()
    export_date_range(args.start_year, args.end_year, args.region)


if __name__ == "__main__":
    main()
