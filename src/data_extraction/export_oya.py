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


def _wait_for_tasks(tasks, poll_interval: int = 60) -> None:
    pending = {task.id for task in tasks}
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
            time.sleep(poll_interval)


def _submit_in_batches(task_factories) -> None:
    for offset in range(0, len(task_factories), TASK_BATCH_SIZE):
        tasks = [
            task
            for factory in task_factories[offset : offset + TASK_BATCH_SIZE]
            for task in factory()
        ]
        _wait_for_tasks(tasks)


def _submit(date: datetime):
    date_str = date.strftime("%Y-%m-%d")
    start = ee.Date(date_str)
    collection = (
        ee.ImageCollection("projects/global-precipitation-nowcast/assets/global_estimation")
        .filterDate(start, start.advance(1, "day"))
        .filterBounds(earth_engine.DOMAIN_POLYGON)
        .select(["precipitation"])
    )
    image = earth_engine.aggregate_oya_day(collection, date_str).clip(
        earth_engine.DOMAIN_POLYGON
    ).unmask(-9999.0)
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f"v2_clean_oya_{date_str}",
        folder=Config.GEE_DRIVE_FOLDER,
        fileNamePrefix=f"v2_clean/oya/{date.year}/{date.month:02d}/oya_{date_str}",
        region=earth_engine.DOMAIN_POLYGON,
        scale=5566,
        crs="EPSG:4326",
        maxPixels=1e13,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True, "noData": -9999},
    )
    start_task_with_retry(task)
    return task


def export_date_range(start_year: int, end_year: int):
    earth_engine.initialize()
    destination_root = Path(Config.RAW_DATA_DIR) / "v2_clean"
    month = datetime(start_year, 1, 1)
    final_month = datetime(end_year, 12, 1)
    while month <= final_month:
        next_month = month + relativedelta(months=1)
        functions = []
        date = month
        while date < next_month:
            local = destination_root / "oya" / f"{date.year}" / f"{date.month:02d}" / f"oya_{date:%Y-%m-%d}.tif"
            if not local.exists():
                functions.append(lambda value=date: [_submit(value)])
            date += relativedelta(days=1)
        if functions:
            _submit_in_batches(functions)
            drive.sync_oya(destination_root)
        month = next_month


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()
    export_date_range(args.start_year, args.end_year)


if __name__ == "__main__":
    main()
