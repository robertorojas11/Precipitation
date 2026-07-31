"""Orchestrator for the end-to-end data extraction pipeline.

This module coordinates GEE exports, Drive synchronization, and NPZ conversion.
It processes data month-by-month to stay within GEE concurrent task limits.
"""

import os
import sys
import time
import argparse
import glob
from datetime import datetime
from dateutil.relativedelta import relativedelta
import ee

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.config import Config
from src.data_extraction import gee_extractor
from src.data_extraction import drive_manager
from src.data_extraction import npz_converter
from src.utils.retries import get_info_with_retry, get_task_status_with_retry, start_task_with_retry

logger = Config.get_logger()
TASK_BATCH_SIZE = 50

def _local_tif_exists(dataset, date_str):
    """Checks if a raw GeoTIFF already exists on local storage.

    Args:
        dataset (str): Name of the dataset (e.g., 'era5').
        date_str (str): ISO date string (YYYY-MM-DD).

    Returns:
        bool: True if the file exists, False otherwise.
    """
    year, month, _ = date_str.split('-')
    tif_name = f"{dataset}_{date_str}.tif"
    local_path = os.path.join(Config.RAW_DATA_DIR, dataset, year, month, tif_name)
    return os.path.exists(local_path)

def submit_in_batches(task_fns):
    """Submits GEE export tasks in batches and waits for their completion.

    Args:
        task_fns (list): A list of functions that, when called, submit a GEE task.

    Returns:
        list: All submitted tasks.
    """
    all_tasks = []
    batch = []

    for fn in task_fns:
        new_tasks = fn()
        batch.extend(new_tasks)
        if len(batch) >= TASK_BATCH_SIZE:
            logger.info(f"Batch of {len(batch)} tasks submitted. Waiting for completion...")
            wait_for_tasks(batch)
            all_tasks.extend(batch)
            batch = []

    if batch:
        logger.info(f"Final batch of {len(batch)} tasks submitted. Waiting...")
        wait_for_tasks(batch)
        all_tasks.extend(batch)

    return all_tasks

def wait_for_tasks(tasks, poll_interval=60):
    """Polls GEE until all provided tasks reach a terminal state.

    Args:
        tasks (list): A list of ee.batch.Task objects.
        poll_interval (int): Seconds to wait between status checks.

    Returns:
        bool: True when all tasks have finished.
    """
    if not tasks:
        return True

    logger.info(f"Monitoring {len(tasks)} GEE tasks...")
    task_ids = [t.id for t in tasks]

    while task_ids:
        for t_id in list(task_ids):
            status = get_task_status_with_retry(t_id)
            state = status['state']
            if state in ['COMPLETED', 'FAILED', 'CANCELLED']:
                if state == 'FAILED':
                    logger.error(f"Task {t_id} FAILED: {status.get('error_message', 'Unknown')}")
                else:
                    logger.info(f"Task {t_id}: {state}")
                task_ids.remove(t_id)

        if task_ids:
            logger.info(f"{len(task_ids)} tasks still running. Sleeping {poll_interval}s...")
            time.sleep(poll_interval)

    logger.info("All tasks finished.")
    return True

def run_month(year, month, target):
    """Runs the full pipeline (export -> download -> convert) for a single month.

    Args:
        year (int): The year to process.
        month (int): The month to process.
        target (str): The target dataset name (e.g., 'chirps').
    """
    logger.info(f"\n{'='*50}\nProcessing {year}-{month:02d} | target={target}\n{'='*50}")

    days_start = datetime(year, month, 1)
    days_end = days_start + relativedelta(months=1)
    days_count = (days_end - days_start).days

    era5_fns, era5_pl_fns, target_fns = [], [], []
    skipped = 0

    for d in range(days_count):
        d_str = (days_start + relativedelta(days=d)).strftime("%Y-%m-%d")

        if not _local_tif_exists("era5", d_str):
            era5_fns.append(lambda y=year, m=month, d=d_str: [
                t for t in [_submit_era5_day(y, m, d)]
            ])
        else:
            skipped += 1

        if not _local_tif_exists("era5_pl", d_str):
            era5_pl_fns.append(lambda y=year, m=month, d=d_str: [
                t for t in [_submit_era5_pl_day(y, m, d)]
            ])

        if not _local_tif_exists(target, d_str):
            target_fns.append(lambda y=year, m=month, d=d_str, tgt=target: [
                t for t in [_submit_target_day(y, m, d, tgt)] if t is not None
            ])

    logger.info(f"Days in month: {days_count} | Skipped (already local): {skipped}")

    if era5_fns:
        logger.info(f"-- Submitting ERA5 surface exports to Drive --")
        submit_in_batches(era5_fns)
        logger.info(f"-- Downloading ERA5 surface from Drive --")
        drive_manager.sync_dataset("era5")

    if era5_pl_fns:
        logger.info(f"-- Submitting ERA5 pressure level exports to Drive --")
        submit_in_batches(era5_pl_fns)
        logger.info(f"-- Downloading ERA5 pressure levels from Drive --")
        drive_manager.sync_dataset("era5_pl")

    if target_fns:
        logger.info(f"-- Submitting {target.upper()} exports to Drive --")
        submit_in_batches(target_fns)
        logger.info(f"-- Downloading {target.upper()} from Drive --")
        drive_manager.sync_dataset(target)

    logger.info("-- Converting downloaded TIFFs to NPZ --")
    npz_converter.run_conversion(target)

def _submit_era5_day(year, month, date_str):
    """Submits a GEE export task for a single day of ERA5 surface data.

    Args:
        year (int): Year of the data.
        month (int): Month of the data.
        date_str (str): ISO date string (YYYY-MM-DD).

    Returns:
        ee.batch.Task: The submitted GEE task.
    """
    d_start = ee.Date(date_str)
    d_end = d_start.advance(1, 'day')
    era5 = (ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
            .filterDate(d_start, d_end)
            .filterBounds(gee_extractor.DOMAIN_POLYGON)
            .select(gee_extractor.SURFACE_BANDS))
    daily_precip = era5.select(['total_precipitation_hourly']).sum()
    daily_others = era5.select(gee_extractor.OTHER_BANDS).mean()
    img = daily_precip.addBands(daily_others).clip(gee_extractor.DOMAIN_POLYGON)
    task = ee.batch.Export.image.toDrive(
        image=img,
        description=f"era5_{date_str}",
        folder=Config.GEE_DRIVE_FOLDER,
        fileNamePrefix=f"era5/{year}/{month:02d}/era5_{date_str}",
        region=gee_extractor.DOMAIN_POLYGON,
        scale=27750, crs='EPSG:4326', maxPixels=1e13
    )
    start_task_with_retry(task)
    logger.info(f"Submitted ERA5 surface to Drive: {date_str} → {task.id}")
    return task

def _submit_era5_pl_day(year, month, date_str):
    """Submits a GEE export task for a single day of ERA5 pressure level data.

    Args:
        year (int): Year of the data.
        month (int): Month of the data.
        date_str (str): ISO date string (YYYY-MM-DD).

    Returns:
        ee.batch.Task: The submitted GEE task.
    """
    d_start = ee.Date(date_str)
    d_end = d_start.advance(1, 'day')
    era5_pl = (ee.ImageCollection("ECMWF/ERA5/HOURLY")
               .filterDate(d_start, d_end)
               .filterBounds(gee_extractor.DOMAIN_POLYGON)
               .select(gee_extractor.PRESSURE_BANDS))
    img = era5_pl.mean().clip(gee_extractor.DOMAIN_POLYGON)
    task = ee.batch.Export.image.toDrive(
        image=img,
        description=f"era5_pl_{date_str}",
        folder=Config.GEE_DRIVE_FOLDER,
        fileNamePrefix=f"era5_pl/{year}/{month:02d}/era5_pl_{date_str}",
        region=gee_extractor.DOMAIN_POLYGON,
        scale=27750, crs='EPSG:4326', maxPixels=1e13
    )
    start_task_with_retry(task)
    logger.info(f"Submitted ERA5 pressure to Drive: {date_str} → {task.id}")
    return task


def _submit_target_day(year, month, date_str, target):
    """Submits a GEE export task for a single day of the target dataset.

    Args:
        year (int): Year of the data.
        month (int): Month of the data.
        date_str (str): ISO date string (YYYY-MM-DD).
        target (str): Name of the target dataset ('chirps' or 'oya').

    Returns:
        ee.batch.Task or None: The submitted GEE task or None if no data.
    """
    d_start = ee.Date(date_str)
    d_end = d_start.advance(1, 'day')

    if target == "chirps":
        col = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
               .filterDate(d_start, d_end)
               .filterBounds(gee_extractor.DOMAIN_POLYGON)
               .select('precipitation'))
        img_list = col.map(lambda i: i.updateMask(i.gte(0))).toList(1)
        if get_info_with_retry(img_list.length()) == 0:
            logger.warning(f"No CHIRPS image for {date_str}")
            return None
        img = ee.Image(img_list.get(0)).clip(gee_extractor.DOMAIN_POLYGON)
    else:
        col = (ee.ImageCollection("projects/global-precipitation-nowcast/assets/global_estimation")
               .filterDate(d_start, d_end)
               .filterBounds(gee_extractor.DOMAIN_POLYGON)
               .select(['precipitation']))
        if get_info_with_retry(col.size()) == 0:
            logger.warning(f"No OYA image for {date_str}")
            return None
        img = col.sum().multiply(0.5).clip(gee_extractor.DOMAIN_POLYGON)

    task = ee.batch.Export.image.toDrive(
        image=img,
        description=f"{target}_{date_str}",
        folder=Config.GEE_DRIVE_FOLDER,
        fileNamePrefix=f"{target}/{year}/{month:02d}/{target}_{date_str}",
        region=gee_extractor.DOMAIN_POLYGON,
        scale=5566, crs='EPSG:4326', maxPixels=1e13
    )
    start_task_with_retry(task)
    logger.info(f"Submitted {target.upper()} to Drive: {date_str} → {task.id}")
    return task

def run_batch(start_year, end_year, target="chirps"):
    """Orchestrates the pipeline month-by-month for a specified year range.

    Args:
        start_year (int): The starting year of the range.
        end_year (int): The ending year of the range.
        target (str): Name of the target dataset ('chirps' or 'oya').
    """
    if not gee_extractor.initialize_gee():
        sys.exit(1)
        
    if gee_extractor.AUTH_MODE == 'service_account':
        logger.error(
            "\n" + "="*80 + "\n"
            "CRITICAL ERROR: Google Earth Engine has initialized using a Service Account.\n"
            "Because GEE Service Accounts do not have personal Google Drive storage space,\n"
            "all exports to Google Drive will FAIL with 'Service accounts do not have storage quota'.\n\n"
            "To resolve this, you MUST authenticate using your personal Google account. Please run:\n"
            "    earthengine authenticate --auth_mode=notebook\n"
            "in your terminal, follow the instructions to log in, and then re-run the pipeline.\n"
            "Once authenticated, GEE will use your personal Drive quota to export files,\n"
            "while the service account will still be used to download them to local storage.\n" +
            "="*80 + "\n"
        )
        sys.exit(1)

    current_dt = datetime(start_year, 1, 1)
    end_dt = datetime(end_year, 12, 1)

    while current_dt <= end_dt:
        run_month(current_dt.year, current_dt.month, target)
        current_dt += relativedelta(months=1)

    logger.info("All months complete. Running final status report...")
    from src.utils import status_report
    status_report.check_completeness(target, start_year, end_year)
    status_report.check_storage_efficiency(target)

def main():
    parser = argparse.ArgumentParser(description="Run the full Data Extraction Pipeline.")
    parser.add_argument("--start_year", type=int, required=True)
    parser.add_argument("--end_year", type=int, required=True)
    parser.add_argument("--target", type=str, choices=["chirps", "oya"], default="chirps")

    args = parser.parse_args()
    run_batch(args.start_year, args.end_year, args.target)

if __name__ == "__main__":
    main()
