import os
import sys
import time
import argparse
import glob
from datetime import datetime
from dateutil.relativedelta import relativedelta
import ee

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.config import Config
from src.data_extraction import gee_extractor
from src.data_extraction import drive_manager
from src.data_extraction import npz_converter

logger = Config.get_logger()

# Max tasks to submit at once before waiting for completion.
# 93 tasks/month across 3 datasets stays well under the 3,000 GEE concurrent limit.
TASK_BATCH_SIZE = 50

def _local_tif_exists(dataset, date_str):
    """Check if a raw .tif is already downloaded locally."""
    year, month, _ = date_str.split('-')
    tif_name = f"{dataset}_{date_str}.tif"
    local_path = os.path.join(Config.RAW_DATA_DIR, dataset, year, month, tif_name)
    return os.path.exists(local_path)

def submit_in_batches(task_fns):
    """Submit GEE export tasks in batches to avoid hitting concurrent task limits."""
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
    """Poll GEE until all tasks in the list have a terminal state."""
    if not tasks:
        return True

    logger.info(f"Monitoring {len(tasks)} GEE tasks...")
    task_ids = [t.id for t in tasks]

    while task_ids:
        for t_id in list(task_ids):
            status = ee.data.getTaskStatus(t_id)[0]
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
    """Run full pipeline for a single month: export → wait → download → convert."""
    logger.info(f"\n{'='*50}\nProcessing {year}-{month:02d} | target={target}\n{'='*50}")

    # --- Step 1: Build per-day export callables (skip if already local) ---
    days_start = datetime(year, month, 1)
    days_end = days_start + relativedelta(months=1)
    days_count = (days_end - days_start).days

    era5_fns, era5_pl_fns, target_fns = [], [], []
    skipped = 0

    for d in range(days_count):
        d_str = (days_start + relativedelta(days=d)).strftime("%Y-%m-%d")

        # ERA5 surface
        if not _local_tif_exists("era5", d_str):
            era5_fns.append(lambda y=year, m=month, d=d_str: [
                t for t in [_submit_era5_day(y, m, d)]
            ])
        else:
            skipped += 1

        # ERA5 pressure levels
        if not _local_tif_exists("era5_pl", d_str):
            era5_pl_fns.append(lambda y=year, m=month, d=d_str: [
                t for t in [_submit_era5_pl_day(y, m, d)]
            ])

        # Target dataset
        if not _local_tif_exists(target, d_str):
            target_fns.append(lambda y=year, m=month, d=d_str, tgt=target: [
                t for t in [_submit_target_day(y, m, d, tgt)]
            ])

    logger.info(f"Days in month: {days_count} | Skipped (already local): {skipped}")

    # --- Step 2: Submit sequentially per dataset to keep usage low ---
    # ERA5 surface first
    if era5_fns:
        logger.info("-- Submitting ERA5 surface exports to Drive --")
        submit_in_batches(era5_fns)
        logger.info("-- Downloading ERA5 surface from Drive --")
        drive_manager.sync_dataset("era5")

    # ERA5 pressure levels
    if era5_pl_fns:
        logger.info("-- Submitting ERA5 pressure level exports to Drive --")
        submit_in_batches(era5_pl_fns)
        logger.info("-- Downloading ERA5 pressure levels from Drive --")
        drive_manager.sync_dataset("era5_pl")

    # Target dataset
    if target_fns:
        logger.info(f"-- Submitting {target.upper()} exports to Drive --")
        submit_in_batches(target_fns)
        logger.info(f"-- Downloading {target.upper()} from Drive --")
        drive_manager.sync_dataset(target)

    # --- Step 3: Convert to NPZ ---
    logger.info("-- Converting downloaded TIFFs to NPZ --")
    npz_converter.run_conversion(target)

def _submit_era5_day(year, month, date_str):
    """Submit a single ERA5 surface day export to Drive and return the task."""
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
    task.start()
    logger.info(f"Submitted ERA5 surface to Drive: {date_str} → {task.id}")
    return task

def _submit_era5_pl_day(year, month, date_str):
    """Submit a single ERA5 pressure level day export to Drive and return the task."""
    d_start = ee.Date(date_str)
    d_end = d_start.advance(1, 'day')
    # Using HOURLY collection as DAILY lacks pressure levels in GEE
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
    task.start()
    logger.info(f"Submitted ERA5 pressure to Drive: {date_str} → {task.id}")
    return task


def _submit_target_day(year, month, date_str, target):
    """Submit a single CHIRPS or Oya day export to Drive and return the task."""
    d_start = ee.Date(date_str)
    d_end = d_start.advance(1, 'day')

    if target == "chirps":
        col = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
               .filterDate(d_start, d_end)
               .filterBounds(gee_extractor.DOMAIN_POLYGON)
               .select('precipitation'))
        img_list = col.map(lambda i: i.updateMask(i.gte(0))).toList(1)
        if img_list.length().getInfo() == 0:
            logger.warning(f"No CHIRPS image for {date_str}")
            return None
        img = ee.Image(img_list.get(0)).clip(gee_extractor.DOMAIN_POLYGON)
    else:  # oya
        col = (ee.ImageCollection("projects/global-precipitation-nowcast/assets/global_estimation")
               .filterDate(d_start, d_end)
               .filterBounds(gee_extractor.DOMAIN_POLYGON)
               .select(['precipitation']))
        img = col.sum().multiply(0.5).clip(gee_extractor.DOMAIN_POLYGON)

    task = ee.batch.Export.image.toDrive(
        image=img,
        description=f"{target}_{date_str}",
        folder=Config.GEE_DRIVE_FOLDER,
        fileNamePrefix=f"{target}/{year}/{month:02d}/{target}_{date_str}",
        region=gee_extractor.DOMAIN_POLYGON,
        scale=5566, crs='EPSG:4326', maxPixels=1e13
    )
    task.start()
    logger.info(f"Submitted {target.upper()} to Drive: {date_str} → {task.id}")
    return task

def run_batch(start_year, end_year, target="chirps"):
    """Run the full pipeline month-by-month for a year range."""
    if not gee_extractor.initialize_gee():
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
