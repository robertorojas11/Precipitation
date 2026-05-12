import os
import sys
import time
import argparse
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

def wait_for_tasks(tasks, poll_interval=60):
    """Wait for a list of GEE tasks to complete."""
    if not tasks:
        return True
        
    logger.info(f"Waiting for {len(tasks)} GEE tasks to complete...")
    task_ids = [t.id for t in tasks]
    
    while task_ids:
        for t_id in list(task_ids):
            status = ee.data.getTaskStatus(t_id)[0]
            state = status['state']
            
            if state in ['COMPLETED', 'FAILED', 'CANCELLED']:
                logger.info(f"Task {t_id} finished with state: {state}")
                if state == 'FAILED':
                    logger.error(f"Task failed: {status.get('error_message', 'Unknown error')}")
                task_ids.remove(t_id)
                
        if task_ids:
            logger.info(f"{len(task_ids)} tasks remaining. Sleeping for {poll_interval}s...")
            time.sleep(poll_interval)
            
    logger.info("All tracked tasks have finished.")
    return True

def run_batch(start_year, end_year, target="chirps"):
    """Run extraction, download, and conversion pipeline."""
    if not gee_extractor.initialize_gee():
        sys.exit(1)
        
    drive_folder = Config.GEE_DRIVE_FOLDER
    
    start_dt = datetime(start_year, 1, 1)
    end_dt = datetime(end_year, 12, 1)
    
    current_dt = start_dt
    
    while current_dt <= end_dt:
        year = current_dt.year
        month = current_dt.month
        logger.info(f"\n{'='*40}\nProcessing {year}-{month:02d}\n{'='*40}")
        
        # 1. Trigger Exports
        tasks = []
        tasks.extend(gee_extractor.export_era5(year, month, drive_folder))
        tasks.extend(gee_extractor.export_era5_pressure(year, month, drive_folder))
        
        if target == "chirps":
            tasks.extend(gee_extractor.export_chirps(year, month, drive_folder))
        elif target == "oya":
            tasks.extend(gee_extractor.export_oya(year, month, drive_folder))
            
        # 2. Wait for GEE
        wait_for_tasks(tasks)
        
        # 3. Download from Drive
        logger.info("\n--- Syncing from Drive ---")
        drive_manager.sync_dataset("era5")
        drive_manager.sync_dataset("era5_pl")
        drive_manager.sync_dataset(target)
        
        # 4. Convert to NPZ (can do this month by month or all at once at the end)
        # We will do it in a batch for the whole dataset at the end to generate the index properly,
        # but running it iteratively is fine since the function scans all downloaded TIFFs.
        logger.info("\n--- Converting to NPZ ---")
        npz_converter.run_conversion(target)
        
        current_dt += relativedelta(months=1)
        
    logger.info("Batch processing completed!")

def main():
    parser = argparse.ArgumentParser(description="Run the full Data Extraction Pipeline.")
    parser.add_argument("--start_year", type=int, required=True, help="Start year (e.g., 2004)")
    parser.add_argument("--end_year", type=int, required=True, help="End year (e.g., 2019)")
    parser.add_argument("--target", type=str, choices=["chirps", "oya"], default="chirps", help="Target dataset")
    
    args = parser.parse_args()
    
    run_batch(args.start_year, args.end_year, args.target)

if __name__ == "__main__":
    main()
