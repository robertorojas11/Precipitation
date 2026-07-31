import os
import sys
import glob
import pandas as pd
from datetime import datetime, timedelta

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.config import Config

logger = Config.get_logger()

def check_completeness(target_name, start_year=2004, end_year=2025):
    """Verify that all expected dates have valid NPZ files."""
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    
    expected_days = (end_date - start_date).days + 1
    logger.info(f"Checking completeness for {target_name} ({start_year}-{end_year})")
    logger.info(f"Expected total days: {expected_days}")
    
    index_path = os.path.join(Config.LOCAL_DATA_DIR, "metadata", f"dataset_index_{target_name}.csv")
    if not os.path.exists(index_path):
        logger.error(f"Index file not found: {index_path}")
        return
        
    df = pd.read_csv(index_path)
    # Filter only valid flags
    valid_df = df[df['valid_flag'] == True]
    
    # Check date coverage
    found_dates = pd.to_datetime(valid_df['date']).dt.date.tolist()
    expected_dates = [ (start_date + timedelta(days=i)).date() for i in range(expected_days) ]
    
    missing_dates = set(expected_dates) - set(found_dates)
    
    completeness_pct = (len(found_dates) / expected_days) * 100
    
    logger.info(f"Completeness Metric: {completeness_pct:.2f}% ({len(found_dates)}/{expected_days})")
    if missing_dates:
        logger.warning(f"Missing {len(missing_dates)} dates. First few missing: {sorted(list(missing_dates))[:5]}")
    else:
        logger.info("SUCCESS: 100% of requested days are available and valid.")

def check_storage_efficiency(target_name):
    """Report disk usage metrics for the processed dataset."""
    processed_dir = os.path.join(Config.PROCESSED_DATA_DIR, target_name)
    if not os.path.exists(processed_dir):
        logger.error(f"Processed directory not found: {processed_dir}")
        return
        
    npz_files = glob.glob(os.path.join(processed_dir, "**", "*.npz"), recursive=True)
    
    if not npz_files:
        logger.warning(f"No .npz files found in {processed_dir}")
        return
        
    total_size_bytes = sum(os.path.getsize(f) for f in npz_files)
    total_size_gb = total_size_bytes / (1024 ** 3)
    avg_size_mb = (total_size_bytes / len(npz_files)) / (1024 ** 2)
    
    logger.info(f"Storage Efficiency Metric for {target_name}:")
    logger.info(f"  Total NPZ files: {len(npz_files)}")
    logger.info(f"  Total disk space used: {total_size_gb:.2f} GB")
    logger.info(f"  Average size per day: {avg_size_mb:.2f} MB")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline Success Metrics Status Report")
    parser.add_argument("--target", type=str, choices=["chirps", "oya"], default="chirps")
    parser.add_argument("--start", type=int, default=2004)
    parser.add_argument("--end", type=int, default=2025)
    args = parser.parse_args()
    
    logger.info("="*50)
    logger.info("PIPELINE STATUS REPORT")
    logger.info("="*50)
    check_completeness(args.target, args.start, args.end)
    logger.info("-" * 50)
    check_storage_efficiency(args.target)
    logger.info("="*50)

if __name__ == "__main__":
    main()
