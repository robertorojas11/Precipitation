import os
import sys
import argparse
import glob
from datetime import datetime
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import pandas as pd

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.config import Config

logger = Config.get_logger()

def get_split(date_str):
    """Determine data split based on date."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    year = dt.year
    if 2004 <= year <= 2015:
        return "train"
    elif 2016 <= year <= 2017:
        return "val"
    elif 2018 <= year <= 2019:
        return "test"
    else:
        return "other"

def process_date(date_str, target_name, era5_path, target_path, processed_dir):
    """Resample ERA5 and save paired NPZ."""
    split = get_split(date_str)
    out_dir = os.path.join(processed_dir, target_name, split)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{date_str}.npz")
    
    if os.path.exists(out_path):
        logger.info(f"NPZ already exists for {date_str}, skipping.")
        return split, out_path, True
        
    try:
        with rasterio.open(target_path) as tgt_src:
            target_data = tgt_src.read(1)
            # Expand dims to (H, W, 1) as requested by architecture
            target_data = np.expand_dims(target_data, axis=-1)
            tgt_profile = tgt_src.profile
            tgt_transform = tgt_src.transform
            tgt_crs = tgt_src.crs
            tgt_height = tgt_src.height
            tgt_width = tgt_src.width
            
        with rasterio.open(era5_path) as era5_src:
            era5_data = era5_src.read() # Shape: (Bands, H, W)
            era5_bands = era5_src.count
            
            # Destination array for resampled ERA5: (Bands, H, W)
            resampled_era5 = np.empty((era5_bands, tgt_height, tgt_width), dtype=np.float32)
            
            reproject(
                source=era5_data,
                destination=resampled_era5,
                src_transform=era5_src.transform,
                src_crs=era5_src.crs,
                dst_transform=tgt_transform,
                dst_crs=tgt_crs,
                resampling=Resampling.bilinear
            )
            
            # Change shape from (Bands, H, W) to (H, W, Bands)
            resampled_era5 = np.transpose(resampled_era5, (1, 2, 0))
            
        np.savez_compressed(
            out_path,
            inputs=resampled_era5,
            target=target_data,
            date=date_str
        )
        logger.info(f"Created {out_path}")
        return split, out_path, True
        
    except Exception as e:
        logger.error(f"Error processing {date_str}: {e}")
        return split, "", False

def run_conversion(target_name):
    """Run the conversion pipeline for a target dataset."""
    logger.info(f"Starting NPZ conversion for target: {target_name}")
    
    era5_dir = os.path.join(Config.RAW_DATA_DIR, "era5")
    target_dir = os.path.join(Config.RAW_DATA_DIR, target_name)
    processed_dir = Config.PROCESSED_DATA_DIR
    metadata_dir = os.path.join(Config.LOCAL_DATA_DIR, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Find all target files
    target_files = glob.glob(os.path.join(target_dir, "**", f"{target_name}_*.tif"), recursive=True)
    logger.info(f"Found {len(target_files)} {target_name} files.")
    
    records = []
    
    for tgt_path in target_files:
        filename = os.path.basename(tgt_path)
        date_str = filename.replace(f"{target_name}_", "").replace(".tif", "")
        
        year, month, _ = date_str.split('-')
        era5_path = os.path.join(era5_dir, year, month, f"era5_{date_str}.tif")
        
        if not os.path.exists(era5_path):
            logger.warning(f"ERA5 file missing for {date_str}: {era5_path}")
            records.append({
                "date": date_str,
                "split": get_split(date_str),
                "era5_path": era5_path,
                "target_path": tgt_path,
                "valid_flag": False
            })
            continue
            
        split, npz_path, valid = process_date(date_str, target_name, era5_path, tgt_path, processed_dir)
        
        records.append({
            "date": date_str,
            "split": split,
            "era5_path": era5_path,
            "target_path": tgt_path,
            "npz_path": npz_path,
            "valid_flag": valid
        })
        
    # Save index
    index_path = os.path.join(metadata_dir, f"dataset_index_{target_name}.csv")
    df = pd.DataFrame(records)
    if not df.empty:
        # Sort by date
        df = df.sort_values(by="date")
        df.to_csv(index_path, index=False)
        logger.info(f"Saved dataset index to {index_path}")
    else:
        logger.warning(f"No records processed for {target_name}.")

def main():
    parser = argparse.ArgumentParser(description="Convert raw GeoTIFFs to paired NPZ files.")
    parser.add_argument("--target", type=str, choices=["chirps", "oya"], required=True, help="Target dataset")
    args = parser.parse_args()
    
    run_conversion(args.target)

if __name__ == "__main__":
    main()
