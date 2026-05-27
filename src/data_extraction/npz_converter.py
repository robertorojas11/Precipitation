"""Dataset conversion from GeoTIFF to compressed NPZ format.

This module handles spatial resampling of ERA5 data to match the resolution
of target datasets (CHIRPS/Oya), stacks atmospheric bands, and saves
the paired inputs/targets as compressed .npz archives for model training.
"""

import os
import sys
import argparse
import glob
from datetime import datetime
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.config import Config

logger = Config.get_logger()

def get_split(date_str):
    """Determines the dataset split (train/val/test) based on the date.

    Args:
        date_str (str): ISO date string (YYYY-MM-DD).

    Returns:
        str: The split name ('train', 'val', 'test', or 'other').
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    year = dt.year
    if 2004 <= year <= 2015:
        return "train"
    elif 2016 <= year <= 2017:
        return "val"
    elif 2018 <= year <= 2019:
        return "test"
    return "other"

def process_date(date_str, target_name, era5_path, era5_pl_path, target_path, processed_dir):
    """Resamples ERA5 data to the target grid and saves a paired NPZ file.

    Args:
        date_str (str): ISO date string (YYYY-MM-DD).
        target_name (str): Name of the target dataset (e.g., 'chirps').
        era5_path (str): Local path to the ERA5 surface GeoTIFF.
        era5_pl_path (str): Local path to the ERA5 pressure level GeoTIFF.
        target_path (str): Local path to the target GeoTIFF.
        processed_dir (str): Root directory for saving processed files.

    Returns:
        tuple: (split_name, output_path, success_flag).
    """
    split = get_split(date_str)
    out_dir = os.path.join(processed_dir, target_name, split)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{date_str}.npz")
    
    if os.path.exists(out_path):
        logger.debug(f"NPZ already exists for {date_str}, skipping.")
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
            
        with rasterio.open(era5_pl_path) as era5_pl_src:
            era5_pl_data = era5_pl_src.read()
            era5_pl_bands = era5_pl_src.count
            
            resampled_era5_pl = np.empty((era5_pl_bands, tgt_height, tgt_width), dtype=np.float32)
            
            reproject(
                source=era5_pl_data,
                destination=resampled_era5_pl,
                src_transform=era5_pl_src.transform,
                src_crs=era5_pl_src.crs,
                dst_transform=tgt_transform,
                dst_crs=tgt_crs,
                resampling=Resampling.bilinear
            )
            
            resampled_era5_pl = np.transpose(resampled_era5_pl, (1, 2, 0))
            
        # Stack surface and pressure level bands: 10 surface + 8 pressure = (H, W, 18)
        stacked_era5 = np.concatenate([resampled_era5, resampled_era5_pl], axis=-1)
            
        np.savez_compressed(
            out_path,
            inputs=stacked_era5,
            target=target_data,
            date=date_str
        )
        logger.info(f"Created {out_path}")
        return split, out_path, True
        
    except Exception as e:
        logger.error(f"Error processing {date_str}: {e}")
        return split, "", False

def run_conversion(target_name):
    """Orchestrates the conversion process for all available files of a target.

    Args:
        target_name (str): Name of the target dataset (e.g., 'chirps').
    """
    logger.info(f"Starting NPZ conversion for target: {target_name}")
    
    era5_dir = os.path.join(Config.RAW_DATA_DIR, "era5")
    era5_pl_dir = os.path.join(Config.RAW_DATA_DIR, "era5_pl")
    target_dir = os.path.join(Config.RAW_DATA_DIR, target_name)
    processed_dir = Config.PROCESSED_DATA_DIR
    metadata_dir = os.path.join(Config.LOCAL_DATA_DIR, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Find all target files
    target_files = glob.glob(os.path.join(target_dir, "**", f"{target_name}_*.tif"), recursive=True)
    logger.info(f"Found {len(target_files)} {target_name} files.")
    
    records = []
    skipped_count = 0
    
    for tgt_path in target_files:
        filename = os.path.basename(tgt_path)
        date_str = filename.replace(f"{target_name}_", "").replace(".tif", "")
        
        year, month, _ = date_str.split('-')
        era5_path = os.path.join(era5_dir, year, month, f"era5_{date_str}.tif")
        era5_pl_path = os.path.join(era5_pl_dir, year, month, f"era5_pl_{date_str}.tif")
        
        # Check if NPZ already exists
        split = get_split(date_str)
        npz_path = os.path.join(processed_dir, target_name, split, f"{date_str}.npz")
        if os.path.exists(npz_path):
            skipped_count += 1
            records.append({
                "date": date_str,
                "split": split,
                "era5_path": era5_path,
                "era5_pl_path": era5_pl_path,
                "target_path": tgt_path,
                "npz_path": npz_path,
                "valid_flag": True
            })
            continue
            
        if not os.path.exists(era5_path) or not os.path.exists(era5_pl_path):
            logger.warning(f"ERA5 or ERA5_PL file missing for {date_str}")
            records.append({
                "date": date_str,
                "split": get_split(date_str),
                "era5_path": era5_path,
                "era5_pl_path": era5_pl_path,
                "target_path": tgt_path,
                "valid_flag": False
            })
            continue
            
        split, npz_path, valid = process_date(date_str, target_name, era5_path, era5_pl_path, tgt_path, processed_dir)
        
        records.append({
            "date": date_str,
            "split": split,
            "era5_path": era5_path,
            "era5_pl_path": era5_pl_path,
            "target_path": tgt_path,
            "npz_path": npz_path,
            "valid_flag": valid
        })
        
    if skipped_count > 0:
        logger.info(f"NPZ already exists for {skipped_count} dates, skipping conversion for those dates.")
        
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
