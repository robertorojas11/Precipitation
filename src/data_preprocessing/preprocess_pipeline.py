"""Data preprocessing pipeline orchestrator for precipitation downscaling.

Crops ERA5 inputs and targets to the Mexico domain bounding box,
computes Upslope and Spectral physics models, and saves the stacked arrays
back into the NPZ archives.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.config import Config
from src.data_preprocessing.physics_models import (
    resample_dem,
    compute_terrain_gradients,
    compute_upslope_model,
    compute_spectral_model
)

logger = Config.get_logger()

# Mexico bounding box indices in raw 0.05° grid:
ROW_START, ROW_END = 49, 509   # Latitude 35°N to 12°N
COL_START, COL_END = 270, 990  # Longitude 120°W to 84°W
HEIGHT, WIDTH = ROW_END - ROW_START, COL_END - COL_START # 460 x 720

def run_preprocessing_pipeline(target_name, overwrite=False):
    """Orchestrates the preprocessing of raw NPZ files.

    Loads the static DEM once, computes its spatial gradients, and then
    processes each date by cropping and calculating physical model outputs.
    """
    logger.info(f"Starting Step 2 Preprocessing Pipeline for target: {target_name}")
    
    # 1. Load and prepare static topography data
    dem_path = os.path.join(Config.RAW_DATA_DIR, "dem", "nasadem_mexico_1km.tif")
    if not os.path.exists(dem_path):
        logger.error(f"NASADEM file not found: {dem_path}. Please download/extract it first.")
        return
        
    logger.info("Resampling and cropping NASADEM to 5km target grid...")
    try:
        elevation = resample_dem(dem_path, HEIGHT, WIDTH)
        # Compute spatial derivatives
        dz_dx, dz_dy = compute_terrain_gradients(elevation)
    except Exception as e:
        logger.error(f"Failed to prepare terrain data: {e}")
        return
        
    # Expand elevation to (H, W, 1) for stacking
    elevation_chan = np.expand_dims(elevation, axis=-1)
    
    # 2. Load dataset index
    metadata_dir = os.path.join(Config.LOCAL_DATA_DIR, "metadata")
    index_path = os.path.join(metadata_dir, f"dataset_index_{target_name}.csv")
    if not os.path.exists(index_path):
        logger.error(f"Dataset index file not found: {index_path}")
        return
        
    df = pd.read_csv(index_path)
    # Filter for files that are marked valid
    valid_df = df[df['valid_flag'] == True]
    
    if valid_df.empty:
        logger.warning("No valid records found in the dataset index.")
        return
        
    logger.info(f"Processing {len(valid_df)} dates...")
    processed_count = 0
    skipped_count = 0
    
    for idx, row in valid_df.iterrows():
        npz_path = row['npz_path']
        date_str = row['date']
        
        if not os.path.exists(npz_path):
            logger.warning(f"File listed in index does not exist: {npz_path}")
            continue
            
        try:
            # Check if file has already been preprocessed
            if not overwrite:
                with np.load(npz_path) as data:
                    if 'upslope' in data.files and 'spectral' in data.files and 'elevation' in data.files:
                        # Ensure shapes are already cropped
                        if data['inputs'].shape[0] == HEIGHT and data['inputs'].shape[1] == WIDTH:
                            skipped_count += 1
                            continue
                            
            with np.load(npz_path) as data:
                inputs_raw = data['inputs'] # Shape: (H_raw, W_raw, 18)
                target_raw = data['target'] # Shape: (H_raw, W_raw, 1)
                
            # Crop to Mexico bounding box
            inputs_cropped = inputs_raw[ROW_START:ROW_END, COL_START:COL_END, :]
            target_cropped = target_raw[ROW_START:ROW_END, COL_START:COL_END, :]
            
            # Quality Control: clip negative targets to 0
            target_cropped = np.maximum(target_cropped, 0.0)
            
            # Extract winds and RH at 850 hPa (indices: u=13, v=15, rh=17)
            u_850 = inputs_cropped[:, :, 13]
            v_850 = inputs_cropped[:, :, 15]
            rh_850 = inputs_cropped[:, :, 17]
            
            # 3. Compute physical model fields
            upslope = compute_upslope_model(u_850, v_850, rh_850, dz_dx, dz_dy)
            spectral = compute_spectral_model(u_850, v_850, elevation)
            
            # Expand dimensions to (H, W, 1) for stacking
            upslope_chan = np.expand_dims(upslope, axis=-1)
            spectral_chan = np.expand_dims(spectral, axis=-1)
            
            # 4. Save preprocessed arrays in-place
            # Store inputs, target (cropped), upslope, spectral, and elevation
            np.savez_compressed(
                npz_path,
                inputs=inputs_cropped,
                target=target_cropped,
                upslope=upslope_chan,
                spectral=spectral_chan,
                elevation=elevation_chan,
                date=date_str
            )
            
            processed_count += 1
            if processed_count % 500 == 0:
                logger.info(f"Successfully preprocessed {processed_count} files...")
                
        except Exception as e:
            logger.error(f"Error preprocessing {date_str} ({npz_path}): {e}")
            
    logger.info(f"Preprocessing completed. Processed: {processed_count} | Skipped (already done): {skipped_count}")

def main():
    parser = argparse.ArgumentParser(description="Step 2 Preprocessing & Physics Models Pipeline")
    parser.add_argument("--target", type=str, choices=["chirps", "oya"], required=True, help="Target dataset")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing preprocessed files")
    args = parser.parse_args()
    
    run_preprocessing_pipeline(args.target, args.overwrite)

if __name__ == "__main__":
    main()
