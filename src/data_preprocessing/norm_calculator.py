"""Z-score normalization statistics calculator for precipitation downscaling.

This module computes the mean and standard deviation for all 18 input bands and
the target precipitation over the training split (2004–2019) of the dataset.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.config import Config

logger = Config.get_logger()

# Mexico bounding box indices in raw 0.05° grid:
ROW_START, ROW_END = 49, 509   # Latitude 35°N to 12°N
COL_START, COL_END = 270, 990  # Longitude 120°W to 84°W
HEIGHT, WIDTH = ROW_END - ROW_START, COL_END - COL_START # 460 x 720

def compute_normalization_stats(target_name, log_transform_precip=True):
    """Computes Z-score statistics (mean/std) on the training dataset.

    Args:
        target_name (str): 'chirps' or 'oya'.
        log_transform_precip (bool): If True, apply log1p to precip fields.
    """
    logger.info(f"Computing normalization stats for {target_name} training split...")
    
    metadata_dir = os.path.join(Config.LOCAL_DATA_DIR, "metadata")
    index_path = os.path.join(metadata_dir, f"dataset_index_{target_name}.csv")
    
    if not os.path.exists(index_path):
        logger.error(f"Dataset index file not found: {index_path}")
        return
        
    df = pd.read_csv(index_path)
    
    # Filter for valid training records only (split == 'train' and valid_flag == True)
    train_df = df[(df['split'] == 'train') & (df['valid_flag'] == True)]
    
    if train_df.empty:
        logger.warning("No valid training records found in dataset index.")
        return
        
    num_samples = len(train_df)
    logger.info(f"Found {num_samples} training samples.")
    
    # Grid dimensions after cropping
    num_pixels = HEIGHT * WIDTH
    total_pixels = num_samples * num_pixels
    
    # Accumulators for input bands (up to 21 bands)
    num_channels = 21
    input_sum = np.zeros(num_channels, dtype=np.float64)
    input_sum_sq = np.zeros(num_channels, dtype=np.float64)
    input_counts = np.zeros(num_channels, dtype=np.float64)
    
    # Accumulators for target (1 band)
    target_sum = 0.0
    target_sum_sq = 0.0
    target_counts = 0.0
    
    processed_count = 0
    
    for idx, row in train_df.iterrows():
        npz_path = row['npz_path']
        
        # Robust path replacement for migrated environments
        if not os.path.exists(npz_path) and npz_path.startswith("./data/"):
            npz_path = npz_path.replace("./data/processed", Config.PROCESSED_DATA_DIR).replace("./data/raw", Config.RAW_DATA_DIR)
            
        if not os.path.exists(npz_path):
            logger.warning(f"File missing during stat calculation: {npz_path}")
            continue
            
        try:
            with np.load(npz_path) as data:
                inputs = data['inputs']
                target = data['target']
                
                # Check if file has already been preprocessed (has physics channels)
                is_preprocessed = ('upslope' in data.files and 'spectral' in data.files and 'elevation' in data.files)
                
                if is_preprocessed:
                    inputs_cropped = inputs
                    target_cropped = target[:, :, 0] if len(target.shape) == 3 else target
                    upslope = data['upslope']
                    spectral = data['spectral']
                    elevation = data['elevation']
                    
                    # Concatenate to 21 bands
                    inputs_all = np.concatenate([inputs_cropped, upslope, spectral, elevation], axis=-1)
                else:
                    # Crop raw files
                    inputs_cropped = inputs[ROW_START:ROW_END, COL_START:COL_END, :]
                    target_cropped = target[ROW_START:ROW_END, COL_START:COL_END, 0]
                    # Fill dummy values for the extra 3 channels
                    dummy_extra = np.zeros((HEIGHT, WIDTH, 3), dtype=inputs_cropped.dtype)
                    inputs_all = np.concatenate([inputs_cropped, dummy_extra], axis=-1)
            
            # Quality Control: clip negative precip target values to 0
            target_cropped = np.maximum(target_cropped, 0.0)
            
            # Extract and copy to avoid modifying original array references
            inputs_proc = np.array(inputs_all, dtype=np.float64)
            target_proc = np.array(target_cropped, dtype=np.float64)
            
            if log_transform_precip:
                # Band 0 of inputs is 'total_precipitation_hourly'
                inputs_proc[:, :, 0] = np.log1p(inputs_proc[:, :, 0])
                target_proc = np.log1p(target_proc)
                
            # Accumulate inputs (ignoring NaNs)
            non_nan_inputs = ~np.isnan(inputs_proc)
            input_sum += np.nansum(inputs_proc, axis=(0, 1))
            input_sum_sq += np.nansum(inputs_proc ** 2, axis=(0, 1))
            input_counts += np.sum(non_nan_inputs, axis=(0, 1))
            
            # Accumulate target (ignoring NaNs)
            non_nan_target = ~np.isnan(target_proc)
            target_sum += np.nansum(target_proc)
            target_sum_sq += np.nansum(target_proc ** 2)
            target_counts += np.sum(non_nan_target)
            
            processed_count += 1
            if processed_count % 500 == 0:
                logger.info(f"Processed {processed_count}/{num_samples} samples...")
                
        except Exception as e:
            logger.error(f"Error reading {npz_path}: {e}")
            
    if processed_count == 0:
        logger.error("No samples were successfully processed.")
        return
        
    # Compute final means
    input_mean = input_sum / np.maximum(input_counts, 1.0)
    target_mean = target_sum / max(target_counts, 1.0)
    
    # Compute final standard deviations
    input_var = (input_sum_sq / np.maximum(input_counts, 1.0)) - (input_mean ** 2)
    # Prevent negative variance due to floating point precision
    input_var = np.maximum(input_var, 0.0)
    input_std = np.sqrt(input_var)
    # Prevent division by zero
    input_std[input_std == 0.0] = 1.0
    
    target_var = (target_sum_sq / max(target_counts, 1.0)) - (target_mean ** 2)
    target_var = max(target_var, 0.0)
    target_std = np.sqrt(target_var)
    if target_std == 0.0:
        target_std = 1.0
        
    # Build statistics dictionary
    stats = {
        "target_name": target_name,
        "log_transform_precip": log_transform_precip,
        "processed_samples": processed_count,
        "crop_indices": {
            "row_start": ROW_START,
            "row_end": ROW_END,
            "col_start": COL_START,
            "col_end": COL_END
        },
        "input_mean": input_mean.tolist(),
        "input_std": input_std.tolist(),
        "target_mean": float(target_mean),
        "target_std": float(target_std)
    }
    
    # Save statistics
    out_dir = os.path.join(Config.LOCAL_DATA_DIR, "metadata")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"norm_stats_{target_name}.json")
    
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=4)
        
    logger.info(f"Successfully computed and saved normalization statistics to: {out_path}")
    logger.info(f"Target mean: {stats['target_mean']:.4f} | Target std: {stats['target_std']:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Calculate Z-score normalization stats for training data.")
    parser.add_argument("--target", type=str, choices=["chirps", "oya"], required=True, help="Target dataset")
    parser.add_argument("--no-log", action="store_true", help="Disable log-transform on precipitation fields")
    args = parser.parse_args()
    
    compute_normalization_stats(args.target, log_transform_precip=(not args.no_log))

if __name__ == "__main__":
    main()
