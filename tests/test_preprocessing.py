"""Verification test script for Step 2 (Data Preprocessing & Physics Models).

Loads a single NPZ file from the dataset index, performs the preprocessing
and physical computations, validates shapes and values, computes normalization
stats, and verifies the custom PyTorch dataset loader.
"""

import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.config import Config
from src.data_preprocessing.physics_models import resample_dem, compute_terrain_gradients
from src.data_preprocessing.preprocess_pipeline import run_preprocessing_pipeline
from src.data_preprocessing.norm_calculator import compute_normalization_stats
from src.data_preprocessing.dataset import PrecipDataset

logger = Config.get_logger()

# Mexico bounding box shape: 460 x 720
HEIGHT, WIDTH = 460, 720

def test_preprocessing_end_to_end():
    logger.info("="*60)
    logger.info("RUNNING END-TO-END PREPROCESSING TEST")
    logger.info("="*60)
    
    # 1. Verify NASADEM elevation and gradient computation
    dem_path = os.path.join(Config.RAW_DATA_DIR, "dem", "nasadem_mexico_1km.tif")
    if not os.path.exists(dem_path):
        logger.error(f"FAILURE: NASADEM missing at {dem_path}")
        return False
        
    try:
        logger.info("Testing NASADEM resampling and gradients...")
        elevation = resample_dem(dem_path, HEIGHT, WIDTH)
        dz_dx, dz_dy = compute_terrain_gradients(elevation)
        
        assert elevation.shape == (HEIGHT, WIDTH), f"Expected DEM shape {(HEIGHT, WIDTH)}, got {elevation.shape}"
        assert dz_dx.shape == (HEIGHT, WIDTH), "dz_dx shape mismatch"
        assert dz_dy.shape == (HEIGHT, WIDTH), "dz_dy shape mismatch"
        logger.info("NASADEM and gradients computation: PASSED ✅")
    except Exception as e:
        logger.error(f"FAILURE: Terrain verification failed: {e}")
        return False
        
    # 2. Find a valid record in the index
    index_path = os.path.join(Config.LOCAL_DATA_DIR, "metadata", "dataset_index_chirps.csv")
    if not os.path.exists(index_path):
        logger.error(f"FAILURE: Dataset index missing at {index_path}")
        return False
        
    df = pd.read_csv(index_path)
    train_records = df[(df['split'] == 'train') & (df['valid_flag'] == True)]
    if train_records.empty:
        logger.error("FAILURE: No valid training records in the index to test with.")
        return False
        
    test_record = train_records.iloc[0]
    npz_path = test_record['npz_path']
    date_str = test_record['date']
    
    logger.info(f"Using test date: {date_str} (path: {npz_path})")
    
    # Create backup of the raw file
    backup_path = npz_path + ".backup"
    shutil.copyfile(npz_path, backup_path)
    logger.info("Created backup of raw NPZ file.")
    
    success = False
    try:
        # 3. Preprocess the single date
        # Temporarily update the index to only contain this test date so we can run the pipeline quickly
        temp_index_path = index_path + ".temp"
        shutil.copyfile(index_path, temp_index_path)
        
        df_single = df[df['date'] == date_str].copy()
        df_single.to_csv(index_path, index=False)
        
        logger.info("Running preprocess pipeline on test date...")
        run_preprocessing_pipeline("chirps", overwrite=True)
        
        # Verify preprocessed file content
        with np.load(npz_path) as data:
            assert 'inputs' in data, "inputs missing in NPZ"
            assert 'target' in data, "target missing in NPZ"
            assert 'upslope' in data, "upslope missing in NPZ"
            assert 'spectral' in data, "spectral missing in NPZ"
            assert 'elevation' in data, "elevation missing in NPZ"
            
            inputs = data['inputs']
            target = data['target']
            upslope = data['upslope']
            spectral = data['spectral']
            elevation_np = data['elevation']
            
            assert inputs.shape == (HEIGHT, WIDTH, 18), f"Inputs shape mismatch: {inputs.shape}"
            assert target.shape == (HEIGHT, WIDTH, 1), f"Target shape mismatch: {target.shape}"
            assert upslope.shape == (HEIGHT, WIDTH, 1), f"Upslope shape mismatch: {upslope.shape}"
            assert spectral.shape == (HEIGHT, WIDTH, 1), f"Spectral shape mismatch: {spectral.shape}"
            assert elevation_np.shape == (HEIGHT, WIDTH, 1), f"Elevation shape mismatch: {elevation_np.shape}"
            
            logger.info(f"Inputs min/max: {np.nanmin(inputs):.4f}/{np.nanmax(inputs):.4f}")
            logger.info(f"Target min/max: {np.nanmin(target):.4f}/{np.nanmax(target):.4f}")
            logger.info(f"Upslope min/max: {np.nanmin(upslope):.4f}/{np.nanmax(upslope):.4f}")
            logger.info(f"Spectral min/max: {np.nanmin(spectral):.4f}/{np.nanmax(spectral):.4f}")
            
        logger.info("Pipeline preprocessed file verification: PASSED ✅")
        
        # 4. Verify Z-score norm stats calculator
        logger.info("Computing Z-score statistics...")
        compute_normalization_stats("chirps", log_transform_precip=True)
        
        stats_path = os.path.join(Config.LOCAL_DATA_DIR, "metadata", "norm_stats_chirps.json")
        assert os.path.exists(stats_path), f"Stats file not saved at {stats_path}"
        
        with open(stats_path, 'r') as f:
            stats = json.load(f)
            assert len(stats['input_mean']) == 21, f"Expected 21 means, got {len(stats['input_mean'])}"
            assert len(stats['input_std']) == 21, f"Expected 21 stds, got {len(stats['input_std'])}"
            
        logger.info("Normalization stats calculation: PASSED ✅")
        
        # 5. Verify PyTorch custom dataset loader
        logger.info("Loading dataset via PrecipDataset...")
        dataset = PrecipDataset("chirps", split="train", transform=True, log_transform_precip=True)
        assert len(dataset) == 1, f"Expected dataset length 1, got {len(dataset)}"
        
        x, y = dataset[0]
        assert x.shape == (21, HEIGHT, WIDTH), f"Expected inputs shape (21, {HEIGHT}, {WIDTH}), got {x.shape}"
        assert y.shape == (1, HEIGHT, WIDTH), f"Expected target shape (1, {HEIGHT}, {WIDTH}), got {y.shape}"
        assert not torch.isnan(x).any(), "NaN values found in inputs tensor!"
        assert not torch.isnan(y).any(), "NaN values found in target tensor!"
        
        logger.info(f"Inputs Tensor mean: {x.mean().item():.4f} | std: {x.std().item():.4f}")
        logger.info(f"Target Tensor mean: {y.mean().item():.4f} | std: {y.std().item():.4f}")
        logger.info("PyTorch Dataset loader verification: PASSED ✅")
        
        success = True
        
    except Exception as e:
        logger.error(f"FAILURE during verification: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Restore backup and temporary index
        if os.path.exists(backup_path):
            shutil.copyfile(backup_path, npz_path)
            os.remove(backup_path)
            
        if os.path.exists(temp_index_path):
            shutil.copyfile(temp_index_path, index_path)
            os.remove(temp_index_path)
            
        logger.info("Cleaned up backup files and restored indices.")
        
    return success

if __name__ == "__main__":
    passed = test_preprocessing_end_to_end()
    sys.exit(0 if passed else 1)
