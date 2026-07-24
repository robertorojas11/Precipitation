"""PyTorch dataset class for precipitation downscaling.

Loads preprocessed NPZ files containing ERA5 inputs, physical model outputs,
and topography, and normalizes them dynamically based on Z-score statistics.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.config import Config

logger = Config.get_logger()

class PrecipDataset(Dataset):
    """Custom PyTorch Dataset for precipitation downscaling (Mexico domain)."""

    def __init__(self, target_name, split, transform=True, log_transform_precip=True):
        """Initializes the dataset.

        Args:
            target_name (str): 'chirps' or 'oya'.
            split (str): 'train', 'val', or 'test'.
            transform (bool): Whether to apply Z-score normalization.
            log_transform_precip (bool): Whether to apply log1p to precip variables.
        """
        self.target_name = target_name
        self.split = split
        self.transform = transform
        self.log_transform_precip = log_transform_precip
        
        # Load dataset index
        metadata_dir = os.path.join(Config.LOCAL_DATA_DIR, "metadata")
        index_path = os.path.join(metadata_dir, f"dataset_index_{target_name}.csv")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Dataset index file not found: {index_path}")
            
        df = pd.read_csv(index_path)
        # Filter by split and valid_flag
        self.records = df[(df['split'] == split) & (df['valid_flag'] == True)].copy()
        self.records = self.records.reset_index(drop=True)
        
        if len(self.records) == 0:
            logger.warning(f"PrecipDataset initialized with 0 samples for target: {target_name}, split: {split}")
            
        # Load normalization statistics
        stats_path = os.path.join(metadata_dir, f"norm_stats_{target_name}.json")
        if self.transform:
            if not os.path.exists(stats_path):
                raise FileNotFoundError(
                    f"Normalization stats file not found: {stats_path}. "
                    "Please run src/data_preprocessing/norm_calculator.py first."
                )
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                
            # Convert stats to PyTorch Tensors for easy broadcasting: shape (C, 1, 1)
            self.input_mean = torch.tensor(stats['input_mean'], dtype=torch.float32).view(-1, 1, 1)
            self.input_std = torch.tensor(stats['input_std'], dtype=torch.float32).view(-1, 1, 1)
            self.target_mean = torch.tensor(stats['target_mean'], dtype=torch.float32).view(-1, 1, 1)
            self.target_std = torch.tensor(stats['target_std'], dtype=torch.float32).view(-1, 1, 1)
            
            # Warn if user config contradicts pre-computed stats config
            if stats.get('log_transform_precip', True) != self.log_transform_precip:
                logger.warning(
                    f"Configured log_transform_precip={self.log_transform_precip} "
                    f"differs from stats file configuration ({stats.get('log_transform_precip')})."
                )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        """Loads a single preprocessed sample.

        Returns:
            tuple: (inputs, target)
                inputs: shape (21, H, W) normalized tensor
                target: shape (1, H, W) normalized tensor
        """
        row = self.records.iloc[idx]
        npz_path = row['npz_path']
        
        # Robust path replacement for migrated environments
        if not os.path.exists(npz_path) and npz_path.startswith("./data/"):
            npz_path = npz_path.replace("./data/processed", Config.PROCESSED_DATA_DIR).replace("./data/raw", Config.RAW_DATA_DIR)
            
        with np.load(npz_path) as data:
            inputs = data['inputs']     # (H, W, 18)
            target = data['target']     # (H, W, 1)
            upslope = data['upslope']   # (H, W, 1)
            spectral = data['spectral'] # (H, W, 1)
            elevation = data['elevation'] # (H, W, 1)
            
        # Concatenate inputs to 21 channels: (H, W, 21)
        inputs_all = np.concatenate([inputs, upslope, spectral, elevation], axis=-1)
        
        # Transpose to PyTorch shape layout: (C, H, W)
        inputs_t = torch.tensor(inputs_all, dtype=torch.float32).permute(2, 0, 1)
        target_t = torch.tensor(target, dtype=torch.float32).permute(2, 0, 1)
        
        # Apply transformation/normalization
        if self.transform:
            # 1. Log transform if active (precipitation values are in channel 0)
            if self.log_transform_precip:
                inputs_t[0] = torch.log1p(inputs_t[0])
                target_t[0] = torch.log1p(target_t[0])
                
            # 2. Z-score scale
            inputs_t = (inputs_t - self.input_mean) / self.input_std
            target_t = (target_t - self.target_mean) / self.target_std
            
        # Fill NaNs with 0.0 (e.g. masked ocean regions)
        inputs_t = torch.nan_to_num(inputs_t, nan=0.0)
        target_t = torch.nan_to_num(target_t, nan=0.0)
        
        return inputs_t, target_t
