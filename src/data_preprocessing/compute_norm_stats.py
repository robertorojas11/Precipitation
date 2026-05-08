import os
import json
import numpy as np
from src.utils.config import setup_config

logger = setup_config(__name__)

def compute_normalization_statistics(data_dir: str, output_path: str):
    """
    Computes mean and standard deviation for the 13 ERA5-Land bands.
    
    STAKEHOLDER / PLACEHOLDER:
    Currently, the dataset is not fully downloaded, and `dataset_index.csv` 
    is missing. This script iterates over `.npz` files in the data_dir.
    In a fully functional pipeline, it should only compute stats on the 'train' split 
    (2004-2016) based on `dataset_index.csv`.
    """
    logger.info(f"Computing normalization statistics over {data_dir}...")
    
    # Placeholder: Assuming we have a mock structure for now.
    # In production, we iterate over train set npz files, load them, and aggregate mean/std.
    
    # Mock statistics
    mock_mean = [0.0] * 13
    mock_std = [1.0] * 13
    
    stats = {
        "mean": mock_mean,
        "std": mock_std
    }
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)
        
    logger.info(f"Saved mock normalization stats to {output_path}")

if __name__ == "__main__":
    local_data_dir = os.getenv("LOCAL_DATA_DIR", "./data/era5_oya_mexico")
    output_file = os.path.join(local_data_dir, "norm_stats.json")
    os.makedirs(local_data_dir, exist_ok=True)
    compute_normalization_statistics(local_data_dir, output_file)
