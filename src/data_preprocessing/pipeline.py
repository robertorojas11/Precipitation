import os
import json
import yaml
import numpy as np
# import tensorflow as tf # Optional for now, assuming tf is installed or will be.

def load_config(config_path="pipeline_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_dataset(split, config):
    """
    Builds the tf.data.Dataset for the given split.
    
    STAKEHOLDER / PLACEHOLDER:
    - Missing `dataset_index.csv`: We need the index to filter valid files.
    - Missing `tf.data` full implementation: We mock the dataset return for testing.
    
    Expected behavior:
    1. Read dataset_index.csv for `split`.
    2. Read norm_stats.json and apply Z-score normalization to the 13 input bands.
    3. Patch the (1156, 3796) images into (128, 128) patches.
    4. Filter patches with <5% valid pixels.
    5. For training, filter fully dry patches and apply w_wet/w_dry sample weights.
    """
    print(f"Building dataset for split: {split}")
    
    # 1. Load normalization stats (will raise FileNotFoundError if not generated)
    local_data_dir = os.getenv("LOCAL_DATA_DIR", "./data/era5_oya_mexico")
    norm_path = os.path.join(local_data_dir, "norm_stats.json")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Missing {norm_path}. Run compute_norm_stats.py first.")
        
    with open(norm_path, 'r') as f:
        stats = json.load(f)
        
    # Placeholder: Return a mocked list instead of tf.data.Dataset
    batch_size = config.get("batch_size", 16)
    print(f"Dataset successfully initialized with batch size {batch_size}. Mocking tf.data.Dataset object...")
    
    return "MOCK_TF_DATASET"
