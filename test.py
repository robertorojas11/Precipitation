import os
from src.utils.config import setup_config
from src.data_preprocessing.compute_norm_stats import compute_normalization_statistics
from src.data_preprocessing.pipeline import build_dataset, load_config

logger = setup_config(__name__)

def test_pipeline():
    """
    Test script to verify the Data Preprocessing Pipeline execution.
    It runs the compute_norm_stats first, then attempts to build the dataset.
    """
    logger.info("Starting Pipeline Test...")
    
    # 1. Configuration
    config = load_config("pipeline_config.yaml")
    local_data_dir = os.getenv("LOCAL_DATA_DIR", "./data/era5_oya_mesoamerica")
    output_file = os.path.join(local_data_dir, "norm_stats.json")
    
    # 2. Run normalization stats computation
    logger.info("Testing compute_norm_stats...")
    os.makedirs(local_data_dir, exist_ok=True)
    compute_normalization_statistics(local_data_dir, output_file)
    
    # 3. Test dataset building
    logger.info("Testing build_dataset...")
    try:
        train_ds = build_dataset("train", config)
        logger.info(f"Train dataset successfully built: {train_ds}")
    except Exception as e:
        logger.error(f"Failed to build dataset: {e}")
        
    logger.info("Pipeline Test Complete.")

if __name__ == "__main__":
    test_pipeline()
