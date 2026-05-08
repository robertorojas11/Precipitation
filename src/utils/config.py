import logging
import os
from dotenv import load_dotenv

def setup_config(log_name="downscaling_logger"):
    load_dotenv()   
    
    log_dir = "./data/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "downscaling.log")
    
    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(log_name)

def get_base_dir():
    """Get the base directory of the project"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))