import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import logging

class Config:
    # Google Cloud / Earth Engine
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    DRIVE_CREDENTIALS_FILE = os.getenv("GOOGLE_DRIVE_CREDENTIALS")
    
    # Storage & Paths
    LOCAL_DATA_DIR = os.getenv("LOCAL_DATA_DIR", "/mnt/data-r2/RobertoRojas/downscaling/era5_oya_mexico")
    RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "/mnt/data-r2/RobertoRojas/downscaling/raw")
    PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "/mnt/data-r2/RobertoRojas/downscaling/processed")
    
    # Shapefiles
    ATLANTICO_SHP_PATH = os.getenv("ATLANTICO_SHP_PATH")
    PACIFICO_SHP_PATH = os.getenv("PACIFICO_SHP_PATH")
    
    # GEE Export Configuration
    GEE_DRIVE_FOLDER = os.getenv("GEE_DRIVE_FOLDER", "Precipitation_Exports")
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "precipitation-dowscaling-exports")

    _logger = None

    @classmethod
    def get_logger(cls):
        if cls._logger is None:
            logger = logging.getLogger("PrecipitationPipeline")
            logger.setLevel(logging.INFO)
            # Create console handler with formatting
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            # Ensure no duplicate handlers
            if not logger.handlers:
                logger.addHandler(ch)
            cls._logger = logger
        return cls._logger

    @classmethod
    def init_directories(cls):
        """Initializes all necessary data directories."""
        directories = [
            cls.LOCAL_DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            os.path.join(cls.RAW_DATA_DIR, "era5"),
            os.path.join(cls.RAW_DATA_DIR, "chirps"),
            os.path.join(cls.RAW_DATA_DIR, "oya"),
            os.path.join(cls.RAW_DATA_DIR, "dem")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Ensured directory exists: {directory}")
