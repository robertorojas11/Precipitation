import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.utils.logging import configure_logging

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
            cls._logger = configure_logging(name="precipitation")
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
