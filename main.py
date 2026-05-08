from src.utils.config import setup_config

if __name__ == "__main__":
    logger = setup_config(__name__)
    logger.info("Starting the precipitation downscaling project")