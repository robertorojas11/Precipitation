import ee
import os
import sys

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import Config

logger = Config.get_logger()

def test_gee_auth():
    """Test Earth Engine initialization using service account."""
    logger.info("Testing Earth Engine Authentication...")
    
    try:
        # Get credentials path from config
        creds_path = Config.SERVICE_ACCOUNT_FILE
        project_id = Config.PROJECT_ID
        
        if not creds_path or not os.path.exists(creds_path):
            logger.error(f"Credentials file not found at {creds_path}")
            return False
            
        logger.info(f"Using credentials file: {creds_path}")
        logger.info(f"Using Project ID: {project_id}")
        
        # Initialize EE using the service account credentials directly via ee.Credentials
        credentials = ee.ServiceAccountCredentials('', creds_path)
        ee.Initialize(credentials, project=project_id)
        
        # Perform a simple test
        logger.info("Testing basic computation...")
        num = ee.Number(1).add(1).getInfo()
        if num == 2:
            logger.info("SUCCESS: Earth Engine initialized correctly!")
            return True
        else:
            logger.error(f"FAILED: Unexpected result from computation: {num}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to initialize Earth Engine: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_gee_auth()
    sys.exit(0 if success else 1)
