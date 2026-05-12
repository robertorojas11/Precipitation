import sys
import os
import ee

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.config import Config
import src.data_extraction.gee_extractor as gee_extractor

logger = Config.get_logger()

def test_1day_export():
    """Trigger a 1-day export for ERA5 to verify aggregation logic."""
    logger.info("Testing 1-day GEE export submission...")
    
    if not gee_extractor.initialize_gee():
        logger.error("Failed to initialize GEE.")
        return False
        
    date_str = "2004-01-01"
    d_start = ee.Date(date_str)
    d_end = d_start.advance(1, 'day')
    drive_folder = "Precipitation_Test_Exports"
    
    try:
        # Test ERA5
        era5 = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY") \
            .filterDate(d_start, d_end) \
            .filterBounds(gee_extractor.DOMAIN_POLYGON) \
            .select(gee_extractor.SURFACE_BANDS)
            
        daily_precip = era5.select(['total_precipitation_hourly']).sum()
        daily_others = era5.select(gee_extractor.OTHER_BANDS).mean()
        img = daily_precip.addBands(daily_others).set('system:time_start', d_start.millis()).clip(gee_extractor.DOMAIN_POLYGON)
        
        task = ee.batch.Export.image.toDrive(
            image=img,
            description=f"test_era5_{date_str}",
            folder=drive_folder,
            fileNamePrefix=f"test_era5/era5_{date_str}",
            region=gee_extractor.DOMAIN_POLYGON,
            scale=27750,
            crs='EPSG:4326',
            maxPixels=1e13
        )
        task.start()
        logger.info(f"SUCCESS: Submitted 1-day ERA5 export test task: {task.id}")
        return True
    except Exception as e:
        logger.error(f"FAILED to submit test task: {e}")
        return False

if __name__ == "__main__":
    success = test_1day_export()
    sys.exit(0 if success else 1)
