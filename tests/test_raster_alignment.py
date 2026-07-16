import os
import sys
import rasterio

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.config import Config

logger = Config.get_logger()

def test_alignment(era5_path, target_path):
    """Verify spatial overlap and CRS between two rasters."""
    logger.info(f"Testing alignment between:\n  1. {era5_path}\n  2. {target_path}")
    
    if not os.path.exists(era5_path):
        logger.error(f"ERA5 file missing: {era5_path}")
        return False
        
    if not os.path.exists(target_path):
        logger.error(f"Target file missing: {target_path}")
        return False
        
    try:
        with rasterio.open(era5_path) as src1, rasterio.open(target_path) as src2:
            crs1 = src1.crs
            crs2 = src2.crs
            
            bounds1 = src1.bounds
            bounds2 = src2.bounds
            
            logger.info(f"ERA5 CRS: {crs1} | Target CRS: {crs2}")
            logger.info(f"ERA5 Bounds: {bounds1}")
            logger.info(f"Target Bounds: {bounds2}")
            
            if crs1 != crs2:
                logger.error("CRS Mismatch!")
                return False
                
            # Verify overlap (Target bounds should be within or exactly match ERA5 bounds)
            # Or at least intersect. Since they are clipped to the same DOMAIN_POLYGON, 
            # they should have very similar bounds.
            if (bounds1.left > bounds2.right or bounds1.right < bounds2.left or
                bounds1.bottom > bounds2.top or bounds1.top < bounds2.bottom):
                logger.error("Rasters do not intersect!")
                return False
                
            logger.info("SUCCESS: Rasters intersect and share CRS.")
            return True
            
    except Exception as e:
        logger.error(f"Error during alignment test: {e}")
        return False

if __name__ == "__main__":
    # Placeholder paths - replace with actual downloaded files for real testing
    # E.g., python tests/test_raster_alignment.py /mnt/data-r2/RobertoRojas/downscaling/raw/era5/2004/01/era5_2004-01-01.tif /mnt/data-r2/RobertoRojas/downscaling/raw/chirps/2004/01/chirps_2004-01-01.tif
    if len(sys.argv) == 3:
        success = test_alignment(sys.argv[1], sys.argv[2])
    else:
        logger.info("Usage: python test_raster_alignment.py <era5_tif> <target_tif>")
        success = True # Just display usage if no args provided
        
    sys.exit(0 if success else 1)
