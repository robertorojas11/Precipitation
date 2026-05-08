import ee
import os
import time
import shapefile
from datetime import datetime, timedelta
from src.utils.config import setup_config

# Load environment variables and configure logger
logger = setup_config(__name__)

# Configuration
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
DRIVE_FOLDER = "era5_oya_mesoamerica_exports"

# ERA5-Land Bands (13 total)
ERA5_BANDS = [
    'temperature_2m', 'dewpoint_temperature_2m', 'surface_pressure',
    'u_component_of_wind_10m', 'v_component_of_wind_10m',
    'total_precipitation_hourly', 'runoff_hourly', 'surface_runoff_hourly',
    'surface_solar_radiation_downwards_hourly', 'surface_net_solar_radiation_hourly',
    'surface_sensible_heat_flux_hourly', 'surface_latent_heat_flux_hourly',
    'volumetric_soil_water_layer_1'
]

# Target Dataset
OYA_COLLECTION = "projects/global-precipitation-nowcast/assets/global_estimation"

def initialize_gee():
    """Initializes Earth Engine with the specified project."""
    try:
        ee.Initialize(project=PROJECT_ID)
        logger.info(f"Earth Engine initialized with project: {PROJECT_ID}")
    except Exception as e:
        logger.error(f"Error initializing Earth Engine: {e}")
        logger.info("Make sure you have run 'earthengine authenticate' or set up a Service Account.")

def get_mesoamerica_geometry():
    """Returns the combined geometry from the Atlantico and Pacifico shapefiles."""
    polygons = []
    atlantico_path = os.path.join("data", "shape_files", "atlantico_shp", "atlantico_shp_grande.shp")
    pacifico_path = os.path.join("data", "shape_files", "pacifico_shp", "pacifico_shp_grande.shp")
    
    for path in [atlantico_path, pacifico_path]:
        sf = shapefile.Reader(path)
        for shape in sf.shapes():
            coords = [list(p) for p in shape.points]
            polygons.append([coords])
            
    return ee.Geometry.MultiPolygon(polygons)

def export_hourly_samples(start_date_str, end_date_str, split='train'):
    """
    Launches GEE export tasks for hourly paired data within a date range.
    To avoid thousands of tasks, we can export as 'ImageStacks' or 
    multi-band images for chunks of time (e.g., daily).
    """
    region = get_mesoamerica_geometry()
    
    # 1. Load Collections
    era5_col = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY") \
        .filterBounds(region) \
        .filterDate(start_date_str, end_date_str) \
        .select(ERA5_BANDS)
    
    oya_col = ee.ImageCollection(OYA_COLLECTION) \
        .filterBounds(region) \
        .filterDate(start_date_str, end_date_str) \
        .select(['precipitation'])

    # Get list of timestamps from ERA5 (as strings for the export names)
    timestamps = era5_col.aggregate_array('system:time_start').getInfo()
    
    logger.info(f"Found {len(timestamps)} hourly samples for {start_date_str} to {end_date_str}.")
    
    for ts in timestamps:
        date = ee.Date(ts)
        date_str = date.format('YYYYMMdd_HH').getInfo()
        
        # Filter both to this specific hour
        era5_img = era5_col.filterDate(date, date.advance(1, 'hour')).first()
        
        # Find matching Oya image (within 30 mins)
        oya_img = oya_col.filterDate(date.advance(-15, 'minutes'), date.advance(45, 'minutes')).first()
        
        # Check if both exist
        if era5_img and oya_img:
            # Combine into one multi-band image (14 bands total)
            combined = era5_img.addBands(oya_img.rename('target_precipitation'))
            
            # Export Task
            task_name = f"mesoamerica_{split}_{date_str}"
            task = ee.batch.Export.image.toDrive(
                image=combined,
                description=task_name,
                folder=DRIVE_FOLDER,
                fileNamePrefix=task_name,
                region=region.getInfo()['coordinates'],
                scale=5000, # 5km resolution
                crs='EPSG:4326',
                fileFormat='GeoTIFF' # GeoTIFF is easier to convert locally than TFRecord for simple arrays
            )
            task.start()
            logger.info(f"Started task: {task_name}")
            
            # To avoid flooding GEE with too many simultaneous tasks, 
            # you might want to add a small sleep or limit the range.
            # GEE has a limit on concurrent tasks (usually 20-40).
            time.sleep(0.5)

if __name__ == "__main__":
    initialize_gee()
    # Example: Export first day of 2016
    # export_hourly_samples('2016-01-01', '2016-01-02', split='train')
