import ee
import os
import argparse
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.config import Config

logger = Config.get_logger()

# Constants
DOMAIN_POLYGON = None # Initialized after ee.Initialize()

SURFACE_BANDS = [
    'total_precipitation_hourly', 'temperature_2m', 'dewpoint_temperature_2m',
    'surface_pressure', 'u_component_of_wind_10m', 'v_component_of_wind_10m',
    'surface_solar_radiation_downwards_hourly', 'surface_sensible_heat_flux_hourly',
    'surface_latent_heat_flux_hourly', 'volumetric_soil_water_layer_1'
]

OTHER_BANDS = [b for b in SURFACE_BANDS if b != 'total_precipitation_hourly']

def initialize_gee():
    """Initialize Google Earth Engine."""
    try:
        credentials = ee.ServiceAccountCredentials('', Config.SERVICE_ACCOUNT_FILE)
        ee.Initialize(credentials, project=Config.PROJECT_ID)
        global DOMAIN_POLYGON
        DOMAIN_POLYGON = ee.Geometry.Polygon([[
            [-133.471, 18.626],  # NW  -- open Pacific
            [-124.199, 33.753],  # N   -- Baja California North
            [-61.429,  32.525],  # NE  -- Caribbean North
            [-38.653,  29.721],  # E   -- open Atlantic North
            [-38.653,  18.490],  # E   -- open Atlantic Mid
            [-38.689,   5.288],  # SE  -- Atlantic equatorial
            [-53.015,   5.069],  # S   -- Guyana coast
            [-80.666,   4.970],  # S   -- Panama/Colombia
            [-117.667,  5.101],  # SW  -- Eastern Pacific equatorial
        ]])
        return True
    except Exception as e:
        logger.error(f"Error initializing GEE: {e}")
        return False

def export_era5(year, month, drive_folder):
    """Export ERA5 data for a given month."""
    start_date = f"{year}-{month:02d}-01"
    end_date = (datetime.strptime(start_date, "%Y-%m-%d") + relativedelta(months=1)).strftime("%Y-%m-%d")
    
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY") \
        .filterDate(start_date, end_date) \
        .filterBounds(DOMAIN_POLYGON) \
        .select(SURFACE_BANDS)

    def aggregate_daily(date_str):
        d_start = ee.Date(date_str)
        d_end = d_start.advance(1, 'day')
        daily_imgs = era5.filterDate(d_start, d_end)
        
        # Sum precipitation, mean for all others
        daily_precip = daily_imgs.select(['total_precipitation_hourly']).sum()
        daily_others = daily_imgs.select(OTHER_BANDS).mean()
        return daily_precip.addBands(daily_others).set('system:time_start', d_start.millis())
        
    # Get all days in month
    days_in_month = (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days
    dates = [ee.Date(start_date).advance(d, 'day').format('YYYY-MM-dd') for d in range(days_in_month)]
    
    daily_collection = ee.ImageCollection(ee.List(dates).map(aggregate_daily))
    
    # We export as a single multiband image or collection of images. For simplicity, we can export
    # day by day to avoid massive files, or export a single multiband per month.
    # The plan suggests downloading "era5_YYYY-MM-DD.tif", so we create a task per day.
    tasks = []
    
    # Need to fetch the list of dates to iterate locally
    local_dates = ee.List(dates).getInfo()
    for d_str in local_dates:
        img = aggregate_daily(d_str).clip(DOMAIN_POLYGON)
        task_name = f"export_era5_{d_str}"
        file_name = f"era5_{d_str}"
        task = ee.batch.Export.image.toDrive(
            image=img,
            description=task_name,
            folder=drive_folder,
            fileNamePrefix=f"era5/{year}/{month:02d}/{file_name}",
            region=DOMAIN_POLYGON,
            scale=27750, # approx 0.25 deg in meters
            crs='EPSG:4326',
            maxPixels=1e13
        )
        task.start()
        tasks.append(task)
        logger.info(f"Started ERA5 export for {d_str}: {task.id}")
        
    return tasks

def export_chirps(year, month, drive_folder):
    """Export CHIRPS data for a given month."""
    start_date = f"{year}-{month:02d}-01"
    end_date = (datetime.strptime(start_date, "%Y-%m-%d") + relativedelta(months=1)).strftime("%Y-%m-%d")
    
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .filterDate(start_date, end_date) \
        .filterBounds(DOMAIN_POLYGON) \
        .select('precipitation')
        
    # CHIRPS is already daily. Just map and export.
    def prep_img(img):
        # Mask pixels < 0
        img = img.updateMask(img.gte(0))
        return img.clip(DOMAIN_POLYGON)
        
    processed = chirps.map(prep_img)
    
    days_in_month = (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days
    dates = [ee.Date(start_date).advance(d, 'day').format('YYYY-MM-dd') for d in range(days_in_month)]
    local_dates = ee.List(dates).getInfo()
    
    tasks = []
    for d_str in local_dates:
        d_start = ee.Date(d_str)
        d_end = d_start.advance(1, 'day')
        img_list = processed.filterDate(d_start, d_end).toList(1)
        # Check if image exists for that day
        size = img_list.length().getInfo()
        if size > 0:
            img = ee.Image(img_list.get(0))
            task_name = f"export_chirps_{d_str}"
            file_name = f"chirps_{d_str}"
            task = ee.batch.Export.image.toDrive(
                image=img,
                description=task_name,
                folder=drive_folder,
                fileNamePrefix=f"chirps/{year}/{month:02d}/{file_name}",
                region=DOMAIN_POLYGON,
                scale=5566, # approx 0.05 deg in meters
                crs='EPSG:4326',
                maxPixels=1e13
            )
            task.start()
            tasks.append(task)
            logger.info(f"Started CHIRPS export for {d_str}: {task.id}")
            
    return tasks

def export_oya(year, month, drive_folder):
    """Export Oya data for a given month."""
    start_date = f"{year}-{month:02d}-01"
    end_date = (datetime.strptime(start_date, "%Y-%m-%d") + relativedelta(months=1)).strftime("%Y-%m-%d")
    
    oya = ee.ImageCollection("projects/global-precipitation-nowcast/assets/global_estimation") \
        .filterDate(start_date, end_date) \
        .filterBounds(DOMAIN_POLYGON) \
        .select(['precipitation'])
        
    def aggregate_daily(date_str):
        d_start = ee.Date(date_str)
        d_end = d_start.advance(1, 'day')
        daily_imgs = oya.filterDate(d_start, d_end)
        
        # mm/hr × 0.5hr per image = mm per 30-min slot; sum gives mm/day
        daily_oya = daily_imgs.sum().multiply(0.5)
        return daily_oya.set('system:time_start', d_start.millis())
        
    days_in_month = (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days
    dates = [ee.Date(start_date).advance(d, 'day').format('YYYY-MM-dd') for d in range(days_in_month)]
    local_dates = ee.List(dates).getInfo()
    
    tasks = []
    for d_str in local_dates:
        img = aggregate_daily(d_str).clip(DOMAIN_POLYGON)
        task_name = f"export_oya_{d_str}"
        file_name = f"oya_{d_str}"
        task = ee.batch.Export.image.toDrive(
            image=img,
            description=task_name,
            folder=drive_folder,
            fileNamePrefix=f"oya/{year}/{month:02d}/{file_name}",
            region=DOMAIN_POLYGON,
            scale=5566, # approx 0.05 deg in meters
            crs='EPSG:4326',
            maxPixels=1e13
        )
        task.start()
        tasks.append(task)
        logger.info(f"Started Oya export for {d_str}: {task.id}")
            
    return tasks

def export_dem(drive_folder):
    """Export NASADEM data."""
    dem = ee.Image("NASA/NASADEM_HGT/001").select('elevation').clip(DOMAIN_POLYGON)
    
    task_name = "export_nasadem"
    file_name = "nasadem_mexico_1km"
    task = ee.batch.Export.image.toDrive(
        image=dem,
        description=task_name,
        folder=drive_folder,
        fileNamePrefix=f"dem/{file_name}",
        region=DOMAIN_POLYGON,
        scale=1000, # resampled to 1km
        crs='EPSG:4326',
        maxPixels=1e13
    )
    task.start()
    logger.info(f"Started DEM export: {task.id}")
    return [task]

def main():
    parser = argparse.ArgumentParser(description="GEE Extractor for Precipitation Downscaling")
    parser.add_argument("--dataset", type=str, choices=["era5", "chirps", "oya", "dem"], required=True)
    parser.add_argument("--start", type=str, help="Start month YYYY-MM")
    parser.add_argument("--end", type=str, help="End month YYYY-MM")
    
    args = parser.parse_args()
    
    if not initialize_gee():
        sys.exit(1)
        
    drive_folder = Config.GEE_DRIVE_FOLDER
        
    if args.dataset == "dem":
        export_dem(drive_folder)
        sys.exit(0)
        
    if not args.start or not args.end:
        logger.error("--start and --end are required for timeseries datasets.")
        sys.exit(1)
        
    start_dt = datetime.strptime(args.start, "%Y-%m")
    end_dt = datetime.strptime(args.end, "%Y-%m")
    
    current_dt = start_dt
    while current_dt <= end_dt:
        year = current_dt.year
        month = current_dt.month
        
        if args.dataset == "era5":
            export_era5(year, month, drive_folder)
        elif args.dataset == "chirps":
            export_chirps(year, month, drive_folder)
        elif args.dataset == "oya":
            export_oya(year, month, drive_folder)
            
        current_dt += relativedelta(months=1)

if __name__ == "__main__":
    main()
