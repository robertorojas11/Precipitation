import ee
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_extraction import gee_extractor
from src.utils.config import Config

if not gee_extractor.initialize_gee():
    sys.exit(1)

d_str = "2004-04-17"
d_start = ee.Date(d_str)
d_end = d_start.advance(1, 'day')

era5 = (ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
        .filterDate(d_start, d_end)
        .filterBounds(gee_extractor.DOMAIN_POLYGON)
        .select(gee_extractor.SURFACE_BANDS))

daily_precip = era5.select(['total_precipitation_hourly']).sum()
daily_others = era5.select(gee_extractor.OTHER_BANDS).mean()
img = daily_precip.addBands(daily_others).clip(gee_extractor.DOMAIN_POLYGON)

task = ee.batch.Export.image.toDrive(
    image=img,
    description=f"test_fail_{d_str}",
    folder=Config.GEE_DRIVE_FOLDER,
    fileNamePrefix=f"test_fail_{d_str}",
    region=gee_extractor.DOMAIN_POLYGON,
    scale=27750, crs='EPSG:4326', maxPixels=1e13
)

task.start()
print(f"Started test export: {task.id}")

while True:
    status = ee.data.getTaskStatus(task.id)[0]
    state = status['state']
    print(f"Status: {state}")
    if state in ['COMPLETED', 'FAILED', 'CANCELLED']:
        if state == 'FAILED':
            print(f"ERROR MESSAGE: {status.get('error_message')}")
        break
    time.sleep(10)
