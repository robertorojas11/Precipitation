import ee
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_extraction import gee_extractor

if not gee_extractor.initialize_gee():
    sys.exit(1)

d_str = "2004-04-17"
d_start = ee.Date(d_str)
d_end = d_start.advance(1, 'day')

era5 = (ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
        .filterDate(d_start, d_end)
        .filterBounds(gee_extractor.DOMAIN_POLYGON))

for b in gee_extractor.SURFACE_BANDS:
    print(f"Testing band: {b}...")
    try:
        # Try to compute the mean for this specific band
        daily_mean = era5.select([b]).mean()
        
        # Force computation by getting a sample pixel
        val = daily_mean.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=gee_extractor.DOMAIN_POLYGON.centroid(),
            scale=27750
        ).getInfo()
        print(f"  {b} OK: {val}")
    except Exception as e:
        print(f"  {b} FAILED: {e}")
