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

first = era5.first()
band_names = first.bandNames().getInfo()
print(f"Available Bands: {band_names}")

print("\nChecking requested bands:")
for b in gee_extractor.SURFACE_BANDS:
    exists = b in band_names
    print(f"  {b}: {exists}")
