import ee
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_extraction import gee_extractor

if not gee_extractor.initialize_gee():
    sys.exit(1)

col_id = "ECMWF/ERA5/HOURLY"
d_str = "2004-04-17"
d_start = ee.Date(d_str)
d_end = d_start.advance(1, 'hour') # Just one hour

col = (ee.ImageCollection(col_id)
        .filterDate(d_start, d_end)
        .filterBounds(gee_extractor.DOMAIN_POLYGON))

size = col.size().getInfo()
print(f"Number of images in 1 hour: {size}")

if size > 0:
    first = col.first()
    print(f"Properties: {first.propertyNames().getInfo()}")
    # Check if there is a 'level' or 'pressure' property
    level = first.get('pressure').getInfo() if 'pressure' in first.propertyNames().getInfo() else 'N/A'
    print(f"Pressure Level Property: {level}")
    
    # List all bands in the first image
    print(f"Bands in first image: {first.bandNames().getInfo()}")
