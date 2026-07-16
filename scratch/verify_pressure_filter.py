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
d_end = d_start.advance(1, 'hour')

# Filter for pressure = 1000
col_1000 = (ee.ImageCollection(col_id)
            .filterDate(d_start, d_end)
            .filter(ee.Filter.eq('pressure', 1000)))

size = col_1000.size().getInfo()
print(f"Images with pressure=1000 in 1 hour: {size}")

if size > 0:
    first = col_1000.first()
    print(f"Bands at 1000hPa: {first.bandNames().getInfo()}")
