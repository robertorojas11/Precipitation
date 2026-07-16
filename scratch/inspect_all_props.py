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

col = (ee.ImageCollection(col_id)
        .filterDate(d_start, d_end))

# Get the first image regardless of bounds
first = col.first()
props = first.propertyNames().getInfo()
print(f"Properties of an image in {col_id}:")
for p in props:
    print(f"  {p}: {first.get(p).getInfo()}")

# Check for 'level' or 'pressure_level'
for p in ['level', 'pressure_level', 'pressure']:
    count = col.filter(ee.Filter.notNull([p])).size().getInfo()
    print(f"Images with property '{p}': {count}")
