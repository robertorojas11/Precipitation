import ee
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_extraction import gee_extractor

if not gee_extractor.initialize_gee():
    sys.exit(1)

col_id = "ECMWF/ERA5/HOURLY"
d_str = "2020-04-17" # Check a recent year
d_start = ee.Date(d_str)
d_end = d_start.advance(1, 'hour')

col = (ee.ImageCollection(col_id).filterDate(d_start, d_end))
first = col.first()
band_names = first.bandNames().getInfo()

temp_bands = [b for b in band_names if 'temperature' in b and 'hPa' in b]
print(f"Atmospheric temperature levels in {col_id} (2020):")
for b in temp_bands:
    print(f"  {b}")
