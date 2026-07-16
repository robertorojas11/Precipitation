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

first = col.first()
band_names = first.bandNames().getInfo()

print(f"FULL BAND LIST ({len(band_names)} bands):")
print(band_names)
