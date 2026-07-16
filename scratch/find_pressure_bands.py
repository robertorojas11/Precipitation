import ee
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_extraction import gee_extractor

if not gee_extractor.initialize_gee():
    sys.exit(1)

# Check HOURLY collection instead of DAILY
col_id = "ECMWF/ERA5/HOURLY"
d_str = "2004-04-17"
d_start = ee.Date(d_str)
d_end = d_start.advance(1, 'day')

col = (ee.ImageCollection(col_id)
        .filterDate(d_start, d_end)
        .filterBounds(gee_extractor.DOMAIN_POLYGON))

first = col.first()
band_names = first.bandNames().getInfo()
print(f"Collection: {col_id}")
print(f"Available Bands: {band_names[:20]} ... (total {len(band_names)})")

print("\nChecking for pressure bands:")
for b in gee_extractor.PRESSURE_BANDS:
    exists = b in band_names
    print(f"  {b}: {exists}")
