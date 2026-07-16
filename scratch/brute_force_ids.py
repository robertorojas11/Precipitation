import ee
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_extraction import gee_extractor

if not gee_extractor.initialize_gee():
    sys.exit(1)

# Try to list all collections in the ECMWF/ERA5 path
# We can't use listAssets on public folders directly, but we can try common variations
variations = [
    "ECMWF/ERA5/HOURLY",
    "ECMWF/ERA5/DAILY",
    "ECMWF/ERA5/MONTHLY",
    "ECMWF/ERA5/HOURLY_SURFACE",
    "ECMWF/ERA5/HOURLY_PRESSURE",
    "ECMWF/ERA5/HOURLY_ATMOSPHERIC",
    "ECMWF/ERA5_LAND/HOURLY",
    "ECMWF/ERA5/PRELIMINARY/HOURLY"
]

print("Checking ERA5 IDs...")
for v in variations:
    try:
        ee.ImageCollection(v).limit(1).size().getInfo()
        print(f"  EXISTS: {v}")
    except:
        pass
