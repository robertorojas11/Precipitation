import ee
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_extraction import gee_extractor

if not gee_extractor.initialize_gee():
    sys.exit(1)

# List assets in the ECMWF folder
try:
    assets = ee.data.listAssets({'parent': 'projects/ecmwf/assets'})
    for a in assets['assets']:
        print(f"Asset: {a['id']}")
except:
    # If projects/ecmwf/assets doesn't work, try a broad search
    print("Could not list projects/ecmwf/assets directly.")

# Alternative: list common ERA5 IDs
common_ids = [
    "ECMWF/ERA5/DAILY",
    "ECMWF/ERA5/HOURLY",
    "ECMWF/ERA5_LAND/HOURLY",
    "ECMWF/ERA5_LAND/DAILY_AGGR",
    "ECMWF/ERA5/PRESSURE_LEVELS", # Guessing
    "ECMWF/ERA5/ATMOSPHERIC_LEVELS" # Guessing
]

for cid in common_ids:
    try:
        ee.ImageCollection(cid).first().getInfo()
        print(f"ID exists: {cid}")
    except:
        pass
