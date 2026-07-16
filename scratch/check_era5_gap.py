import ee
import os
import sys

# Correct path for scratch folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_extraction import gee_extractor
from src.utils.config import Config

if not gee_extractor.initialize_gee():
    print("Failed to initialize GEE")
    sys.exit(1)

dates = ["2004-04-17", "2004-04-18", "2004-04-19", "2004-04-20"]

for d_str in dates:
    d_start = ee.Date(d_str)
    d_end = d_start.advance(1, 'day')
    
    era5 = (ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
            .filterDate(d_start, d_end)
            .filterBounds(gee_extractor.DOMAIN_POLYGON))
    
    try:
        count = era5.size().getInfo()
        print(f"Date: {d_str} | Image Count: {count}")
        
        if count > 0:
            # Check if bands are present
            first = era5.first()
            band_names = first.bandNames().getInfo()
            print(f"  Bands: {len(band_names)}")
            # Check if any SURFACE_BANDS are missing
            missing = [b for b in gee_extractor.SURFACE_BANDS if b not in band_names]
            if missing:
                print(f"  MISSING BANDS: {missing}")
    except Exception as e:
        print(f"Error checking {d_str}: {e}")
