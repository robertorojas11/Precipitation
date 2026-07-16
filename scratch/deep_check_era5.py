import ee
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_extraction import gee_extractor

if not gee_extractor.initialize_gee():
    sys.exit(1)

dates = ["2004-04-17"]
requested_bands = gee_extractor.SURFACE_BANDS

for d_str in dates:
    d_start = ee.Date(d_str)
    d_end = d_start.advance(1, 'day')
    
    era5 = (ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
            .filterDate(d_start, d_end)
            .filterBounds(gee_extractor.DOMAIN_POLYGON))
    
    img_list = era5.toList(24)
    size = img_list.length().getInfo()
    
    print(f"Date: {d_str} | Images: {size}")
    
    for i in range(size):
        img = ee.Image(img_list.get(i))
        id_str = img.id().getInfo()
        b_names = img.bandNames().getInfo()
        missing = [b for b in requested_bands if b not in b_names]
        if missing:
            print(f"  Image {id_str} is MISSING BANDS: {missing}")
        else:
            # Try a small computation to see if it crashes
            try:
                _ = img.select(requested_bands).reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=gee_extractor.DOMAIN_POLYGON.centroid(),
                    scale=27750
                ).getInfo()
            except Exception as e:
                print(f"  Image {id_str} CRASHED during computation: {e}")
