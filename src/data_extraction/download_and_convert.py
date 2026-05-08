import os
import io
import numpy as np
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from dotenv import load_dotenv
from tqdm import tqdm
import rasterio
from src.utils.config import setup_config

# Load environment variables
logger = setup_config(__name__)

# Configuration
DRIVE_CREDENTIALS_PATH = os.getenv("GOOGLE_DRIVE_CREDENTIALS")
LOCAL_DATA_DIR = os.getenv("LOCAL_DATA_DIR", "./data/era5_oya_mesoamerica")
DRIVE_FOLDER_NAME = "era5_oya_mesoamerica_exports"

def get_drive_service():
    """Authenticates and returns the Google Drive API service."""
    creds = service_account.Credentials.from_service_account_file(
        DRIVE_CREDENTIALS_PATH, 
        scopes=['https://www.googleapis.com/auth/drive']
    )
    return build('drive', 'v3', credentials=creds)

def find_folder_id(service, folder_name):
    """Finds the ID of a folder by name."""
    query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    items = results.get('files', [])
    if not items:
        return None
    return items[0]['id']

def download_file(service, file_id, file_name, destination_path):
    """Downloads a file from Drive to a local path."""
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(destination_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    return destination_path

def convert_tif_to_npz(tif_path, npz_path):
    """Converts a GeoTIFF (14 bands) to the specific NPZ structure."""
    with rasterio.open(tif_path) as src:
        data = src.read()  # Shape (14, H, W)
        
        # Split into inputs (13 bands) and target (1 band)
        # Note: rasterio reads as (Bands, H, W), we need (H, W, Bands)
        data = np.moveaxis(data, 0, -1)
        
        inputs = data[:, :, :13]
        target = data[:, :, 13:14]
        
        # Save as NPZ
        np.savez_compressed(npz_path, inputs=inputs, target=target)
    
    return npz_path

def process_exports():
    """Main loop to download and convert files."""
    service = get_drive_service()
    folder_id = find_folder_id(service, DRIVE_FOLDER_NAME)
    
    if not folder_id:
        logger.error(f"Folder '{DRIVE_FOLDER_NAME}' not found in Google Drive.")
        return

    # List files in the folder
    query = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(q=query, fields='files(id, name)').execute()
    files = results.get('files', [])
    
    logger.info(f"Found {len(files)} files to process.")
    
    for f in tqdm(files):
        file_id = f['id']
        file_name = f['name']
        
        if not file_name.endswith('.tif'):
            continue
            
        # Parse split and date from name: mexico_{split}_{YYYYMMdd_HH}.tif
        parts = file_name.replace('.tif', '').split('_')
        if len(parts) < 3: continue
        
        split = parts[1]
        date_str = parts[2]
        year = date_str[:4]
        month = date_str[4:6]
        
        # Define local paths
        target_dir = os.path.join(LOCAL_DATA_DIR, split, year, month)
        os.makedirs(target_dir, exist_ok=True)
        
        temp_tif = os.path.join(target_dir, file_name)
        final_npz = os.path.join(target_dir, file_name.replace('.tif', '.npz'))
        
        # Download
        download_file(service, file_id, file_name, temp_tif)
        
        # Convert
        convert_tif_to_npz(temp_tif, final_npz)
        
        # Cleanup temp TIF
        os.remove(temp_tif)
        
        #Delete from Drive after success
        service.files().delete(fileId=file_id).execute()

if __name__ == "__main__":
    # process_exports()
    pass
