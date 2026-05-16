"""Google Drive management for GEE exports.

This module handles authentication with Google Drive, searching for exported
GeoTIFFs, downloading them to local storage, and cleaning up Drive storage.
"""

import os
import sys
import argparse
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.config import Config

logger = Config.get_logger()
SCOPES = ['https://www.googleapis.com/auth/drive']

def get_drive_service():
    """Authenticates and returns a Google Drive API service instance.

    Returns:
        googleapiclient.discovery.Resource: An authorized Drive API service.
    """
    creds = service_account.Credentials.from_service_account_file(
        Config.DRIVE_CREDENTIALS_FILE, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

def find_folder(service, folder_name):
    """Finds a Google Drive folder ID by its name.

    Args:
        service (googleapiclient.discovery.Resource): Drive API service.
        folder_name (str): The name of the folder to find.

    Returns:
        str or None: The folder ID if found, otherwise None.
    """
    query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='nextPageToken, files(id, name)').execute()
    items = results.get('files', [])
    return items[0]['id'] if items else None

def download_file(service, file_id, file_name, destination_path):
    """Downloads a file from Google Drive to a local path.

    Args:
        service (googleapiclient.discovery.Resource): Drive API service.
        file_id (str): The Drive file ID.
        file_name (str): The name of the file (for logging).
        destination_path (str): The local path where the file will be saved.
    """
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(destination_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    logger.info(f"Downloading {file_name}...")
    while not done:
        _, done = downloader.next_chunk()
    logger.info(f"Finished downloading: {destination_path}")
    
def delete_file(service, file_id, file_name):
    """Moves a file to the Google Drive trash.

    Args:
        service (googleapiclient.discovery.Resource): Drive API service.
        file_id (str): The Drive file ID.
        file_name (str): The name of the file (for logging).
    """
    try:
        service.files().update(fileId=file_id, body={'trashed': True}).execute()
        logger.info(f"Moved to Drive trash: {file_name}")
    except Exception as e:
        logger.warning(f"Could not trash {file_name}: {e}")


def sync_dataset(dataset):
    """Synchronizes a dataset by downloading matching files from Drive and cleaning up.

    Args:
        dataset (str): The dataset name (e.g., 'era5', 'chirps').
    """
    service = get_drive_service()
    
    main_folder_id = find_folder(service, Config.GEE_DRIVE_FOLDER)
    if not main_folder_id:
        logger.error(f"Main export folder '{Config.GEE_DRIVE_FOLDER}' not found in Drive.")
        return
        
    query = f"name contains '{dataset}' and name contains '.tif' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='nextPageToken, files(id, name)').execute()
    items = results.get('files', [])
    
    if not items:
        logger.info(f"No .tif files found for dataset {dataset} in Drive.")
        return
        
    logger.info(f"Found {len(items)} potential files for {dataset}.")
    warning_shown = False
    
    for item in items:
        file_id = item['id']
        full_name = item['name']
        file_name = os.path.basename(full_name)
        
        try:
            if dataset == "era5" and file_name.startswith("era5_pl_"):
                continue

            if not file_name.startswith(f"{dataset}_") or '-' not in file_name:
                continue
                
            parts = file_name.replace('.tif', '').split('_')
            if len(parts) < 2: continue
            date_part = parts[-1] 
            date_elements = date_part.split('-')
            if len(date_elements) < 3: continue
            
            year, month = date_elements[0], date_elements[1]
            dest_dir = os.path.join(Config.RAW_DATA_DIR, dataset, year, month)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, file_name)
            
            if not os.path.exists(dest_path):
                download_file(service, file_id, full_name, dest_path)
                try:
                    service.files().update(fileId=file_id, body={'trashed': True}).execute()
                except:
                    if not warning_shown:
                        logger.warning(f"Note: Automatic cleanup failed due to Drive permissions.")
                        warning_shown = True
            else:
                try:
                    service.files().update(fileId=file_id, body={'trashed': True}).execute()
                except:
                    pass 
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")




def main():
    parser = argparse.ArgumentParser(description="Download GEE exports from Google Drive")
    parser.add_argument("--dataset", type=str, choices=["era5", "chirps", "oya", "dem"], required=True)
    args = parser.parse_args()
    
    sync_dataset(args.dataset)

if __name__ == "__main__":
    main()
