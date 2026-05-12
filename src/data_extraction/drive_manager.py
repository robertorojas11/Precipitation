import os
import sys
import argparse
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.config import Config

logger = Config.get_logger()

SCOPES = ['https://www.googleapis.com/auth/drive']

def get_drive_service():
    """Authenticate and return Google Drive service."""
    creds = service_account.Credentials.from_service_account_file(
        Config.DRIVE_CREDENTIALS_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)
    return service

def find_folder(service, folder_name):
    """Find a folder ID by name."""
    query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='nextPageToken, files(id, name)').execute()
    items = results.get('files', [])
    if not items:
        return None
    return items[0]['id']

def download_file(service, file_id, file_name, destination_path):
    """Download a file from Google Drive."""
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(destination_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    logger.info(f"Downloading {file_name}...")
    while done is False:
        status, done = downloader.next_chunk()
        # if status:
        #     logger.info(f"Download {int(status.progress() * 100)}%.")
    logger.info(f"Finished downloading: {destination_path}")
    
def delete_file(service, file_id, file_name):
    """Delete a file from Google Drive."""
    try:
        service.files().delete(fileId=file_id).execute()
        logger.info(f"Deleted from Drive: {file_name}")
    except Exception as e:
        logger.error(f"Failed to delete {file_name} from Drive: {e}")

def sync_dataset(dataset):
    """Sync dataset files from Drive to local raw directory."""
    service = get_drive_service()
    
    # 1. Find main export folder
    main_folder_id = find_folder(service, Config.GEE_DRIVE_FOLDER)
    if not main_folder_id:
        logger.error(f"Main export folder '{Config.GEE_DRIVE_FOLDER}' not found in Drive.")
        return
        
    # 2. Get files inside the dataset prefix (Since GEE exports use prefix e.g., 'era5/2004/01/era5_2004-01-01.tif')
    # Drive API doesn't easily search by path, so we search by name prefix if possible, or list all files in the folder.
    # Note: GEE Export with fileNamePrefix creates nested folders in Drive if they don't exist.
    
    # For now, a simple search for files containing the dataset name
    query = f"name contains '{dataset}' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='nextPageToken, files(id, name)').execute()
    items = results.get('files', [])
    
    if not items:
        logger.info(f"No files found for dataset {dataset} in Drive.")
        return
        
    logger.info(f"Found {len(items)} files for {dataset}.")
    
    for item in items:
        file_id = item['id']
        file_name = item['name']
        
        # Determine year and month from filename (e.g. era5_2004-01-01.tif)
        try:
            date_str = file_name.split('_')[1].split('.')[0]
            year, month, _ = date_str.split('-')
            
            dest_dir = os.path.join(Config.RAW_DATA_DIR, dataset, year, month)
            os.makedirs(dest_dir, exist_ok=True)
            
            dest_path = os.path.join(dest_dir, file_name)
            
            if not os.path.exists(dest_path):
                download_file(service, file_id, file_name, dest_path)
                delete_file(service, file_id, file_name)
            else:
                logger.info(f"File already exists locally: {dest_path}")
                # We also want to delete it from Drive if it exists locally to clear space
                delete_file(service, file_id, file_name)
        except Exception as e:
            logger.error(f"Skipping {file_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download GEE exports from Google Drive")
    parser.add_argument("--dataset", type=str, choices=["era5", "chirps", "oya", "dem"], required=True)
    args = parser.parse_args()
    
    sync_dataset(args.dataset)

if __name__ == "__main__":
    main()
