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
    """Move a file to trash (Editors often can't permanently delete files they don't own)."""
    try:
        service.files().update(fileId=file_id, body={'trashed': True}).execute()
        logger.info(f"Moved to Drive trash: {file_name}")
    except Exception as e:
        logger.warning(f"Could not trash {file_name} (Owner permissions required). Please empty your Precipitation_Exports folder manually later. Error: {e}")


def sync_dataset(dataset):
    """Sync dataset files from Drive to local raw directory."""
    service = get_drive_service()
    
    # 1. Find main export folder
    main_folder_id = find_folder(service, Config.GEE_DRIVE_FOLDER)
    if not main_folder_id:
        logger.error(f"Main export folder '{Config.GEE_DRIVE_FOLDER}' not found in Drive.")
        return
        
    # 2. Get files inside the dataset prefix
    # We only want .tif files that match our naming convention: {dataset}_{YYYY-MM-DD}.tif
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
                # Try to delete, but stay silent if it fails (known permission issue)
                try:
                    service.files().update(fileId=file_id, body={'trashed': True}).execute()
                except:
                    if not warning_shown:
                        logger.warning(f"Note: Automatic cleanup failed due to Drive permissions. Please manually empty '{Config.GEE_DRIVE_FOLDER}' in your browser to save space.")
                        warning_shown = True
            else:
                # File is already local, try one last time to trash it quietly
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
