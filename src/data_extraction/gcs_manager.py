import os
import sys
import argparse
from google.cloud import storage
from google.oauth2 import service_account

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.config import Config

logger = Config.get_logger()

def get_gcs_client():
    """Return an authenticated GCS client using the service account."""
    creds = service_account.Credentials.from_service_account_file(
        Config.SERVICE_ACCOUNT_FILE
    )
    return storage.Client(credentials=creds, project=Config.PROJECT_ID)

def sync_dataset(dataset):
    """Download all blobs for a dataset from GCS to local raw dir, then delete from GCS."""
    client = get_gcs_client()
    bucket = client.bucket(Config.GCS_BUCKET_NAME)
    prefix = f"{dataset}/"

    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        logger.info(f"No files found in GCS for dataset: {dataset}")
        return

    logger.info(f"Found {len(blobs)} files for {dataset} in GCS.")

    for blob in blobs:
        # blob.name = e.g. "era5/2004/01/era5_2004-01-01.tif"
        parts = blob.name.split("/")
        file_name = parts[-1]

        if not file_name.endswith(".tif"):
            continue

        try:
            # Parse date from filename: era5_2004-01-01.tif
            date_str = file_name.replace(f"{dataset}_", "").replace(".tif", "")
            year, month, _ = date_str.split("-")

            dest_dir = os.path.join(Config.RAW_DATA_DIR, dataset, year, month)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, file_name)

            if not os.path.exists(dest_path):
                logger.info(f"Downloading {blob.name} ...")
                blob.download_to_filename(dest_path)
                logger.info(f"Saved: {dest_path}")
            else:
                logger.info(f"Already exists locally: {dest_path}")

            # Delete from GCS to free space
            blob.delete()
            logger.info(f"Deleted from GCS: {blob.name}")

        except Exception as e:
            logger.error(f"Failed to process {blob.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download GEE exports from GCS")
    parser.add_argument("--dataset", type=str,
                        choices=["era5", "era5_pl", "chirps", "oya", "dem"],
                        required=True)
    args = parser.parse_args()
    sync_dataset(args.dataset)

if __name__ == "__main__":
    main()
