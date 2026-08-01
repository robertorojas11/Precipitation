"""Download completed Oya exports from Google Drive."""

from __future__ import annotations

from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from src.utils.config import Config
from src.utils.retries import execute_with_retry, retry_on_network_error


DRIVE_SCOPE = ["https://www.googleapis.com/auth/drive"]


def _service():
    if not Config.DRIVE_CREDENTIALS_FILE:
        raise RuntimeError("GOOGLE_DRIVE_CREDENTIALS is not configured")
    credentials = service_account.Credentials.from_service_account_file(
        Config.DRIVE_CREDENTIALS_FILE,
        scopes=DRIVE_SCOPE,
    )
    return build("drive", "v3", credentials=credentials)


def _folder_id(service, folder_name: str) -> str:
    escaped_name = folder_name.replace("'", "\\'")
    query = (
        "mimeType='application/vnd.google-apps.folder' "
        f"and name='{escaped_name}' and trashed=false"
    )
    response = execute_with_retry(
        service.files().list(q=query, spaces="drive", fields="files(id,name)")
    )
    folders = response.get("files", [])
    if not folders:
        raise FileNotFoundError(f"Google Drive folder not found: {folder_name}")
    return folders[0]["id"]


@retry_on_network_error()
def _download(service, file_id: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        downloader = MediaIoBaseDownload(
            stream,
            service.files().get_media(fileId=file_id),
        )
        complete = False
        while not complete:
            _, complete = downloader.next_chunk()


def sync_oya(destination_root: Path | str) -> int:
    """Download Oya GeoTIFFs and trash Drive copies after successful writes."""
    service = _service()
    parent = _folder_id(service, Config.GEE_DRIVE_FOLDER)
    query = (
        f"'{parent}' in parents and name contains 'oya_' "
        "and name contains '.tif' and trashed=false"
    )
    response = execute_with_retry(
        service.files().list(
            q=query,
            spaces="drive",
            fields="files(id,name)",
        )
    )
    downloaded = 0
    for item in response.get("files", []):
        name = Path(item["name"]).name
        date = name.removesuffix(".tif").split("_")[-1]
        year, month, _ = date.split("-")
        destination = Path(destination_root) / "oya" / year / month / name
        if not destination.exists():
            _download(service, item["id"], destination)
            downloaded += 1
        execute_with_retry(
            service.files().update(fileId=item["id"], body={"trashed": True})
        )
    return downloaded
