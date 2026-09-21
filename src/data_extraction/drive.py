"""Download completed Earth Engine exports from Google Drive."""

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


def parse_prefixed_export_name(name: str, name_prefix: str) -> tuple[str, str, str, str]:
    """Parse a flat ``"<name_prefix>_<region>_<YYYY-MM-DD>.tif"`` Drive
    filename into ``(region_name, date_str, year, month)``.

    Pure-Python helper factored out of :func:`sync_prefixed` so the parsing
    logic is unit-testable without hitting the Drive API. The date is always
    the trailing ``YYYY-MM-DD`` segment; the region name is whatever sits
    between the prefix and the date (region names may themselves contain
    underscores, e.g. ``mexico_pacifico``, so a plain ``str.split("_")`` is
    not enough).
    """
    stem = name.removesuffix(".tif")
    head = f"{name_prefix}_"
    if not stem.startswith(head):
        raise ValueError(f"{name!r} does not start with prefix {name_prefix!r}")
    date_str = stem[-10:]  # "YYYY-MM-DD" is always exactly 10 characters
    region_name = stem[len(head): -11]  # strip "<prefix>_" and "_<date>"
    if not region_name:
        raise ValueError(f"{name!r} has no region segment between prefix and date")
    year, month, _ = date_str.split("-")
    return region_name, date_str, year, month


def sync_prefixed(destination_root: Path | str, *, name_prefix: str, subdir: str) -> int:
    """Download completed Earth Engine exports named
    ``"<name_prefix>_<region>_<YYYY-MM-DD>.tif"`` (flat Drive filename, no
    embedded "/" -- see :func:`parse_prefixed_export_name`), writing each to
    ``destination_root/subdir/<region>/<year>/<month>/<name>`` and trashing
    the Drive copy after a successful download. Generalizes :func:`sync_oya`
    for any region-scoped export family (``era5``, ``imerg``, region-scoped
    ``oya``).
    """
    service = _service()
    parent = _folder_id(service, Config.GEE_DRIVE_FOLDER)
    query = (
        f"'{parent}' in parents and name contains '{name_prefix}_' "
        "and name contains '.tif' and trashed=false"
    )
    response = execute_with_retry(
        service.files().list(q=query, spaces="drive", fields="files(id,name)")
    )
    downloaded = 0
    for item in response.get("files", []):
        name = Path(item["name"]).name
        region_name, _date_str, year, month = parse_prefixed_export_name(name, name_prefix)
        destination = Path(destination_root) / subdir / region_name / year / month / name
        if not destination.exists():
            _download(service, item["id"], destination)
            downloaded += 1
        execute_with_retry(
            service.files().update(fileId=item["id"], body={"trashed": True})
        )
    return downloaded
