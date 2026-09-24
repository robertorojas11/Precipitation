"""Download completed Earth Engine exports from Google Drive."""

from __future__ import annotations

from pathlib import Path
import re

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from src.data_preprocessing import regions as region_defs
from src.utils.config import Config
from src.utils.retries import execute_with_retry, retry_on_network_error


DRIVE_SCOPE = ["https://www.googleapis.com/auth/drive"]
logger = Config.get_logger()


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


def _try_trash(service, file_id: str, name: str) -> None:
    """Best-effort cleanup: move a synced export to the Drive trash.

    Exports authenticated via personal Google OAuth (``earth_engine.py``'s
    preferred path) land in the *user's* Drive, while the sync step here
    authenticates as the service account (``GOOGLE_DRIVE_CREDENTIALS``) --
    two different identities. If the export folder is only shared with the
    service account as a Viewer (read access, enough to list and download),
    trashing raises HTTP 403 even though the download already succeeded.
    That must not lose the fact that the file was retrieved, so this is
    swallowed with a warning rather than propagated -- an untrashed export
    left behind in Drive is a nuisance, not a data-loss risk.

    Caught broadly (not just ``HttpError``): found in production,
    2026-09-23, a transient read timeout (``TimeoutError``, not an
    ``HttpError``) surfaced here after ``execute_with_retry`` exhausted its
    8 attempts and killed an otherwise-healthy multi-day acquisition
    process, losing hours of unrelated progress over one file's cleanup
    step. Nothing this function does is worth crashing the caller over.
    """
    try:
        execute_with_retry(service.files().update(fileId=file_id, body={"trashed": True}))
    except Exception as error:
        logger.warning(
            "Could not trash Drive export %r (file_id=%s): %s. Downloaded copy is safe; "
            "the file remains in Drive -- share the export folder with the service "
            "account as Editor (not just Viewer) to enable automatic cleanup.",
            name, file_id, error,
        )


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


def _try_download(service, file_id: str, destination: Path, name: str) -> bool:
    """Best-effort download: same rationale as :func:`_try_trash`.

    ``_download`` is already wrapped in ``@retry_on_network_error()``, so by
    the time an exception reaches here every transient retry has already
    been exhausted -- any further failure (e.g. a sustained read timeout)
    should be logged and skipped, not allowed to crash a multi-day sync
    over one file. The file simply stays missing locally and is retried on
    the next ``sync()`` pass, since it still matches the query
    (``trashed=false``, and trashing this project's own copy never
    succeeds anyway -- see :func:`_try_trash`).
    """
    try:
        _download(service, file_id, destination)
        return True
    except Exception as error:
        logger.warning(
            "Could not download Drive export %r (file_id=%s): %s. Will retry on the "
            "next sync.", name, file_id, error,
        )
        return False


def _list_all(service, query: str) -> list[dict]:
    """Page through ``files().list()`` and return every match.

    A single ``.list()`` call only returns one page (Drive v3's default page
    size is 100); without following ``nextPageToken``, any query matching
    more than a page's worth of files would silently only ever see a subset
    of them. Found in production, 2026-09-21, once acquisition ran long
    enough to accumulate hundreds of matching exports in one Drive folder.
    """
    items: list[dict] = []
    page_token = None
    while True:
        response = execute_with_retry(
            service.files().list(
                q=query, spaces="drive", fields="nextPageToken, files(id,name)",
                pageSize=1000, pageToken=page_token,
            )
        )
        items.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            return items


def sync_oya(destination_root: Path | str) -> int:
    """Download Oya GeoTIFFs and trash Drive copies after successful writes."""
    service = _service()
    parent = _folder_id(service, Config.GEE_DRIVE_FOLDER)
    query = (
        f"'{parent}' in parents and name contains 'oya_' "
        "and name contains '.tif' and trashed=false"
    )
    downloaded = 0
    for item in _list_all(service, query):
        name = Path(item["name"]).name
        date = name.removesuffix(".tif").split("_")[-1]
        year, month, _ = date.split("-")
        destination = Path(destination_root) / "oya" / year / month / name
        if destination.exists():
            _try_trash(service, item["id"], name)
        elif _try_download(service, item["id"], destination, name):
            downloaded += 1
            _try_trash(service, item["id"], name)
    return downloaded


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_prefixed_export_name(name: str, name_prefix: str) -> tuple[str, str, str, str]:
    """Parse a flat ``"<name_prefix>_<region>_<YYYY-MM-DD>.tif"`` Drive
    filename into ``(region_name, date_str, year, month)``.

    Pure-Python helper factored out of :func:`sync_prefixed` so the parsing
    logic is unit-testable without hitting the Drive API. The date is always
    the trailing ``YYYY-MM-DD`` segment; the region name is whatever sits
    between the prefix and the date (region names may themselves contain
    underscores, e.g. ``mexico_pacifico``, so a plain ``str.split("_")`` is
    not enough).

    The Drive API's ``name contains '<name_prefix>_'`` query is a plain
    substring match with no anchoring, so it also matches an unrelated file
    like ``era5_pl_2020-01-01.tif`` when ``name_prefix="era5"`` (found in
    production, 2026-09-21: a legacy/unrelated export sharing the same Drive
    folder). Requiring the parsed region to be one of this project's
    registered regions (``src.data_preprocessing.regions``) rejects that
    kind of accidental collision instead of silently mis-filing it.
    """
    stem = name.removesuffix(".tif")
    head = f"{name_prefix}_"
    if not stem.startswith(head):
        raise ValueError(f"{name!r} does not start with prefix {name_prefix!r}")
    date_str = stem[-10:]  # "YYYY-MM-DD" is always exactly 10 characters
    region_name = stem[len(head): -11]  # strip "<prefix>_" and "_<date>"
    if not region_name:
        raise ValueError(f"{name!r} has no region segment between prefix and date")
    if not _DATE_RE.match(date_str):
        raise ValueError(f"{name!r} does not end in a YYYY-MM-DD date")
    if region_name not in region_defs.list_regions():
        raise ValueError(
            f"{name!r} parsed to unrecognized region {region_name!r} -- likely an "
            "unrelated file sharing a Drive folder, not a genuine collision. "
            f"Known regions: {region_defs.list_regions()}"
        )
    year, month, _ = date_str.split("-")
    return region_name, date_str, year, month


def sync_prefixed(
    destination_root: Path | str, *, name_prefix: str, subdir: str, region_name: str | None = None
) -> int:
    """Download completed Earth Engine exports named
    ``"<name_prefix>_<region>_<YYYY-MM-DD>.tif"`` (flat Drive filename, no
    embedded "/" -- see :func:`parse_prefixed_export_name`), writing each to
    ``destination_root/subdir/<region>/<year>/<month>/<name>`` and trashing
    the Drive copy after a successful download. Generalizes :func:`sync_oya`
    for any region-scoped export family (``era5``, ``imerg``, region-scoped
    ``oya``).

    Pass ``region_name`` (recommended) to scope the Drive query to
    ``"<name_prefix>_<region_name>_"`` instead of just ``"<name_prefix>_"``.
    Without it, a process syncing one region's exports also lists, downloads
    (harmlessly re-skipped once local) and attempts to trash every *other*
    region's exports sharing the same Drive folder and prefix -- found in
    production, 2026-09-21, running three regions in parallel: each was
    listing and processing all three regions' files on every sync, tripling
    Drive API traffic for no benefit (the file still ends up in the right
    place either way, since the region is parsed from its own name, but the
    wasted listing/trash-attempt overhead compounds as more files
    accumulate).
    """
    service = _service()
    parent = _folder_id(service, Config.GEE_DRIVE_FOLDER)
    name_filter = f"{name_prefix}_{region_name}_" if region_name else f"{name_prefix}_"
    query = (
        f"'{parent}' in parents and name contains '{name_filter}' "
        "and name contains '.tif' and trashed=false"
    )
    downloaded = 0
    for item in _list_all(service, query):
        name = Path(item["name"]).name
        try:
            parsed_region, _date_str, year, month = parse_prefixed_export_name(name, name_prefix)
        except ValueError as error:
            logger.warning("Skipping Drive file %r: %s", name, error)
            continue
        destination = Path(destination_root) / subdir / parsed_region / year / month / name
        if destination.exists():
            _try_trash(service, item["id"], name)
        elif _try_download(service, item["id"], destination, name):
            downloaded += 1
            _try_trash(service, item["id"], name)
    return downloaded
