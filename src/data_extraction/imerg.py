"""Acquire IMERG V07 Final Run precipitation from NASA GES DISC as the
primary hourly ground-truth target (PASO 3.2 of
``docs/decisions/0002-implementacion-v2.md``).

The half-hourly ``GPM_3IMERGHH.07`` product is downloaded from GES DISC,
subset to a region bounding box (``src.data_preprocessing.regions``), and
**averaged from half-hourly to hourly** to match ERA5's temporal resolution
(RainShift, P2 M203). Output is one NetCDF per day with 24 hourly steps::

    $V2_RAW_DATA_DIR/imerg/<region>/imerg_hourly_YYYY-MM-DD.nc

Resilience mirrors ``export_oya.py``: bounded retries with backoff, a hashed
download manifest, and resume by local-file presence. GES DISC serves files
over HTTPS directly (no task-polling queue like Earth Engine), so retry
happens per-granule instead.

Credentials are mandatory and checked up front. Supported, in priority
order:

1. ``EARTHDATA_TOKEN`` -> ``Authorization: Bearer`` header (recommended)
2. ``EARTHDATA_USERNAME`` + ``EARTHDATA_PASSWORD`` -> HTTP basic
3. a ``~/.netrc`` entry for ``urs.earthdata.nasa.gov``

The module never fabricates data; a missing credential raises with the exact
variable name to set (see ``env.example``).

Run from the repository root::

    python -m src.data_extraction.imerg --start 2001-01-01 --end 2022-12-31 --region mexico_pacifico
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import time
from typing import Iterator

import numpy as np

from src.data_preprocessing import regions as region_defs
from src.utils.config import Config
from src.utils.logging import configure_logging
from src.utils.provenance import file_sha256, load_manifest, write_manifest

logger = configure_logging(name="precipitation.imerg")

GESDISC_ROOT = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07"
SLOTS_PER_DAY = 48  # half-hourly
GRANULE_VERSION = "V07B"


class MissingCredentialsError(RuntimeError):
    """Raised when no usable NASA Earthdata credentials are available."""


def _parse_date(text: str) -> date:
    return date.fromisoformat(text)


def _iter_days(start: date, end: date) -> Iterator[date]:
    if start > end:
        raise ValueError("--start must not be after --end")
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def resolve_earthdata_auth() -> dict:
    """Return an auth descriptor or raise ``MissingCredentialsError``."""
    if Config.EARTHDATA_TOKEN:
        return {"kind": "bearer", "token": Config.EARTHDATA_TOKEN}
    if Config.EARTHDATA_USERNAME and Config.EARTHDATA_PASSWORD:
        return {
            "kind": "basic",
            "user": Config.EARTHDATA_USERNAME,
            "password": Config.EARTHDATA_PASSWORD,
        }
    netrc = Path.home() / ".netrc"
    if netrc.exists() and "urs.earthdata.nasa.gov" in netrc.read_text():
        return {"kind": "netrc"}
    raise MissingCredentialsError(
        "No NASA Earthdata credentials found. Set EARTHDATA_TOKEN (a bearer "
        "token from https://urs.earthdata.nasa.gov/), or EARTHDATA_USERNAME + "
        "EARTHDATA_PASSWORD, or add a 'machine urs.earthdata.nasa.gov' entry "
        "to ~/.netrc. See env.example."
    )


def _make_session(auth: dict):
    import requests

    session = requests.Session()
    if auth["kind"] == "bearer":
        session.headers["Authorization"] = f"Bearer {auth['token']}"
    elif auth["kind"] == "basic":
        session.auth = (auth["user"], auth["password"])
    return session


def granule_urls(day: date) -> list[str]:
    """Return the 48 half-hourly granule URLs for ``day`` (GES DISC layout)."""
    doy = day.timetuple().tm_yday
    stamp = day.strftime("%Y%m%d")
    urls: list[str] = []
    for slot in range(SLOTS_PER_DAY):
        minutes = slot * 30
        s_h, s_m = divmod(minutes, 60)
        end_total = minutes + 29
        e_h, e_m = divmod(end_total, 60)
        name = (
            f"3B-HHR.MS.MRG.3IMERG.{stamp}"
            f"-S{s_h:02d}{s_m:02d}00-E{e_h:02d}{e_m:02d}59.{minutes:04d}.{GRANULE_VERSION}.HDF5"
        )
        urls.append(f"{GESDISC_ROOT}/{day.year}/{doy:03d}/{name}")
    return urls


def _download_with_retry(session, url: str, *, max_retries: int = 6,
                          initial_delay: float = 3.0, backoff: float = 2.0) -> bytes:
    import requests

    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, timeout=120, allow_redirects=True)
            if response.status_code in (401, 403):
                raise MissingCredentialsError(
                    f"Earthdata rejected the request ({response.status_code}) for "
                    f"{url}. Check EARTHDATA_TOKEN / credentials and that the GES "
                    "DISC 'GPM_3IMERGHH.07' application is authorized in your URS "
                    "profile."
                )
            if response.status_code == 404:
                raise FileNotFoundError(url)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException:
            if attempt == max_retries:
                raise
            wait = delay * (0.5 + np.random.random())
            logger.warning("granule fetch failed (%s/%s); retry in %.1fs", attempt, max_retries, wait)
            time.sleep(wait)
            delay *= backoff
    raise RuntimeError("unreachable")


def _read_granule(content: bytes, region: region_defs.Region) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(precip[lat,lon], lats, lons)`` for the region bbox subset.

    IMERG V07 stores ``/Grid/precipitation`` as ``(time=1, lon, lat)`` mm/hr.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The 'h5py' package is required to read IMERG HDF5 granules. "
            "Install it with: ./venv/bin/pip install h5py"
        ) from exc
    import io

    with h5py.File(io.BytesIO(content), "r") as handle:
        grid = handle["Grid"]
        lons = grid["lon"][:]
        lats = grid["lat"][:]
        precip = np.asarray(grid["precipitation"][0])  # (lon, lat)
        fill = grid["precipitation"].attrs.get("_FillValue", -9999.9)

    lon_min, lat_min, lon_max, lat_max = region.bbox
    lon_sel = (lons >= lon_min) & (lons <= lon_max)
    lat_sel = (lats >= lat_min) & (lats <= lat_max)
    sub = precip[np.ix_(lon_sel, lat_sel)].T  # -> (lat, lon)
    sub = np.where(sub == fill, np.nan, sub).astype(np.float32)
    return sub, lats[lat_sel], lons[lon_sel]


def halfhourly_to_hourly(slots: np.ndarray) -> np.ndarray:
    """Average 48 half-hourly rate fields (mm/hr) into 24 hourly means.

    ``slots`` has shape ``(48, lat, lon)``.
    """
    if slots.shape[0] != SLOTS_PER_DAY:
        raise ValueError(f"expected {SLOTS_PER_DAY} slots, got {slots.shape[0]}")
    pairs = slots.reshape(24, 2, *slots.shape[1:])
    with np.errstate(invalid="ignore"):
        return np.nanmean(pairs, axis=1).astype(np.float32)


def _write_day_netcdf(path: Path, hourly: np.ndarray, lats: np.ndarray,
                       lons: np.ndarray, day: date) -> None:
    import netCDF4

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".nc.part")
    with netCDF4.Dataset(tmp, "w") as ds:
        ds.createDimension("time", 24)
        ds.createDimension("lat", lats.size)
        ds.createDimension("lon", lons.size)
        t = ds.createVariable("time", "i4", ("time",))
        t.units = f"hours since {day.isoformat()} 00:00:00"
        t[:] = np.arange(24)
        vlat = ds.createVariable("lat", "f4", ("lat",))
        vlon = ds.createVariable("lon", "f4", ("lon",))
        vlat[:] = lats
        vlon[:] = lons
        var = ds.createVariable(
            "precipitation", "f4", ("time", "lat", "lon"), zlib=True,
            fill_value=np.float32(np.nan),
        )
        var.units = "mm/hr"
        var.long_name = "IMERG V07 Final Run hourly-averaged precipitation rate"
        var[:] = hourly
        ds.source = "GPM_3IMERGHH.07 (NASA GES DISC), half-hourly averaged to hourly"
    tmp.replace(path)


def _target_path(root: Path, region_name: str, day: date) -> Path:
    return root / "imerg" / region_name / f"imerg_hourly_{day.isoformat()}.nc"


def _manifest_path(root: Path, region_name: str) -> Path:
    return root / "imerg" / region_name / f"manifest_imerg_{region_name}.json"


def download_range(
    start: date,
    end: date,
    region: region_defs.Region,
    *,
    root: Path | None = None,
    session=None,
    dry_run: bool = False,
) -> Path:
    """Download + hourly-average IMERG for the date range. Returns manifest path."""
    root = Path(root or Config.V2_RAW_DATA_DIR)
    manifest_path = _manifest_path(root, region.name)
    previous = load_manifest(manifest_path) or {}
    known = {rec["path"]: rec for rec in previous.get("records", [])}

    resolve_earthdata_auth()  # verify credentials even in dry-run
    if session is None and not dry_run:
        session = _make_session(resolve_earthdata_auth())

    records: list[dict] = []
    for day in _iter_days(start, end):
        target = _target_path(root, region.name, day)
        rel = str(target.relative_to(root))
        if target.exists() and rel in known and target.stat().st_size > 0:
            logger.info("skip %s (already present)", rel)
            records.append(known[rel])
            continue
        if dry_run:
            logger.info("would fetch %d granules for %s -> %s", SLOTS_PER_DAY, day, rel)
            records.append({
                "path": rel, "day": day.isoformat(), "granules": SLOTS_PER_DAY,
                "sha256": None, "bytes": None,
            })
            continue

        slots = []
        lats = lons = None
        for url in granule_urls(day):
            content = _download_with_retry(session, url)
            field, lats, lons = _read_granule(content, region)
            slots.append(field)
        hourly = halfhourly_to_hourly(np.stack(slots))
        _write_day_netcdf(target, hourly, lats, lons, day)
        records.append({
            "path": rel,
            "day": day.isoformat(),
            "granules": SLOTS_PER_DAY,
            "sha256": file_sha256(target),
            "bytes": target.stat().st_size,
        })
        logger.info("wrote %s", rel)

    manifest = write_manifest(
        manifest_path,
        kind="imerg_hourly",
        records=records,
        extra={
            "product": "GPM_3IMERGHH.07",
            "region": region_defs.region_summary(region.name),
            "range": {"start": start.isoformat(), "end": end.isoformat()},
            "temporal_reduction": "half-hourly rate -> hourly mean (RainShift, P2 M203)",
        },
    )
    logger.info("manifest %s (%d records, sha256=%s)", manifest_path,
                manifest["record_count"], manifest["manifest_sha256"][:12])
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="first day, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="last day, YYYY-MM-DD (inclusive)")
    parser.add_argument("--region", required=True, choices=region_defs.list_regions())
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    download_range(
        _parse_date(args.start),
        _parse_date(args.end),
        region_defs.get_region(args.region),
        root=args.root,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
