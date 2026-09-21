"""Acquire IMERG V07 Final Run precipitation via Google Earth Engine
(revision 2026-09-21 of PASO 3.2, docs/decisions/0002-implementacion-v2.md).

**Why Earth Engine instead of direct NASA GES DISC HTTP access.** Same
rationale as ``era5_ee.py``: the project has Earth Engine + Drive
service-account credentials already configured
(``auth/service_account.json``), and Earth Engine hosts IMERG V07 directly
(``NASA/GPM_L3/IMERG_V07``), half-hourly, band ``precipitation`` (mm/hr,
the calibrated combined microwave+IR estimate -- the Final-Run-equivalent
product, confirmed against the Earth Engine Data Catalog page,
2026-09-21). No NASA Earthdata Login token is required. This supersedes the
earlier GES-DISC-HTTP-based ``imerg.py`` (same filename, rewritten in
place; see git history) -- the target product (IMERG V07) and its role
(primary hourly ground truth, RainShift P2 M203) are unchanged, only the
acquisition backend differs.

Caveat carried over from the Earth Engine catalog page: recent granules are
marked "provisional" until NASA replaces them with a "permanent" version;
the provider's IMERG V07 "permanent" product line ends 2025-09-30 ahead of
a V08 transition, so acquisition for dates at or after that boundary will
only ever see provisional data.

**Delivery**, mirroring ``era5_ee.py``: one 48-band GeoTIFF per
``(region, day)`` (one band per half-hour, band names
``"<slot-index>_precipitation"`` from ``toBands()``), submitted as Earth
Engine batch export-to-Drive tasks and synced locally via
``drive.sync_prefixed``, with a hashed manifest
(``src.utils.provenance``). Resumable by local file presence.
``halfhourly_to_hourly`` still reduces the 48 half-hourly rate bands to 24
hourly means (RainShift, P2 M203), now operating on the locally downloaded
GeoTIFF's bands instead of remote HDF5 granules.

Run from the repository root::

    python -m src.data_extraction.imerg --start-year 2020 --end-year 2020 --region mexico_pacifico
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import ee
import numpy as np
from dateutil.relativedelta import relativedelta

from src.data_extraction import drive, earth_engine
from src.data_preprocessing import regions as region_defs
from src.utils.config import Config
from src.utils.logging import configure_logging
from src.utils.provenance import file_sha256, load_manifest, write_manifest
from src.utils.retries import start_task_with_retry

logger = configure_logging(name="precipitation.imerg")

IMERG_COLLECTION = "NASA/GPM_L3/IMERG_V07"
IMERG_BAND = "precipitation"
SLOTS_PER_DAY = 48  # half-hourly
IMERG_SCALE_METERS = 11132  # native IMERG grid, 0.1 deg


def _daily_stack(date_str: str) -> "ee.Image":
    """One 48-band image for ``date_str``: 48 half-hourly ``precipitation``
    snapshots (mm/hr, calibrated), band-order chronological via ``toBands()``."""
    start = ee.Date(date_str)
    collection = (
        ee.ImageCollection(IMERG_COLLECTION)
        .filterDate(start, start.advance(1, "day"))
        .select([IMERG_BAND])
    )
    return collection.toBands().set("system:time_start", start.millis())


def _submit(date: datetime, region_name: str):
    date_str = date.strftime("%Y-%m-%d")
    geometry = earth_engine.region_geometry(region_name)
    image = _daily_stack(date_str).clip(geometry)
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f"v2_generative_imerg_{region_name}_{date_str}",
        folder=Config.GEE_DRIVE_FOLDER,
        fileNamePrefix=f"imerg_{region_name}_{date_str}",
        region=geometry,
        scale=IMERG_SCALE_METERS,
        crs="EPSG:4326",
        maxPixels=1e13,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    start_task_with_retry(task)
    return task


def _local_target(root: Path, region_name: str, date: datetime) -> Path:
    return (
        root / "imerg" / region_name / f"{date.year}" / f"{date.month:02d}"
        / f"imerg_{region_name}_{date:%Y-%m-%d}.tif"
    )


def _manifest_path(root: Path, region_name: str) -> Path:
    return root / "imerg" / region_name / f"manifest_imerg_{region_name}.json"


def halfhourly_to_hourly(slots: np.ndarray) -> np.ndarray:
    """Average 48 half-hourly rate fields (mm/hr) into 24 hourly means
    (mm/hr) -- RainShift, P2 M203. ``slots``: ``(48, lat, lon)``."""
    if slots.shape[0] != SLOTS_PER_DAY:
        raise ValueError(f"expected {SLOTS_PER_DAY} slots, got {slots.shape[0]}")
    pairs = slots.reshape(24, 2, *slots.shape[1:])
    with np.errstate(invalid="ignore"):
        return np.nanmean(pairs, axis=1).astype(np.float32)


def read_daily_geotiff_hourly(path: Path) -> np.ndarray:
    """Read a downloaded 48-band daily GeoTIFF and reduce it to 24 hourly
    means via :func:`halfhourly_to_hourly`."""
    import rasterio

    with rasterio.open(path) as source:
        if source.count != SLOTS_PER_DAY:
            raise ValueError(f"expected {SLOTS_PER_DAY} bands in {path}, got {source.count}")
        slots = source.read().astype(np.float32)  # (48, H, W)
    return halfhourly_to_hourly(slots)


def download_range(
    start_year: int,
    end_year: int,
    region_name: str,
    *,
    root: Path | None = None,
    submit_in_batches=None,
    sync=None,
) -> Path:
    """Acquire every missing day in ``[start_year, end_year]`` for
    ``region_name``. ``submit_in_batches``/``sync`` are injectable for
    testing. Returns the manifest path.
    """
    region_defs.get_region(region_name)  # raises on an unknown region name
    root = Path(root or Config.V2_RAW_DATA_DIR)
    manifest_path = _manifest_path(root, region_name)
    previous = load_manifest(manifest_path) or {}
    known = {rec["path"]: rec for rec in previous.get("records", [])}

    earth_engine.initialize()
    if submit_in_batches is None:
        from src.data_extraction.export_oya import _submit_in_batches as submit_in_batches
    if sync is None:
        sync = lambda: drive.sync_prefixed(root, name_prefix="imerg", subdir="imerg")

    month = datetime(start_year, 1, 1)
    final_month = datetime(end_year, 12, 1)
    while month <= final_month:
        next_month = month + relativedelta(months=1)
        functions = []
        date = month
        while date < next_month:
            if not _local_target(root, region_name, date).exists():
                functions.append(lambda value=date: [_submit(value, region_name)])
            date += relativedelta(days=1)
        if functions:
            logger.info("Submitting %d missing IMERG days for %s", len(functions), month.strftime("%Y-%m"))
            submit_in_batches(functions)
            downloaded = sync()
            logger.info("Downloaded %d IMERG files for %s", downloaded, month.strftime("%Y-%m"))
        else:
            logger.info("IMERG month already complete: %s", month.strftime("%Y-%m"))
        month = next_month

    records = []
    day = datetime(start_year, 1, 1)
    final_day = datetime(end_year, 12, 31)
    while day <= final_day:
        target = _local_target(root, region_name, day)
        if target.exists():
            rel = str(target.relative_to(root))
            records.append(
                known.get(rel)
                or {"path": rel, "day": day.strftime("%Y-%m-%d"), "sha256": file_sha256(target),
                    "bytes": target.stat().st_size}
            )
        day += relativedelta(days=1)

    manifest = write_manifest(
        manifest_path,
        kind="imerg",
        records=records,
        extra={
            "collection": IMERG_COLLECTION,
            "band": IMERG_BAND,
            "region": region_defs.region_summary(region_name),
            "range": {"start_year": start_year, "end_year": end_year},
            "band_layout": "toBands(): 48 half-hourly bands per day, '<slot-index>_precipitation', mm/hr",
            "temporal_reduction": "half-hourly rate -> hourly mean applied later via halfhourly_to_hourly (RainShift, P2 M203)",
        },
    )
    logger.info("manifest %s (%d records, sha256=%s)", manifest_path,
                manifest["record_count"], manifest["manifest_sha256"][:12])
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--region", required=True, choices=region_defs.list_regions())
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    download_range(args.start_year, args.end_year, args.region, root=args.root)


if __name__ == "__main__":
    main()
