"""Acquire pure ERA5 convective + large-scale precipitation via Google Earth
Engine (revision 2026-09-21 of PASO 3.1,
docs/decisions/0002-implementacion-v2.md).

**Why Earth Engine instead of the Copernicus CDS API.** The project has
Earth Engine + Drive service-account credentials already configured
(``auth/service_account.json``, the same ones ``export_oya.py`` uses), not
CDS credentials. Earth Engine's curated ``ECMWF/ERA5/HOURLY`` collection
(the **pure** ERA5 product -- not ERA5-Land, which P1/spateGAN-ERA5
excludes by design, matrix M102) happens to expose exactly the two
single-level variables P1 needs as coarse input:

* ``mean_convective_precipitation_rate``
* ``mean_large_scale_precipitation_rate``

at ERA5's native ~0.25 deg resolution, "time-aggregated over the one hour
ending at the validity date and time" (hour-end timestamp convention;
verified against the Earth Engine Data Catalog page for
``ECMWF/ERA5/HOURLY``, 2026-09-21). No CDS credential is required. This
supersedes the earlier ``era5_cds.py`` (removed; see git history and
``docs/decisions/0002-implementacion-v2.md`` for the revision note) -- the
physical variables and their role in the pipeline are unchanged, only the
acquisition backend differs.

**Units.** The Earth Engine bands are *rates* in kg/m^2/s (equivalently
mm/s spread over the grid cell). Multiplying by 3600 gives the accumulated
depth in mm for that hour -- the same physical quantity the CDS API would
have returned in metres (after a x1000 unit conversion), just from a
different backend.

**Delivery**, mirroring ``export_oya.py``: one multi-band GeoTIFF per
``(region, day)`` -- 24 hours x 2 variables = 48 bands, band names
``"<hour-index>_<ee band name>"`` from ``ee.ImageCollection.toBands()`` --
submitted as Earth Engine batch export-to-Drive tasks and synced locally via
``drive.sync_prefixed``, with a hashed manifest
(``src.utils.provenance``). Resumable by local file presence.

Run from the repository root::

    python -m src.data_extraction.era5_ee --start-year 2020 --end-year 2020 --region mexico_pacifico
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import ee
from dateutil.relativedelta import relativedelta

from src.data_extraction import drive, earth_engine
from src.data_preprocessing import regions as region_defs
from src.utils.config import Config
from src.utils.logging import configure_logging
from src.utils.provenance import file_sha256, load_manifest, write_manifest
from src.utils.retries import start_task_with_retry

logger = configure_logging(name="precipitation.era5_ee")

ERA5_COLLECTION = "ECMWF/ERA5/HOURLY"
ERA5_BANDS = ("mean_convective_precipitation_rate", "mean_large_scale_precipitation_rate")
SECONDS_PER_HOUR = 3600.0
ERA5_SCALE_METERS = 27830  # native ERA5 grid, ~0.25 deg
TASK_BATCH_SIZE = 20


def rate_to_hourly_mm(rate_kg_m2_s: float) -> float:
    """kg/m^2/s -> accumulated mm for a one-hour window (1 kg/m^2 of liquid
    water == 1 mm depth)."""
    return rate_kg_m2_s * SECONDS_PER_HOUR


def _daily_stack(date_str: str) -> "ee.Image":
    """One 48-band image for ``date_str``: 24 hourly steps x
    ``ERA5_BANDS``, converted from a rate to accumulated mm per hour."""
    start = ee.Date(date_str)
    collection = (
        ee.ImageCollection(ERA5_COLLECTION)
        .filterDate(start, start.advance(1, "day"))
        .select(list(ERA5_BANDS))
    )
    stacked = collection.toBands().multiply(SECONDS_PER_HOUR)
    return stacked.set("system:time_start", start.millis())


def _submit(date: datetime, region_name: str):
    date_str = date.strftime("%Y-%m-%d")
    geometry = earth_engine.region_geometry(region_name)
    image = _daily_stack(date_str).clip(geometry)
    # Flat Drive filename (no "/"): see export_oya.py's region-scoped branch
    # and drive.parse_prefixed_export_name for why this matters.
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f"v2_generative_era5_{region_name}_{date_str}",
        folder=Config.GEE_DRIVE_FOLDER,
        fileNamePrefix=f"era5_{region_name}_{date_str}",
        region=geometry,
        scale=ERA5_SCALE_METERS,
        crs="EPSG:4326",
        maxPixels=1e13,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    start_task_with_retry(task)
    return task


def _local_target(root: Path, region_name: str, date: datetime) -> Path:
    return (
        root / "era5_ee" / region_name / f"{date.year}" / f"{date.month:02d}"
        / f"era5_{region_name}_{date:%Y-%m-%d}.tif"
    )


def _manifest_path(root: Path, region_name: str) -> Path:
    return root / "era5_ee" / region_name / f"manifest_era5_ee_{region_name}.json"


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
    testing; production callers should leave them as the Earth Engine /
    Drive defaults. Returns the manifest path.
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
        sync = lambda: drive.sync_prefixed(root, name_prefix="era5", subdir="era5_ee")

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
            logger.info("Submitting %d missing ERA5 days for %s", len(functions), month.strftime("%Y-%m"))
            submit_in_batches(functions)
            downloaded = sync()
            logger.info("Downloaded %d ERA5 files for %s", downloaded, month.strftime("%Y-%m"))
        else:
            logger.info("ERA5 month already complete: %s", month.strftime("%Y-%m"))
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
        kind="era5_ee",
        records=records,
        extra={
            "collection": ERA5_COLLECTION,
            "bands": list(ERA5_BANDS),
            "region": region_defs.region_summary(region_name),
            "range": {"start_year": start_year, "end_year": end_year},
            "band_layout": "toBands(): 48 bands per day, '<hour-index>_<ee-band-name>', values in mm accumulated per hour",
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
