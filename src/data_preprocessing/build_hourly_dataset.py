r"""Build the hourly ERA5->IMERG training dataset (PASO 3.2/"etapa 2" of the
v2-generative framework -- docs/stages_v2/02-build.md has the full
narrative documentation; this module is the implementation).

Converts raw per-day GeoTIFFs acquired by ``era5_ee.py``/``imerg.py``
(``src/data_extraction/``) into aligned, masked, versioned hourly samples,
one NPZ file per (region, day) holding all 24 hours -- the direct hourly
analogue of the legacy daily pipeline's
``src/data_preprocessing/build_dataset.py``, but for a completely different
grid per region instead of one fixed 460x720 domain.

Pipeline per (region, day):

1. Read the day's 48-band ERA5 GeoTIFF (24 hours x [cp, lsp], see
   ``era5_ee.py``) and reproject all 48 bands at once onto the region's
   canonical grid (:func:`region_grid`), bilinear resampling.
2. Read the day's 48-band IMERG GeoTIFF (48 half-hourly snapshots),
   reproject onto the same canonical grid, then reduce pairs to 24 hourly
   means (:func:`imerg.halfhourly_to_hourly`).
3. Reproject NASADEM elevation onto the canonical grid once per region
   (cached) and derive a land/sea mask from it.
4. Apply the region's basin-intersected mask
   (``src.data_preprocessing.regions.region_mask``).
5. Write one NPZ per day with all 24 hours, plus a per-region CSV index and
   a hashed manifest (``src.utils.provenance``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date, timedelta
from pathlib import Path
import re
import warnings

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from src.data_extraction.imerg import halfhourly_to_hourly
from src.data_preprocessing import regions as region_defs
from src.data_preprocessing.quality import ensure_finite_features, file_sha256, manifest_sha256
from src.utils.config import Config
from src.utils.logging import configure_logging
from src.utils.provenance import write_manifest

logger = configure_logging(name="precipitation.build_hourly")

HOURLY_DATASET_VERSION = "v2_generative_hourly"
IMERG_RESOLUTION_DEG = 0.1  # matches IMERG's native ~0.1 deg grid
TARGET_CRS = "EPSG:4326"
HOURS_PER_DAY = 24

# Provisional year split (documented as OQ-6 in
# docs/decisions/0002-implementacion-v2.md; revisit once real coverage is
# known). Applied uniformly to every region's samples; which region+split
# combinations a training run actually uses (e.g. never training on
# puerto_rico_caribe, per its "eval" role in src/data_preprocessing/regions.py)
# is a training-time decision, not a build-time one.
TRAIN_YEARS = range(2010, 2020)   # 2010-2019
VAL_YEARS = (2020, 2021)
TEST_YEARS = (2022, 2023)

# Elevation above this value (metres) counts as land; NASADEM has no nodata
# over open ocean in this archive (verified: min -71 m, ocean samples read
# exactly 0 m), so a plain threshold is workable but approximate -- same
# caveat as the legacy pipeline's land mask (docs/stages/data-and-resources.md
# "Known limitations"), now without CHIRPS coverage to corroborate it since
# CHIRPS is excluded from this cascade (docs/decisions/0002-implementacion-v2.md
# §1.4).
LAND_ELEVATION_THRESHOLD_M = 0.0


def year_split(year: int) -> str:
    if year in TRAIN_YEARS:
        return "train"
    if year in VAL_YEARS:
        return "val"
    if year in TEST_YEARS:
        return "test"
    raise ValueError(f"Year {year} is outside the configured TRAIN/VAL/TEST ranges")


@dataclass(frozen=True)
class RegionGrid:
    transform: "rasterio.Affine"
    crs: str
    height: int
    width: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def lon_lat_1d(self) -> tuple[np.ndarray, np.ndarray]:
        """Pixel-center longitude/latitude 1-D coordinate arrays."""
        res = self.transform.a  # pixel width == pixel height (square, north-up)
        lon_min = self.transform.c
        lat_max = self.transform.f
        lons = lon_min + res * (np.arange(self.width) + 0.5)
        lats = lat_max - res * (np.arange(self.height) + 0.5)
        return lons, lats


def region_grid(region_name: str, resolution_deg: float = IMERG_RESOLUTION_DEG) -> RegionGrid:
    """The canonical output grid for one region: a clean north-up grid at
    ``resolution_deg`` spanning the region's bounding box exactly, independent
    of any particular download's own pixel snapping (Earth Engine exports are
    snapped to its own global grid origin, which shifts slightly per region --
    see docs/stages_v2/02-build.md). Every source raster is reprojected onto
    this grid, not the other way around.
    """
    region = region_defs.get_region(region_name)
    width = int(round((region.lon_max - region.lon_min) / resolution_deg))
    height = int(round((region.lat_max - region.lat_min) / resolution_deg))
    transform = from_origin(region.lon_min, region.lat_max, resolution_deg, resolution_deg)
    return RegionGrid(transform=transform, crs=TARGET_CRS, height=height, width=width)


def _reproject_all_bands(path: str | Path, grid: RegionGrid, resampling: Resampling) -> np.ndarray:
    """Reproject every band of a source raster onto ``grid``. Returns
    ``(bands, H, W)`` float32, NaN where the source has no coverage."""
    with rasterio.open(path) as source:
        destination = np.full((source.count, grid.height, grid.width), np.nan, dtype=np.float32)
        source_data = source.read(masked=True)
        for band in range(source.count):
            reproject(
                source=np.asarray(source_data[band].filled(np.nan), dtype=np.float32),
                destination=destination[band],
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=np.nan,
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                dst_nodata=np.nan,
                resampling=resampling,
            )
    return destination


_ERA5_BAND_RE = re.compile(
    r"^\d{8}T(?P<hour>\d{2})_mean_(?P<kind>convective|large_scale)_precipitation_rate$"
)


def parse_era5_hour_bands(descriptions: tuple[str | None, ...]) -> dict[int, dict[str, int]]:
    """Map each hour (0-23) to its ``{"cp": band_index, "lsp": band_index}``
    (0-based indices into the array ``_reproject_all_bands`` returns),
    parsed from the ``toBands()`` band descriptions
    (``"<YYYYMMDD>T<HH>_mean_convective_precipitation_rate"`` etc. --
    ``era5_ee.py``'s ``_daily_stack``) rather than assumed from a fixed
    stride, so a change in Earth Engine's ``toBands()`` ordering would raise
    instead of silently misattributing a band.
    """
    by_hour: dict[int, dict[str, int]] = {}
    for index, description in enumerate(descriptions):
        if not description:
            raise ValueError(f"ERA5 band {index} has no description; cannot identify its hour")
        match = _ERA5_BAND_RE.match(description)
        if not match:
            raise ValueError(f"ERA5 band description {description!r} does not match the expected pattern")
        hour = int(match.group("hour"))
        kind = "cp" if match.group("kind") == "convective" else "lsp"
        by_hour.setdefault(hour, {})[kind] = index
    missing = [h for h in range(HOURS_PER_DAY) if by_hour.get(h, {}).keys() != {"cp", "lsp"}]
    if missing:
        raise ValueError(f"ERA5 file is missing cp/lsp bands for hours {missing}")
    return by_hour


def reproject_era5_day(path: str | Path, grid: RegionGrid) -> np.ndarray:
    """Return ``(24, H, W, 2)`` accumulated-mm-per-hour ERA5 [cp, lsp]."""
    with rasterio.open(path) as source:
        descriptions = source.descriptions
    hour_bands = parse_era5_hour_bands(descriptions)
    reprojected = _reproject_all_bands(path, grid, Resampling.bilinear)  # (48, H, W)
    out = np.empty((HOURS_PER_DAY, grid.height, grid.width, 2), dtype=np.float32)
    for hour, bands in hour_bands.items():
        out[hour, ..., 0] = reprojected[bands["cp"]]
        out[hour, ..., 1] = reprojected[bands["lsp"]]
    return out


def reproject_imerg_day(path: str | Path, grid: RegionGrid) -> np.ndarray:
    """Return ``(24, H, W)`` hourly-mean IMERG precipitation rate (mm/hr).

    Reprojecting onto the region's canonical grid can leave a thin border of
    pixels with no source coverage (NaN); if both half-hour slots for such a
    pixel are NaN, ``nanmean`` correctly returns NaN but also emits a
    "Mean of empty slice" RuntimeWarning per occurrence -- harmless (those
    pixels are excluded downstream by the finite-value validity mask,
    ``ensure_finite_features``), just noisy at scale, so it is suppressed
    here rather than left to spam the build log for every border pixel of
    every day.
    """
    reprojected = _reproject_all_bands(path, grid, Resampling.bilinear)  # (48, H, W)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        return halfhourly_to_hourly(reprojected)


def reproject_elevation(dem_path: str | Path, grid: RegionGrid) -> np.ndarray:
    """Return ``(H, W)`` elevation in metres, reprojected onto ``grid``."""
    with rasterio.open(dem_path) as source:
        destination = np.full(grid.shape, np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return destination


def land_mask_from_elevation(elevation: np.ndarray, threshold: float = LAND_ELEVATION_THRESHOLD_M) -> np.ndarray:
    return elevation > threshold


def _source_path(root: Path, kind: str, region_name: str, day: _date) -> Path:
    subdir = "era5_ee" if kind == "era5" else "imerg"
    return (
        root / subdir / region_name / f"{day.year}" / f"{day.month:02d}"
        / f"{'era5' if kind == 'era5' else 'imerg'}_{region_name}_{day.isoformat()}.tif"
    )


def _output_path(output_root: Path, region_name: str, day: _date) -> Path:
    return output_root / region_name / f"{day.year}" / f"{day.month:02d}" / f"{day.isoformat()}.npz"


def build_day(
    region_name: str,
    day: _date,
    *,
    v2_raw_root: Path,
    output_root: Path,
    elevation: np.ndarray,
    land_mask: np.ndarray,
    basin_mask: np.ndarray,
) -> dict:
    """Build one day's aligned hourly sample for one region. Returns an
    index record (always); writes the NPZ only when accepted."""
    grid = region_grid(region_name)
    record: dict = {"date": day.isoformat(), "region": region_name, "split": year_split(day.year), "accepted": False}

    era5_path = _source_path(v2_raw_root, "era5", region_name, day)
    imerg_path = _source_path(v2_raw_root, "imerg", region_name, day)
    missing = [str(p) for p in (era5_path, imerg_path) if not p.exists()]
    if missing:
        record["reject_reasons"] = ["missing_source_file"]
        return record

    inputs = reproject_era5_day(era5_path, grid)   # (24, H, W, 2)
    target = reproject_imerg_day(imerg_path, grid)  # (24, H, W)

    input_valid = ensure_finite_features(inputs) & basin_mask[None, :, :]
    target_valid = np.isfinite(target) & (np.abs(target) < 1.0e10) & basin_mask[None, :, :]

    if not bool(target_valid.any()):
        record["reject_reasons"] = ["no_valid_target_pixels"]
        return record

    output_path = _output_path(output_root, region_name, day)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        inputs=inputs.astype(np.float32),
        target=target.astype(np.float32)[..., None],
        input_valid_mask=input_valid,
        target_valid_mask=target_valid,
        land_mask=land_mask,
        elevation=elevation.astype(np.float32),
        basin_mask=basin_mask,
        date=day.isoformat(),
        region=region_name,
        crs=grid.crs,
        transform=np.asarray(tuple(grid.transform), dtype=np.float64),
        preprocessing_version=HOURLY_DATASET_VERSION,
    )
    record.update(
        accepted=True,
        npz_path=str(output_path),
        valid_target_pixel_hours=int(target_valid.sum()),
        era5_sha256=file_sha256(era5_path),
        imerg_sha256=file_sha256(imerg_path),
    )
    return record


def _iter_days(start: _date, end: _date):
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def build_region(
    region_name: str,
    *,
    v2_raw_root: Path | None = None,
    output_root: Path | None = None,
    start: _date | None = None,
    end: _date | None = None,
    limit: int | None = None,
) -> dict:
    """Build every available day for one region. Writes a CSV index and a
    hashed manifest under ``output_root``. Returns the manifest dict."""
    v2_raw_root = Path(v2_raw_root or Config.V2_RAW_DATA_DIR)
    output_root = Path(output_root or (Path(Config.LOCAL_DATA_DIR) / HOURLY_DATASET_VERSION / "processed"))
    start = start or _date(min(TRAIN_YEARS), 1, 1)
    end = end or _date(max(TEST_YEARS), 12, 31)

    grid = region_grid(region_name)
    dem_path = Path(Config.RAW_DATA_DIR) / "dem" / "nasadem_mexico_1km.tif"
    elevation = reproject_elevation(dem_path, grid)
    land_mask = land_mask_from_elevation(elevation)
    lons, lats = grid.lon_lat_1d()
    basin_mask = region_defs.region_mask(region_name, lons, lats)

    records: list[dict] = []
    for count, day in enumerate(_iter_days(start, end)):
        if limit is not None and count >= limit:
            break
        record = build_day(
            region_name, day,
            v2_raw_root=v2_raw_root, output_root=output_root,
            elevation=elevation, land_mask=land_mask, basin_mask=basin_mask,
        )
        records.append(record)
        if (count + 1) % 200 == 0:
            accepted = sum(r["accepted"] for r in records)
            logger.info("build progress region=%s processed=%d accepted=%d", region_name, count + 1, accepted)

    accepted_count = sum(r["accepted"] for r in records)
    logger.info("build complete region=%s total=%d accepted=%d rejected=%d",
                region_name, len(records), accepted_count, len(records) - accepted_count)

    metadata_dir = output_root.parent / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    import csv

    index_path = metadata_dir / f"dataset_index_hourly_{region_name}.csv"
    fieldnames = sorted({key for record in records for key in record})
    with index_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({k: record.get(k, "") for k in fieldnames})

    manifest = write_manifest(
        metadata_dir / f"manifest_hourly_{region_name}.json",
        kind="build_hourly",
        records=records,
        extra={
            "region": region_defs.region_summary(region_name),
            "grid": {"height": grid.height, "width": grid.width, "transform": list(grid.transform), "crs": grid.crs},
            "preprocessing_version": HOURLY_DATASET_VERSION,
            "accepted": accepted_count,
            "rejected": len(records) - accepted_count,
        },
    )
    return manifest


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, choices=region_defs.list_regions())
    parser.add_argument("--start", type=_date.fromisoformat, default=None)
    parser.add_argument("--end", type=_date.fromisoformat, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    build_region(args.region, start=args.start, end=args.end, limit=args.limit)


if __name__ == "__main__":
    main()
