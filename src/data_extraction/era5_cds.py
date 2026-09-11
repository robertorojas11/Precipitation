"""Acquire pure ERA5 convective + large-scale precipitation from the CDS API.

Extends this project's pre-existing ERA5-pure acquisition (the ``raw/era5_pl``
archive already used by ``build_dataset.py`` for the 850/500 hPa channels,
see ``docs/stages/data-and-resources.md``) to two additional single-level
variables, per PASO 3.1 of ``docs/decisions/0002-implementacion-v2.md``:

* ``convective_precipitation`` (cp)
* ``large_scale_precipitation`` (lsp)

at hourly resolution on the native ERA5 single-level grid (~0.25 deg), from
the ``reanalysis-era5-single-levels`` dataset of the Copernicus Climate Data
Store, scoped to one named region (``src.data_preprocessing.regions``).

Credentials are mandatory. The module fails with an explicit message naming
the missing environment variable and never falls back to synthetic data or
silent placeholders (transversal migration rule; matches the pattern already
established by ``src/utils/config.py`` / ``env.example`` for Earth Engine).

Delivery is one NetCDF per (variable, month), stored under::

    $V2_RAW_DATA_DIR/era5_cds/<region>/<var>/era5_cds_<var>_YYYY-MM.nc

A hashed manifest (``manifest_era5_cds_<region>.json``) records every file
with its SHA-256 and the CDS request that produced it. Re-running skips
months whose file already exists and is listed in the manifest (resumable).

Run from the repository root::

    python -m src.data_extraction.era5_cds --start 2001-01 --end 2022-12 --region mexico_pacifico
"""

from __future__ import annotations

import argparse
from calendar import monthrange
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from src.data_preprocessing import regions as region_defs
from src.utils.config import Config
from src.utils.logging import configure_logging
from src.utils.provenance import file_sha256, load_manifest, write_manifest

logger = configure_logging(name="precipitation.era5_cds")

CDS_DATASET = "reanalysis-era5-single-levels"
ERA5_VARIABLES = ("convective_precipitation", "large_scale_precipitation")
_ALL_HOURS = [f"{h:02d}:00" for h in range(24)]


class MissingCredentialsError(RuntimeError):
    """Raised when neither env vars nor ~/.cdsapirc provide CDS credentials."""


@dataclass(frozen=True)
class Month:
    year: int
    month: int

    def __str__(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    @property
    def days(self) -> list[str]:
        _, last = monthrange(self.year, self.month)
        return [f"{d:02d}" for d in range(1, last + 1)]


def _parse_month(text: str) -> Month:
    parts = text.split("-")
    if len(parts) != 2:
        raise ValueError(f"expected YYYY-MM, got {text!r}")
    return Month(int(parts[0]), int(parts[1]))


def _iter_months(start: Month, end: Month) -> Iterator[Month]:
    if (start.year, start.month) > (end.year, end.month):
        raise ValueError("--start must not be after --end")
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield Month(y, m)
        m += 1
        if m == 13:
            y, m = y + 1, 1


def cds_area(region: region_defs.Region) -> list[float]:
    """CDS ``area`` request field: ``[North, West, South, East]``."""
    lon_min, lat_min, lon_max, lat_max = region.bbox
    return [lat_max, lon_min, lat_min, lon_max]


def resolve_cds_credentials() -> tuple[str, str]:
    """Return ``(url, key)`` for the CDS API or raise ``MissingCredentialsError``.

    Priority: explicit ``CDSAPI_URL`` + ``CDSAPI_KEY`` env vars, then a
    ``~/.cdsapirc`` file (the format ``cdsapi`` reads natively).
    """
    if Config.CDSAPI_URL and Config.CDSAPI_KEY:
        return Config.CDSAPI_URL, Config.CDSAPI_KEY

    rc = Path.home() / ".cdsapirc"
    if rc.exists():
        values: dict[str, str] = {}
        for line in rc.read_text().splitlines():
            if ":" in line:
                name, _, val = line.partition(":")
                values[name.strip()] = val.strip()
        if values.get("url") and values.get("key"):
            return values["url"], values["key"]

    raise MissingCredentialsError(
        "No Copernicus CDS credentials found. Set CDSAPI_URL and CDSAPI_KEY in "
        "your environment (.env), or create ~/.cdsapirc. Get a key from "
        "https://cds.climate.copernicus.eu/ (Your profile -> API key). "
        "See env.example."
    )


def _cds_client():
    """Build a ``cdsapi.Client`` after checking credentials.

    ``cdsapi`` is imported lazily so importing this module (and running the
    unit tests, which inject a fake client) never requires the package.
    """
    url, key = resolve_cds_credentials()
    try:
        import cdsapi
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "The 'cdsapi' package is required for ERA5 acquisition. "
            "Install it with: ./venv/bin/pip install cdsapi"
        ) from exc
    return cdsapi.Client(url=url, key=key, quiet=True, wait_until_complete=True)


def _request_for(variable: str, month: Month, area: list[float]) -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": [variable],
        "year": [f"{month.year:04d}"],
        "month": [f"{month.month:02d}"],
        "day": month.days,
        "time": _ALL_HOURS,
        "area": area,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def _target_path(root: Path, region_name: str, variable: str, month: Month) -> Path:
    return root / "era5_cds" / region_name / variable / f"era5_cds_{variable}_{month}.nc"


def _manifest_path(root: Path, region_name: str) -> Path:
    return root / "era5_cds" / region_name / f"manifest_era5_cds_{region_name}.json"


def download_range(
    start: Month,
    end: Month,
    region: region_defs.Region,
    *,
    root: Path | None = None,
    client=None,
    dry_run: bool = False,
) -> Path:
    """Download every (variable, month) NetCDF in the range. Returns manifest path.

    Resumable: a month whose file already exists and is listed in the
    manifest is skipped. ``client`` may be injected for testing.
    """
    root = Path(root or Config.V2_RAW_DATA_DIR)
    area = cds_area(region)
    manifest_path = _manifest_path(root, region.name)
    previous = load_manifest(manifest_path) or {}
    known = {rec["path"]: rec for rec in previous.get("records", [])}

    if client is None and not dry_run:
        client = _cds_client()
    elif dry_run:
        resolve_cds_credentials()  # still verify credentials exist

    records: list[dict] = []
    for month in _iter_months(start, end):
        for variable in ERA5_VARIABLES:
            target = _target_path(root, region.name, variable, month)
            rel = str(target.relative_to(root))
            if target.exists() and rel in known and target.stat().st_size > 0:
                logger.info("skip %s (already present)", rel)
                records.append(known[rel])
                continue
            request = _request_for(variable, month, area)
            if dry_run:
                logger.info("would request %s -> %s", request["variable"], rel)
                records.append({
                    "path": rel, "variable": variable, "month": str(month),
                    "request": request, "sha256": None, "bytes": None,
                })
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".nc.part")
            logger.info("retrieve %s %s", variable, month)
            client.retrieve(CDS_DATASET, request, str(tmp))
            tmp.replace(target)
            records.append({
                "path": rel,
                "variable": variable,
                "month": str(month),
                "request": request,
                "sha256": file_sha256(target),
                "bytes": target.stat().st_size,
            })

    manifest = write_manifest(
        manifest_path,
        kind="era5_cds",
        records=records,
        extra={
            "dataset": CDS_DATASET,
            "region": region_defs.region_summary(region.name),
            "cds_area": area,
            "variables": list(ERA5_VARIABLES),
            "range": {"start": str(start), "end": str(end)},
        },
    )
    logger.info(
        "manifest %s (%d records, sha256=%s)",
        manifest_path, manifest["record_count"], manifest["manifest_sha256"][:12],
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="first month, YYYY-MM")
    parser.add_argument("--end", required=True, help="last month, YYYY-MM (inclusive)")
    parser.add_argument("--region", required=True, choices=region_defs.list_regions())
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    download_range(
        _parse_month(args.start),
        _parse_month(args.end),
        region_defs.get_region(args.region),
        root=args.root,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
