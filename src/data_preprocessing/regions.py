"""Named training/evaluation regions for the v2 generative framework.

Regions are bounding boxes intersected with the tropical-cyclone basin
polygons at ``ATLANTICO_SHP_PATH`` / ``PACIFICO_SHP_PATH`` (see
``docs/DATA_SOURCES.md`` for what those polygons actually represent -- they
are whole-basin "area of responsibility" shapes, not coastal or
administrative boundaries, so a bounding box alone would let inland pixels
leak into a coastal region).

This module is the single source of truth for "which pixels belong to
region X" used by acquisition, preprocessing, training, and evaluation
(PASO 2 of docs/decisions/0002-implementacion-v2.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np

from src.utils.config import Config

Basin = Literal["atlantico", "pacifico"]
Role = Literal["train", "eval", "train_eval"]


@dataclass(frozen=True)
class Region:
    """A named bounding box scoped to one tropical-cyclone basin polygon."""

    name: str
    basin: Basin
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    role: Role
    description: str

    def __post_init__(self) -> None:
        if self.lon_min >= self.lon_max or self.lat_min >= self.lat_max:
            raise ValueError(f"Region {self.name!r} has a degenerate bounding box")

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """``(lon_min, lat_min, lon_max, lat_max)``."""
        return (self.lon_min, self.lat_min, self.lon_max, self.lat_max)


REGISTRY: dict[str, Region] = {
    "mexico_pacifico": Region(
        name="mexico_pacifico",
        basin="pacifico",
        lon_min=-120.0,
        lon_max=-104.0,
        lat_min=12.0,
        lat_max=32.0,
        role="train_eval",
        description=(
            "Pacific coast of Mexico (Baja California to Chiapas), "
            "intersected with the Eastern Pacific hurricane-basin polygon "
            "at PACIFICO_SHP_PATH."
        ),
    ),
    "mexico_atlantico_golfo": Region(
        name="mexico_atlantico_golfo",
        basin="atlantico",
        lon_min=-98.0,
        lon_max=-84.0,
        lat_min=15.0,
        lat_max=30.0,
        role="train_eval",
        description=(
            "Gulf of Mexico and Caribbean coast of Mexico (Tamaulipas to "
            "Quintana Roo), intersected with the Atlantic hurricane-basin "
            "polygon at ATLANTICO_SHP_PATH."
        ),
    ),
    "puerto_rico_caribe": Region(
        name="puerto_rico_caribe",
        basin="atlantico",
        lon_min=-85.0,
        lon_max=-59.0,
        lat_min=10.0,
        lat_max=23.0,
        role="eval",
        description=(
            "Greater and Lesser Antilles arc (Puerto Rico, Hispaniola, "
            "eastern Cuba, Lesser Antilles), intersected with the Atlantic "
            "hurricane-basin polygon. Not a coastal/administrative "
            "boundary -- a working bounding box confirmed with the project "
            "owner on 2026-09-11 "
            "(docs/decisions/0002-implementacion-v2.md, section 1.4)."
        ),
    ),
}


def list_regions() -> list[str]:
    return sorted(REGISTRY)


def get_region(name: str) -> Region:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown region {name!r}; known regions: {list_regions()}") from exc


def _basin_shapefile_path(basin: Basin) -> str:
    path = {"atlantico": Config.ATLANTICO_SHP_PATH, "pacifico": Config.PACIFICO_SHP_PATH}[basin]
    if not path:
        raise RuntimeError(
            f"{basin.upper()}_SHP_PATH is not set; required to build the {basin} basin mask."
        )
    return path


@lru_cache(maxsize=None)
def _basin_geometry(basin: Basin):
    """Load and cache one basin polygon (dissolved to a single geometry)."""
    import geopandas as gpd

    path = _basin_shapefile_path(basin)
    if not Path(path).exists():
        raise FileNotFoundError(f"{basin} basin shapefile not found at {path}")
    gdf = gpd.read_file(path)
    if gdf.crs is not None and str(gdf.crs).upper() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    return gdf.union_all() if hasattr(gdf, "union_all") else gdf.unary_union


def region_mask(
    name: str,
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    require_basin: bool = True,
) -> np.ndarray:
    """Boolean mask over a ``(len(lat), len(lon))`` grid for region ``name``.

    ``lon``/``lat`` are 1-D coordinate arrays, as produced by the existing
    rasterio-based grid builders in ``build_dataset.py``. When
    ``require_basin`` is True (default), a pixel must additionally fall
    inside the region's basin polygon, not merely inside its bounding box --
    this is what keeps inland pixels out of a coastal region.
    """
    import shapely

    region = get_region(name)
    lon_grid, lat_grid = np.meshgrid(np.asarray(lon), np.asarray(lat))
    bbox_mask = (
        (lon_grid >= region.lon_min)
        & (lon_grid <= region.lon_max)
        & (lat_grid >= region.lat_min)
        & (lat_grid <= region.lat_max)
    )
    if not require_basin:
        return bbox_mask

    geometry = _basin_geometry(region.basin)
    basin_mask = shapely.contains_xy(geometry, lon_grid, lat_grid)
    return bbox_mask & basin_mask


def region_summary(name: str) -> dict:
    """JSON-serializable metadata for manifests/docs (no geopandas import)."""
    region = get_region(name)
    return {
        "name": region.name,
        "basin": region.basin,
        "bbox": list(region.bbox),
        "role": region.role,
        "description": region.description,
    }
