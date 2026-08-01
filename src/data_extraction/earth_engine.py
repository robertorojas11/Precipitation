"""Earth Engine initialization and Oya daily aggregation."""

from __future__ import annotations

import os

import ee

from src.utils.config import Config


DOMAIN_COORDINATES = [[
    [-133.471, 18.626],
    [-124.199, 33.753],
    [-61.429, 32.525],
    [-38.653, 29.721],
    [-38.653, 18.490],
    [-38.689, 5.288],
    [-53.015, 5.069],
    [-80.666, 4.970],
    [-117.667, 5.101],
]]
DOMAIN_POLYGON = None


def initialize() -> None:
    """Initialize Earth Engine, preferring user OAuth for Drive exports."""
    global DOMAIN_POLYGON
    service_account_path = os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
    try:
        try:
            ee.Initialize(project=Config.PROJECT_ID)
        except Exception as user_error:
            if not Config.SERVICE_ACCOUNT_FILE:
                raise RuntimeError("Earth Engine user authentication failed") from user_error
            credentials = ee.ServiceAccountCredentials("", Config.SERVICE_ACCOUNT_FILE)
            ee.Initialize(credentials, project=Config.PROJECT_ID)
    finally:
        if service_account_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path
    DOMAIN_POLYGON = ee.Geometry.Polygon(DOMAIN_COORDINATES)


def aggregate_oya_day(collection, date: str, minimum_valid_slots: int = 30):
    """Convert half-hourly mm/hour rates to masked daily millimetres."""
    start = ee.Date(date)
    daily = collection.filterDate(start, start.advance(1, "day"))
    slot_count = daily.select("precipitation").count().rename("slot_count")
    precipitation = (
        daily.select("precipitation").mean().multiply(24.0).rename("precipitation")
    )
    valid = slot_count.gte(minimum_valid_slots)
    return (
        precipitation.updateMask(valid)
        .addBands(slot_count.updateMask(valid))
        .set("system:time_start", start.millis())
    )
