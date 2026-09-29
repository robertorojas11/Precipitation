"""Tests for the hourly ERA5->IMERG build/alignment stage
(src/data_preprocessing/build_hourly_dataset.py), using small synthetic
GeoTIFFs -- no real acquired data required."""

from datetime import date
from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.data_preprocessing import build_hourly_dataset as B
from src.data_preprocessing import regions


def _era5_descriptions() -> list[str]:
    out = []
    for hour in range(24):
        out.append(f"202001{15:02d}T{hour:02d}_mean_convective_precipitation_rate")
        out.append(f"202001{15:02d}T{hour:02d}_mean_large_scale_precipitation_rate")
    return out


def _write_geotiff(path: Path, data: np.ndarray, transform, descriptions=None) -> None:
    """``data``: (bands, H, W) float32."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2],
        count=data.shape[0], dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(data)
        if descriptions:
            for i, description in enumerate(descriptions, start=1):
                dst.set_band_description(i, description)


class RegionGridTest(unittest.TestCase):
    def test_shape_matches_bbox_and_resolution(self):
        grid = B.region_grid("mexico_pacifico", resolution_deg=0.1)
        region = regions.get_region("mexico_pacifico")
        expected_width = round((region.lon_max - region.lon_min) / 0.1)
        expected_height = round((region.lat_max - region.lat_min) / 0.1)
        self.assertEqual(grid.shape, (expected_height, expected_width))

    def test_lon_lat_1d_matches_rasterio_transform_xy(self):
        grid = B.region_grid("puerto_rico_caribe", resolution_deg=0.5)
        lons, lats = grid.lon_lat_1d()
        # Spot-check against rasterio's own pixel-center computation.
        expected_x, expected_y = rasterio.transform.xy(grid.transform, [0, 2], [0, 3])
        self.assertAlmostEqual(lons[0], expected_x[0], places=6)
        self.assertAlmostEqual(lons[3], expected_x[1], places=6)
        self.assertAlmostEqual(lats[0], expected_y[0], places=6)
        self.assertAlmostEqual(lats[2], expected_y[1], places=6)


class ParseEra5HourBandsTest(unittest.TestCase):
    def test_valid_descriptions_map_every_hour(self):
        mapping = B.parse_era5_hour_bands(tuple(_era5_descriptions()))
        self.assertEqual(len(mapping), 24)
        self.assertEqual(mapping[0], {"cp": 0, "lsp": 1})
        self.assertEqual(mapping[23], {"cp": 46, "lsp": 47})

    def test_missing_hour_raises(self):
        descriptions = _era5_descriptions()[:46]  # drop hour 23
        with self.assertRaises(ValueError):
            B.parse_era5_hour_bands(tuple(descriptions))

    def test_malformed_description_raises(self):
        with self.assertRaises(ValueError):
            B.parse_era5_hour_bands(("not_a_real_band_name",))

    def test_missing_description_raises(self):
        with self.assertRaises(ValueError):
            B.parse_era5_hour_bands((None,))


class LandMaskTest(unittest.TestCase):
    def test_positive_elevation_is_land(self):
        elevation = np.array([[-10.0, 0.0, 5.0], [100.0, -1.0, 0.0]], dtype=np.float32)
        mask = B.land_mask_from_elevation(elevation, threshold=0.0)
        np.testing.assert_array_equal(mask, [[False, False, True], [True, False, False]])


class YearSplitTest(unittest.TestCase):
    def test_train_val_test_boundaries(self):
        self.assertEqual(B.year_split(2010), "train")
        self.assertEqual(B.year_split(2019), "train")
        self.assertEqual(B.year_split(2020), "val")
        self.assertEqual(B.year_split(2021), "val")
        self.assertEqual(B.year_split(2022), "test")
        self.assertEqual(B.year_split(2023), "test")

    def test_out_of_range_year_raises(self):
        with self.assertRaises(ValueError):
            B.year_split(2024)
        with self.assertRaises(ValueError):
            B.year_split(2009)


class _WithSyntheticRegion(unittest.TestCase):
    """Registers a small (~2x2 deg) test region backed by a synthetic basin
    polygon, so build_day/build_region can run fast and offline -- mirrors
    tests/test_regions.py's approach."""

    REGION_NAME = "_test_build_region"

    def setUp(self):
        try:
            import geopandas
            from shapely.geometry import box
        except ImportError:
            raise unittest.SkipTest("geopandas is not installed")

        self._tmpdir = tempfile.TemporaryDirectory()
        shp_path = Path(self._tmpdir.name) / "square_basin.shp"
        square = box(-90.0, 10.0, -70.0, 30.0)  # comfortably covers the test region
        gdf = geopandas.GeoDataFrame({"geometry": [square]}, crs="EPSG:4326")
        gdf.to_file(shp_path)

        regions._basin_geometry.cache_clear()
        self._original_atlantico = regions.Config.ATLANTICO_SHP_PATH
        regions.Config.ATLANTICO_SHP_PATH = str(shp_path)

        regions.REGISTRY[self.REGION_NAME] = regions.Region(
            name=self.REGION_NAME, basin="atlantico",
            lon_min=-80.0, lon_max=-78.0, lat_min=18.0, lat_max=20.0,
            role="eval", description="synthetic test region for build_hourly_dataset",
        )

    def tearDown(self):
        regions.Config.ATLANTICO_SHP_PATH = self._original_atlantico
        regions.REGISTRY.pop(self.REGION_NAME, None)
        regions._basin_geometry.cache_clear()
        self._tmpdir.cleanup()


class BuildDayTest(_WithSyntheticRegion):
    def _write_sources(self, root: Path, day: date, *, cp_value=1.0, lsp_value=0.5, imerg_value=2.0):
        # Small source grids at coarser/finer resolution than the canonical
        # grid, deliberately not aligned to it -- build_day must reproject.
        era5_transform = from_origin(-80.2, 20.2, 0.25, 0.25)
        era5_data = np.empty((48, 10, 10), dtype=np.float32)
        era5_data[0::2] = cp_value
        era5_data[1::2] = lsp_value
        era5_path = B._source_path(root, "era5", self.REGION_NAME, day)
        _write_geotiff(era5_path, era5_data, era5_transform, descriptions=_era5_descriptions())

        imerg_transform = from_origin(-80.2, 20.2, 0.1, 0.1)
        imerg_data = np.full((48, 22, 22), imerg_value, dtype=np.float32)
        imerg_path = B._source_path(root, "imerg", self.REGION_NAME, day)
        _write_geotiff(imerg_path, imerg_data, imerg_transform)
        return era5_path, imerg_path

    def _common_kwargs(self, output_root):
        grid = B.region_grid(self.REGION_NAME)
        elevation = np.full(grid.shape, 50.0, dtype=np.float32)  # all land
        land_mask = B.land_mask_from_elevation(elevation)
        lons, lats = grid.lon_lat_1d()
        basin_mask = regions.region_mask(self.REGION_NAME, lons, lats)
        return dict(output_root=output_root, elevation=elevation, land_mask=land_mask, basin_mask=basin_mask)

    def test_accepted_day_writes_npz_with_expected_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            output_root = Path(tmp) / "processed"
            day = date(2015, 6, 1)  # a "train" year
            self._write_sources(root, day, cp_value=1.2, lsp_value=0.3, imerg_value=4.0)

            record = B.build_day(
                self.REGION_NAME, day, v2_raw_root=root,
                **self._common_kwargs(output_root),
            )

            self.assertTrue(record["accepted"])
            self.assertEqual(record["split"], "train")
            npz_path = Path(record["npz_path"])
            self.assertTrue(npz_path.exists())
            with np.load(npz_path) as payload:
                grid = B.region_grid(self.REGION_NAME)
                self.assertEqual(payload["inputs"].shape, (24, *grid.shape, 2))
                self.assertEqual(payload["target"].shape, (24, *grid.shape, 1))
                # Uniform source fields -> reprojected interior values match
                # the source constants (allow the resampling to touch edges).
                interior = payload["inputs"][0, 2:-2, 2:-2, 0]
                np.testing.assert_allclose(interior, 1.2, atol=1e-3)
                interior_target = payload["target"][0, 2:-2, 2:-2, 0]
                np.testing.assert_allclose(interior_target, 4.0, atol=1e-3)

    def test_missing_source_file_is_rejected_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            output_root = Path(tmp) / "processed"
            day = date(2015, 6, 2)
            record = B.build_day(
                self.REGION_NAME, day, v2_raw_root=root,
                **self._common_kwargs(output_root),
            )
            self.assertFalse(record["accepted"])
            self.assertEqual(record["reject_reasons"], ["missing_source_file"])

    def test_year_outside_split_ranges_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            output_root = Path(tmp) / "processed"
            day = date(2030, 1, 1)
            with self.assertRaises(ValueError):
                B.build_day(self.REGION_NAME, day, v2_raw_root=root, **self._common_kwargs(output_root))


class BuildRegionTest(_WithSyntheticRegion):
    def test_builds_index_and_manifest_over_a_date_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            output_root = Path(tmp) / "processed"
            start, end = date(2015, 6, 1), date(2015, 6, 3)

            era5_transform = from_origin(-80.2, 20.2, 0.25, 0.25)
            imerg_transform = from_origin(-80.2, 20.2, 0.1, 0.1)
            for offset in range(3):
                day = date(2015, 6, 1 + offset)
                era5_data = np.ones((48, 10, 10), dtype=np.float32)
                _write_geotiff(
                    B._source_path(root, "era5", self.REGION_NAME, day),
                    era5_data, era5_transform, descriptions=_era5_descriptions(),
                )
                imerg_data = np.full((48, 22, 22), 3.0, dtype=np.float32)
                _write_geotiff(
                    B._source_path(root, "imerg", self.REGION_NAME, day),
                    imerg_data, imerg_transform,
                )
            # Deliberately leave one day (2015-06-04) unwritten -- not part
            # of [start, end] here, so no rejection expected from it.

            manifest = B.build_region(
                self.REGION_NAME, v2_raw_root=root, output_root=output_root,
                start=start, end=end,
            )

            self.assertEqual(manifest["kind"], "build_hourly")
            self.assertEqual(manifest["record_count"], 3)
            self.assertEqual(manifest["accepted"], 3)
            self.assertEqual(manifest["rejected"], 0)

            metadata_dir = output_root.parent / "metadata"
            index_path = metadata_dir / f"dataset_index_hourly_{self.REGION_NAME}.csv"
            self.assertTrue(index_path.exists())
            manifest_path = metadata_dir / f"manifest_hourly_{self.REGION_NAME}.json"
            self.assertTrue(manifest_path.exists())


if __name__ == "__main__":
    unittest.main()
