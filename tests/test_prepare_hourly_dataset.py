"""Tests for the prepare stage (RainShift M216-M219 recipe --
src/data_preprocessing/prepare_hourly_dataset.py), using small synthetic
built datasets (no real acquired/build data required)."""

from datetime import date
from pathlib import Path
import tempfile
import unittest

import numpy as np
from rasterio.transform import from_origin

from src.data_preprocessing import build_hourly_dataset as B
from src.data_preprocessing import prepare_hourly_dataset as P
from src.data_preprocessing import regions
from tests.test_build_hourly_dataset import _write_geotiff, _era5_descriptions


class LogTransformTest(unittest.TestCase):
    def test_matches_rainshift_formula(self):
        x = np.array([0.0, 1.0, 10.0])
        expected = np.log(np.maximum(x, 0.0) * 1000.0 + 1e-5)
        np.testing.assert_allclose(P.log_transform(x), expected)

    def test_negative_input_is_clamped_to_zero_first(self):
        self.assertAlmostEqual(P.log_transform(np.array([-5.0]))[0], P.log_transform(np.array([0.0]))[0])

    def test_round_trip(self):
        x = np.array([0.0, 0.5, 3.7, 100.0])
        recovered = P.inverse_log_transform(P.log_transform(x))
        np.testing.assert_allclose(recovered, x, atol=1e-6)


class PooledStatsTest(unittest.TestCase):
    def test_matches_rainshift_m219_formula_by_hand(self):
        # Two regions, one channel, hand-computed expected pooled values.
        region_a = {"target": P.ChannelMoments(mean=2.0, std=1.0, count=100)}
        region_b = {"target": P.ChannelMoments(mean=4.0, std=2.0, count=50)}
        pooled = P.pooled_stats({"a": region_a, "b": region_b})

        expected_mu = (2.0 + 4.0) / 2
        expected_var = ((1.0 ** 2 + 2.0 ** 2) + (2.0 ** 2 + 4.0 ** 2)) / 2 - expected_mu ** 2
        self.assertAlmostEqual(pooled["target"].mean, expected_mu, places=10)
        self.assertAlmostEqual(pooled["target"].std, expected_var ** 0.5, places=10)
        self.assertEqual(pooled["target"].count, 150)

    def test_equal_regions_reduce_to_their_common_value(self):
        same = {"x": P.ChannelMoments(mean=5.0, std=1.5, count=10)}
        pooled = P.pooled_stats({"a": same, "b": dict(same)})
        self.assertAlmostEqual(pooled["x"].mean, 5.0, places=10)
        self.assertAlmostEqual(pooled["x"].std, 1.5, places=10)

    def test_requires_at_least_one_region(self):
        with self.assertRaises(ValueError):
            P.pooled_stats({})


class _WithTwoSyntheticRegions(unittest.TestCase):
    """Registers two small synthetic regions (backed by the same synthetic
    basin polygon) so pooled statistics across regions are testable
    offline -- mirrors tests/test_build_hourly_dataset.py's approach."""

    REGION_A = "_test_prepare_region_a"
    REGION_B = "_test_prepare_region_b"

    def setUp(self):
        try:
            import geopandas
            from shapely.geometry import box
        except ImportError:
            raise unittest.SkipTest("geopandas is not installed")

        self._tmpdir = tempfile.TemporaryDirectory()
        shp_path = Path(self._tmpdir.name) / "square_basin.shp"
        square = box(-90.0, 10.0, -70.0, 30.0)
        gdf = geopandas.GeoDataFrame({"geometry": [square]}, crs="EPSG:4326")
        gdf.to_file(shp_path)

        regions._basin_geometry.cache_clear()
        self._original_atlantico = regions.Config.ATLANTICO_SHP_PATH
        regions.Config.ATLANTICO_SHP_PATH = str(shp_path)

        regions.REGISTRY[self.REGION_A] = regions.Region(
            name=self.REGION_A, basin="atlantico",
            lon_min=-80.0, lon_max=-78.0, lat_min=18.0, lat_max=20.0,
            role="train_eval", description="synthetic region A",
        )
        regions.REGISTRY[self.REGION_B] = regions.Region(
            name=self.REGION_B, basin="atlantico",
            lon_min=-76.0, lon_max=-74.0, lat_min=18.0, lat_max=20.0,
            role="train_eval", description="synthetic region B",
        )

    def tearDown(self):
        regions.Config.ATLANTICO_SHP_PATH = self._original_atlantico
        regions.REGISTRY.pop(self.REGION_A, None)
        regions.REGISTRY.pop(self.REGION_B, None)
        regions._basin_geometry.cache_clear()
        self._tmpdir.cleanup()

    def _build_one_day(self, raw_root: Path, region: str, day: date, *, cp, lsp, imerg, elevation=50.0):
        # Source rasters must actually cover the target region's own bbox --
        # region_defs.get_region(region).lon_min/lat_max is the top-left
        # corner every region uses here, so anchor each synthetic source
        # there instead of a single hardcoded origin shared by both regions.
        region_def = regions.get_region(region)
        origin_lon, origin_lat = region_def.lon_min - 0.2, region_def.lat_max + 0.2

        era5_transform = from_origin(origin_lon, origin_lat, 0.25, 0.25)
        era5_data = np.empty((48, 10, 10), dtype=np.float32)
        era5_data[0::2] = cp
        era5_data[1::2] = lsp
        _write_geotiff(
            B._source_path(raw_root, "era5", region, day), era5_data, era5_transform,
            descriptions=_era5_descriptions(),
        )
        imerg_transform = from_origin(origin_lon, origin_lat, 0.1, 0.1)
        imerg_data = np.full((48, 22, 22), imerg, dtype=np.float32)
        _write_geotiff(B._source_path(raw_root, "imerg", region, day), imerg_data, imerg_transform)


class ComputeStatsTest(_WithTwoSyntheticRegions):
    def test_pooled_stats_over_two_regions_with_known_constants(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "raw"
            build_root = Path(tmp) / "build"

            # Region A: cp=1.0, lsp=0.5, imerg=4.0 (constant every hour/pixel).
            self._build_one_day(raw_root, self.REGION_A, date(2015, 6, 1), cp=1.0, lsp=0.5, imerg=4.0)
            # Region B: different constants.
            self._build_one_day(raw_root, self.REGION_B, date(2015, 6, 1), cp=2.0, lsp=1.5, imerg=8.0)

            for region in (self.REGION_A, self.REGION_B):
                B.build_region(region, v2_raw_root=raw_root, output_root=build_root / "processed",
                                start=date(2015, 6, 1), end=date(2015, 6, 1))

            stats = P.compute_stats((self.REGION_A, self.REGION_B), build_root=build_root)

            lo, hi = stats["era5_clip_range_imerg_mm_per_hr"]
            self.assertAlmostEqual(lo, 4.0, delta=1e-3)  # min of {4.0, 8.0}... region-pooled range
            self.assertAlmostEqual(hi, 8.0, delta=1e-3)

            # Region A's cp=1.0 clipped into [4.0, 8.0] -> becomes 4.0 before the log.
            expected_a_cp = P.log_transform(np.array([4.0]))[0]
            self.assertAlmostEqual(stats["per_region"][self.REGION_A]["cp"]["mean"], expected_a_cp, places=3)

            # Pooled mean is the average of the two per-region means (equal region weight).
            expected_pooled_target = (
                stats["per_region"][self.REGION_A]["target"]["mean"]
                + stats["per_region"][self.REGION_B]["target"]["mean"]
            ) / 2
            self.assertAlmostEqual(stats["pooled"]["target"]["mean"], expected_pooled_target, places=6)


class PrepareRegionTest(_WithTwoSyntheticRegions):
    def test_prepared_output_shapes_and_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "raw"
            build_root = Path(tmp) / "build"
            prepared_root = Path(tmp) / "prepared"

            self._build_one_day(raw_root, self.REGION_A, date(2015, 6, 1), cp=1.0, lsp=0.5, imerg=4.0)
            self._build_one_day(raw_root, self.REGION_B, date(2015, 6, 1), cp=2.0, lsp=1.5, imerg=8.0)
            for region in (self.REGION_A, self.REGION_B):
                B.build_region(region, v2_raw_root=raw_root, output_root=build_root / "processed",
                                start=date(2015, 6, 1), end=date(2015, 6, 1))

            stats = P.compute_stats((self.REGION_A, self.REGION_B), build_root=build_root)
            manifest = P.prepare_region(self.REGION_A, stats, build_root=build_root, output_root=prepared_root)

            self.assertEqual(manifest["record_count"], 1)
            npz_path = Path(manifest["records"][0]["path"])
            self.assertTrue(npz_path.exists())
            with np.load(npz_path) as data:
                grid = B.region_grid(self.REGION_A)
                self.assertEqual(data["inputs_norm"].shape, (24, *grid.shape, 2))
                self.assertEqual(data["target_norm"].shape, (24, *grid.shape, 1))
                self.assertEqual(data["invariants_norm"].shape, (*grid.shape, 2))

                # Round-trip: de-normalize + inverse-log an interior valid
                # pixel and recover the known constant IMERG value (4.0).
                pooled_target = P.ChannelMoments(**stats["pooled"]["target"])
                valid = data["target_valid_mask"]
                idx = tuple(np.argwhere(valid)[len(np.argwhere(valid)) // 2])
                normalized = data["target_norm"][idx][0]
                log_value = normalized * pooled_target.std + pooled_target.mean
                recovered = P.inverse_log_transform(np.array([log_value]))[0]
                self.assertAlmostEqual(recovered, 4.0, delta=1e-2)

    def test_zero_shot_region_can_be_prepared_with_training_stats(self):
        """A region that never contributed to compute_stats() (e.g.
        puerto_rico_caribe in production) must still be preparable with the
        shared pooled stats -- this is the whole point of RainShift's
        geographic-generalization setup."""
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "raw"
            build_root = Path(tmp) / "build"
            prepared_root = Path(tmp) / "prepared"

            self._build_one_day(raw_root, self.REGION_A, date(2015, 6, 1), cp=1.0, lsp=0.5, imerg=4.0)
            self._build_one_day(raw_root, self.REGION_B, date(2020, 1, 1), cp=3.0, lsp=2.0, imerg=6.0)
            B.build_region(self.REGION_A, v2_raw_root=raw_root, output_root=build_root / "processed",
                            start=date(2015, 6, 1), end=date(2015, 6, 1))
            B.build_region(self.REGION_B, v2_raw_root=raw_root, output_root=build_root / "processed",
                            start=date(2020, 1, 1), end=date(2020, 1, 1))  # a "val" year -> not used for stats

            stats = P.compute_stats((self.REGION_A,), build_root=build_root)  # A only "trains"
            manifest = P.prepare_region(self.REGION_B, stats, build_root=build_root, output_root=prepared_root)
            self.assertEqual(manifest["record_count"], 1)
            self.assertTrue(Path(manifest["records"][0]["path"]).exists())


if __name__ == "__main__":
    unittest.main()
