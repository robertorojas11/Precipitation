import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data_preprocessing import regions


class RegionRegistryTests(unittest.TestCase):
    def test_known_regions_are_registered(self):
        self.assertEqual(
            regions.list_regions(),
            ["mexico_atlantico_golfo", "mexico_pacifico", "puerto_rico_caribe"],
        )

    def test_unknown_region_raises(self):
        with self.assertRaises(KeyError):
            regions.get_region("atlantis")

    def test_degenerate_bbox_rejected(self):
        with self.assertRaises(ValueError):
            regions.Region(
                name="bad",
                basin="atlantico",
                lon_min=0.0,
                lon_max=0.0,
                lat_min=0.0,
                lat_max=1.0,
                role="eval",
                description="",
            )

    def test_region_summary_is_json_serializable_shape(self):
        summary = regions.region_summary("puerto_rico_caribe")
        self.assertEqual(summary["basin"], "atlantico")
        self.assertEqual(len(summary["bbox"]), 4)
        self.assertEqual(summary["role"], "eval")


class RegionMaskWithSyntheticGeometryTests(unittest.TestCase):
    """Exercise region_mask against a simple synthetic square polygon,
    instead of the real (network-mounted) basin shapefiles."""

    def setUp(self):
        geopandas = self._require_geopandas()
        from shapely.geometry import box

        self._tmpdir = tempfile.TemporaryDirectory()
        shp_path = Path(self._tmpdir.name) / "square_basin.shp"
        # A simple 10x10 degree square, well inside the test region's bbox.
        square = box(-80.0, 10.0, -70.0, 20.0)
        gdf = geopandas.GeoDataFrame({"geometry": [square]}, crs="EPSG:4326")
        gdf.to_file(shp_path)

        regions._basin_geometry.cache_clear()
        self._original_atlantico = regions.Config.ATLANTICO_SHP_PATH
        regions.Config.ATLANTICO_SHP_PATH = str(shp_path)

        regions.REGISTRY["_test_region"] = regions.Region(
            name="_test_region",
            basin="atlantico",
            lon_min=-90.0,
            lon_max=-60.0,
            lat_min=0.0,
            lat_max=30.0,
            role="eval",
            description="synthetic test region",
        )

    def tearDown(self):
        regions.Config.ATLANTICO_SHP_PATH = self._original_atlantico
        regions.REGISTRY.pop("_test_region", None)
        regions._basin_geometry.cache_clear()
        self._tmpdir.cleanup()

    @staticmethod
    def _require_geopandas():
        try:
            import geopandas
        except ImportError:  # pragma: no cover - environment guard
            raise unittest.SkipTest("geopandas is not installed")
        return geopandas

    def test_bbox_mask_ignores_basin_polygon(self):
        lon = np.linspace(-90.0, -60.0, 7)
        lat = np.linspace(0.0, 30.0, 7)
        mask = regions.region_mask("_test_region", lon, lat, require_basin=False)
        self.assertEqual(mask.shape, (7, 7))
        self.assertTrue(mask.all())

    def test_basin_mask_excludes_points_outside_polygon(self):
        lon = np.array([-85.0, -75.0, -65.0])
        lat = np.array([15.0, 15.0, 15.0])
        mask = regions.region_mask("_test_region", lon, lat, require_basin=True)
        # square_basin spans lon [-80,-70], lat [10,20]: only the middle
        # point (-75, 15) should be inside.
        np.testing.assert_array_equal(mask[0], [False, True, False])

    def test_missing_shapefile_env_var_raises(self):
        regions.Config.ATLANTICO_SHP_PATH = ""
        regions._basin_geometry.cache_clear()
        with self.assertRaises(RuntimeError):
            regions.region_mask("_test_region", np.array([-75.0]), np.array([15.0]))


if __name__ == "__main__":
    unittest.main()
