"""Earth Engine acquisition modules (era5_ee, imerg) and the shared Drive
sync helper: path construction, manifest round-trips, and the flat-filename
parsing that makes region-scoped Drive exports resolvable -- no real Earth
Engine or Drive network calls."""

from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.data_extraction import drive, era5_ee, export_oya, imerg
from src.data_preprocessing import regions
from src.utils import provenance


class ParsePrefixedExportNameTest(unittest.TestCase):
    def test_simple_region_name(self):
        region, date_str, year, month = drive.parse_prefixed_export_name(
            "era5_mexico_pacifico_2020-01-05.tif", "era5"
        )
        self.assertEqual((region, date_str, year, month), ("mexico_pacifico", "2020-01-05", "2020", "01"))

    def test_region_name_with_multiple_underscores(self):
        region, date_str, year, month = drive.parse_prefixed_export_name(
            "imerg_puerto_rico_caribe_2021-12-31.tif", "imerg"
        )
        self.assertEqual(region, "puerto_rico_caribe")
        self.assertEqual((date_str, year, month), ("2021-12-31", "2021", "12"))

    def test_wrong_prefix_raises(self):
        with self.assertRaises(ValueError):
            drive.parse_prefixed_export_name("oya_mexico_pacifico_2020-01-05.tif", "era5")

    def test_missing_region_segment_raises(self):
        with self.assertRaises(ValueError):
            drive.parse_prefixed_export_name("era5_2020-01-05.tif", "era5")


class Era5EePathAndUnitsTest(unittest.TestCase):
    def test_rate_to_hourly_mm(self):
        # 1 kg/m^2/s over one hour = 3600 mm accumulated.
        self.assertAlmostEqual(era5_ee.rate_to_hourly_mm(1.0), 3600.0)
        self.assertAlmostEqual(era5_ee.rate_to_hourly_mm(0.0), 0.0)

    def test_local_target_matches_drive_export_naming(self):
        date = datetime(2020, 3, 5)
        local = era5_ee._local_target(Path("/root"), "mexico_pacifico", date)
        self.assertEqual(local, Path("/root/era5_ee/mexico_pacifico/2020/03/era5_mexico_pacifico_2020-03-05.tif"))
        # Round-trip: the flat Drive filename this module would submit,
        # parsed back, must reconstruct the same region/year/month.
        drive_name = f"era5_mexico_pacifico_{date:%Y-%m-%d}.tif"
        region, date_str, year, month = drive.parse_prefixed_export_name(drive_name, "era5")
        self.assertEqual(
            local,
            Path("/root") / "era5_ee" / region / year / month / drive_name,
        )

    def test_download_range_rejects_unknown_region(self):
        with self.assertRaises(KeyError):
            era5_ee.download_range(2020, 2020, "atlantis")


class ImergPathAndReductionTest(unittest.TestCase):
    def test_local_target_matches_drive_export_naming(self):
        date = datetime(2021, 6, 15)
        local = imerg._local_target(Path("/root"), "puerto_rico_caribe", date)
        self.assertEqual(local, Path("/root/imerg/puerto_rico_caribe/2021/06/imerg_puerto_rico_caribe_2021-06-15.tif"))

    def test_halfhourly_to_hourly_averages_pairs(self):
        slots = np.zeros((48, 2, 2), dtype=np.float32)
        slots[0] = 2.0
        slots[1] = 4.0  # hour 0 mean -> 3.0
        hourly = imerg.halfhourly_to_hourly(slots)
        self.assertEqual(hourly.shape, (24, 2, 2))
        np.testing.assert_allclose(hourly[0], 3.0)
        np.testing.assert_allclose(hourly[1], 0.0)

    def test_halfhourly_to_hourly_rejects_wrong_slot_count(self):
        with self.assertRaises(ValueError):
            imerg.halfhourly_to_hourly(np.zeros((47, 2, 2)))

    def test_read_daily_geotiff_hourly_roundtrip(self):
        import rasterio
        from rasterio.transform import from_origin

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "imerg_test.tif"
            data = np.zeros((48, 3, 3), dtype=np.float32)
            data[0] = 1.0
            data[1] = 3.0  # hour 0 -> mean 2.0
            with rasterio.open(
                path, "w", driver="GTiff", height=3, width=3, count=48,
                dtype="float32", crs="EPSG:4326", transform=from_origin(0, 0, 0.1, 0.1),
            ) as dst:
                dst.write(data)
            hourly = imerg.read_daily_geotiff_hourly(path)
            self.assertEqual(hourly.shape, (24, 3, 3))
            np.testing.assert_allclose(hourly[0], 2.0)

    def test_download_range_rejects_unknown_region(self):
        with self.assertRaises(KeyError):
            imerg.download_range(2020, 2020, "atlantis")


class DownloadRangeEndToEndTest(unittest.TestCase):
    """Exercise download_range()'s resume/manifest logic with injected
    submit/sync callables and a mocked Earth Engine initialize()."""

    def test_era5_resumes_and_writes_manifest(self):
        region = "mexico_pacifico"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = era5_ee._local_target(root, region, datetime(2020, 1, 1))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fake-geotiff")

            submit_calls = []
            with patch.object(era5_ee.earth_engine, "initialize", return_value=None):
                manifest_path = era5_ee.download_range(
                    2020, 2020, region, root=root,
                    submit_in_batches=lambda fns: submit_calls.append(len(fns)),
                    sync=lambda: 0,
                )
            manifest = provenance.load_manifest(manifest_path)
            self.assertEqual(manifest["kind"], "era5_ee")
            # Only 2020-01-01 is pre-existing; the other 365 days of 2020
            # should have been queued for submission.
            self.assertEqual(submit_calls[0], 30)  # January minus the 1 existing day
            self.assertTrue(any(rec["day"] == "2020-01-01" for rec in manifest["records"]))

    def test_imerg_skips_submission_when_month_already_complete(self):
        import calendar

        region = "mexico_atlantico_golfo"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for day in range(1, calendar.monthrange(2020, 1)[1] + 1):
                target = imerg._local_target(root, region, datetime(2020, 1, day))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"fake-geotiff")

            submit_calls = []
            with patch.object(imerg.earth_engine, "initialize", return_value=None):
                imerg.download_range(
                    2020, 2020, region, root=root,
                    submit_in_batches=lambda fns: submit_calls.append(len(fns)),
                    sync=lambda: 0,
                )
            # January is fully present -> no submission for it; only later
            # months (Feb..Dec) get queued.
            self.assertEqual(len(submit_calls), 11)


class OyaRegionScopedExportNamingTest(unittest.TestCase):
    """The region-scoped Oya export must use a flat Drive filename (PASO 3
    fix, 2026-09-21) so drive.sync_prefixed can parse it back correctly."""

    def test_local_target_and_flat_drive_name_round_trip(self):
        date = datetime(2020, 3, 5)
        region = "mexico_pacifico"
        local = export_oya._local_target(Path("/root"), date, region)
        drive_name = f"oya_{region}_{date:%Y-%m-%d}.tif"
        parsed_region, date_str, year, month = drive.parse_prefixed_export_name(drive_name, "oya")
        self.assertEqual(local, Path("/root") / "oya" / parsed_region / year / month / drive_name)

    def test_unscoped_export_path_unaffected(self):
        date = datetime(2020, 3, 5)
        local = export_oya._local_target(Path("/root"), date, None)
        self.assertEqual(local, Path("/root/oya/2020/03/oya_2020-03-05.tif"))


if __name__ == "__main__":
    unittest.main()
