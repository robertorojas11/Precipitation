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


class _FakeRequest:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class _FakeFilesResource:
    """Minimal stand-in for ``service.files()`` capturing every ``list()``
    call's kwargs, so query construction and pagination are testable
    without a real Drive API."""

    def __init__(self, pages_by_token: dict):
        self.pages_by_token = pages_by_token
        self.list_calls: list[dict] = []

    def list(self, **kwargs):
        self.list_calls.append(kwargs)
        page = self.pages_by_token[kwargs.get("pageToken")]
        return _FakeRequest(page)


class _FakeService:
    def __init__(self, files_resource):
        self._files = files_resource

    def files(self):
        return self._files


class ListAllPaginationTest(unittest.TestCase):
    def test_follows_next_page_token_until_exhausted(self):
        files_resource = _FakeFilesResource({
            None: {"files": [{"id": "a", "name": "a.tif"}], "nextPageToken": "p2"},
            "p2": {"files": [{"id": "b", "name": "b.tif"}]},
        })
        service = _FakeService(files_resource)
        items = drive._list_all(service, "some query")
        self.assertEqual([i["id"] for i in items], ["a", "b"])
        self.assertEqual(len(files_resource.list_calls), 2)
        self.assertIsNone(files_resource.list_calls[0]["pageToken"])
        self.assertEqual(files_resource.list_calls[1]["pageToken"], "p2")

    def test_single_page_makes_one_call(self):
        files_resource = _FakeFilesResource({None: {"files": [{"id": "a", "name": "a.tif"}]}})
        service = _FakeService(files_resource)
        items = drive._list_all(service, "some query")
        self.assertEqual(len(items), 1)
        self.assertEqual(len(files_resource.list_calls), 1)


class ResilientSyncTest(unittest.TestCase):
    """A single file's download/trash failure must not crash the whole
    sync -- found in production, 2026-09-23: a transient read timeout on
    the trash step (not an HttpError) propagated past the old narrow
    ``except HttpError`` and killed an otherwise-healthy multi-day
    acquisition process."""

    def test_try_trash_swallows_non_http_exceptions(self):
        service = object()
        with patch.object(drive, "execute_with_retry", side_effect=TimeoutError("read timed out")):
            drive._try_trash(service, "file-id", "some_file.tif")  # must not raise

    def test_try_download_returns_false_and_does_not_raise_on_failure(self):
        with patch.object(drive, "_download", side_effect=TimeoutError("read timed out")):
            ok = drive._try_download(object(), "file-id", Path("/tmp/never-written.tif"), "some_file.tif")
        self.assertFalse(ok)

    def test_try_download_returns_true_on_success(self):
        with patch.object(drive, "_download", return_value=None):
            ok = drive._try_download(object(), "file-id", Path("/tmp/whatever.tif"), "some_file.tif")
        self.assertTrue(ok)

    def test_sync_prefixed_continues_past_a_failed_download(self):
        files_resource = _FakeFilesResource({
            None: {"files": [
                {"id": "bad", "name": "era5_mexico_pacifico_2020-01-01.tif"},
                {"id": "good", "name": "era5_mexico_pacifico_2020-01-02.tif"},
            ]},
        })
        service = _FakeService(files_resource)

        def fake_download(svc, file_id, destination, name):
            if file_id == "bad":
                return False
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"data")
            return True

        with patch.object(drive, "_service", return_value=service), \
             patch.object(drive, "_folder_id", return_value="parent-id"), \
             patch.object(drive, "_try_download", side_effect=fake_download), \
             patch.object(drive, "_try_trash", return_value=None):
            with tempfile.TemporaryDirectory() as tmp:
                downloaded = drive.sync_prefixed(Path(tmp), name_prefix="era5", subdir="era5_ee")
                self.assertEqual(downloaded, 1)  # only the good one counted
                self.assertFalse((Path(tmp) / "era5_ee/mexico_pacifico/2020/01/era5_mexico_pacifico_2020-01-01.tif").exists())
                self.assertTrue((Path(tmp) / "era5_ee/mexico_pacifico/2020/01/era5_mexico_pacifico_2020-01-02.tif").exists())


class SyncPrefixedQueryScopingTest(unittest.TestCase):
    """sync_prefixed must scope its Drive query to one region when given,
    to avoid each region's process listing/processing every other region's
    exports too (found in production, 2026-09-21, running 3 regions in
    parallel)."""

    def _run_with_fake_service(self, **sync_kwargs):
        files_resource = _FakeFilesResource({None: {"files": []}})
        service = _FakeService(files_resource)
        with patch.object(drive, "_service", return_value=service), \
             patch.object(drive, "_folder_id", return_value="parent-id"):
            drive.sync_prefixed(Path("/root"), **sync_kwargs)
        return files_resource.list_calls[0]["q"]

    def test_query_includes_region_when_given(self):
        query = self._run_with_fake_service(name_prefix="era5", subdir="era5_ee", region_name="mexico_pacifico")
        self.assertIn("era5_mexico_pacifico_", query)

    def test_query_is_prefix_only_without_region(self):
        query = self._run_with_fake_service(name_prefix="era5", subdir="era5_ee")
        self.assertIn("era5_", query)
        self.assertNotIn("mexico_pacifico", query)


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

    def test_unrelated_file_sharing_prefix_substring_is_rejected(self):
        # Found in production 2026-09-21: an unrelated "era5_pl_*" export in
        # the same Drive folder also matches a naive "name contains 'era5_'"
        # search. "pl" is not a registered region, so this must raise.
        with self.assertRaises(ValueError):
            drive.parse_prefixed_export_name("era5_pl_2025-12-29.tif", "era5")

    def test_malformed_date_suffix_raises(self):
        with self.assertRaises(ValueError):
            drive.parse_prefixed_export_name("era5_mexico_pacifico_not-a-date.tif", "era5")


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
