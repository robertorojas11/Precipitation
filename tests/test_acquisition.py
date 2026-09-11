"""Acquisition modules: credential gating, resume, manifests -- no network I/O."""

import unittest
from datetime import date
from pathlib import Path
import tempfile

import numpy as np

from src.data_extraction import era5_cds, export_oya, imerg
from src.data_preprocessing import regions
from src.utils import provenance
from src.utils.config import Config


class _CredentialGuard(unittest.TestCase):
    """Clear all acquisition credentials for the duration of a test."""

    _KEYS = (
        "CDSAPI_URL", "CDSAPI_KEY",
        "EARTHDATA_TOKEN", "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD",
    )

    def setUp(self):
        self._saved = {k: getattr(Config, k) for k in self._KEYS}
        for k in self._KEYS:
            setattr(Config, k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(Config, k, v)


class Era5CdsTest(_CredentialGuard):
    def test_missing_credentials_raise_with_variable_name(self):
        with self.assertRaises(era5_cds.MissingCredentialsError) as ctx:
            era5_cds.resolve_cds_credentials()
        self.assertIn("CDSAPI_KEY", str(ctx.exception))

    def test_month_iteration_is_inclusive(self):
        months = list(era5_cds._iter_months(era5_cds.Month(2001, 11), era5_cds.Month(2002, 2)))
        self.assertEqual([str(m) for m in months], ["2001-11", "2001-12", "2002-01", "2002-02"])

    def test_cds_area_is_north_west_south_east(self):
        region = regions.get_region("mexico_pacifico")
        area = era5_cds.cds_area(region)
        self.assertEqual(area, [region.lat_max, region.lon_min, region.lat_min, region.lon_max])

    def test_download_writes_files_and_manifest_and_resumes(self):
        Config.CDSAPI_URL, Config.CDSAPI_KEY = "https://example/api", "key"
        calls = []

        class FakeClient:
            def retrieve(self, dataset, request, path):
                calls.append((dataset, request["variable"][0]))
                Path(path).write_bytes(b"NETCDF-FAKE")

        region = regions.get_region("mexico_pacifico")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = era5_cds.download_range(
                era5_cds.Month(2001, 1), era5_cds.Month(2001, 1),
                region, root=root, client=FakeClient(),
            )
            self.assertEqual(len(calls), 2)  # cp + lsp for one month
            manifest = provenance.load_manifest(manifest_path)
            self.assertEqual(manifest["kind"], "era5_cds")
            self.assertEqual(manifest["record_count"], 2)
            for rec in manifest["records"]:
                self.assertEqual(len(rec["sha256"]), 64)
                self.assertTrue((root / rec["path"]).exists())

            # second run: files exist -> no new retrieve calls
            calls.clear()
            era5_cds.download_range(
                era5_cds.Month(2001, 1), era5_cds.Month(2001, 1),
                region, root=root, client=FakeClient(),
            )
            self.assertEqual(calls, [])

    def test_dry_run_still_checks_credentials(self):
        with self.assertRaises(era5_cds.MissingCredentialsError):
            era5_cds.download_range(
                era5_cds.Month(2001, 1), era5_cds.Month(2001, 1),
                regions.get_region("mexico_pacifico"), root=Path("/tmp/never"), dry_run=True,
            )


class ImergTest(_CredentialGuard):
    def test_missing_credentials_raise_with_variable_name(self):
        with self.assertRaises(imerg.MissingCredentialsError) as ctx:
            imerg.resolve_earthdata_auth()
        self.assertIn("EARTHDATA_TOKEN", str(ctx.exception))

    def test_granule_urls_are_48_half_hourly_slots(self):
        urls = imerg.granule_urls(date(2020, 6, 1))
        self.assertEqual(len(urls), imerg.SLOTS_PER_DAY)
        self.assertIn("2020/153/", urls[0])  # day-of-year 153
        self.assertIn("-S000000-E002959.0000.", urls[0])
        self.assertIn("-S233000-E235959.1410.", urls[-1])

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

    def test_download_range_mocked_end_to_end(self):
        Config.EARTHDATA_TOKEN = "tok"
        region = regions.get_region("mexico_pacifico")
        lats = np.linspace(region.lat_min, region.lat_max, 8)
        lons = np.linspace(region.lon_min, region.lon_max, 10)

        def fake_download(session, url, **kw):
            return b"x"

        def fake_read(content, region_):
            return np.full((lats.size, lons.size), 0.5, np.float32), lats, lons

        orig_dl, orig_rd = imerg._download_with_retry, imerg._read_granule
        imerg._download_with_retry = fake_download
        imerg._read_granule = fake_read
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest_path = imerg.download_range(
                    date(2021, 1, 1), date(2021, 1, 2),
                    region, root=root, session=object(),
                )
                manifest = provenance.load_manifest(manifest_path)
                self.assertEqual(manifest["record_count"], 2)  # two days
                for rec in manifest["records"]:
                    path = root / rec["path"]
                    self.assertTrue(path.exists())
                    self.assertEqual(len(rec["sha256"]), 64)
        finally:
            imerg._download_with_retry, imerg._read_granule = orig_dl, orig_rd


class OyaRegionScopingTest(unittest.TestCase):
    def test_unscoped_path_matches_legacy_layout(self):
        from datetime import datetime

        path = export_oya._local_target(Path("/root"), datetime(2020, 3, 5), None)
        self.assertEqual(path, Path("/root/oya/2020/03/oya_2020-03-05.tif"))

    def test_region_scoped_path_is_namespaced(self):
        from datetime import datetime

        path = export_oya._local_target(Path("/root"), datetime(2020, 3, 5), "puerto_rico_caribe")
        self.assertEqual(
            path,
            Path("/root/oya/puerto_rico_caribe/2020/03/oya_puerto_rico_caribe_2020-03-05.tif"),
        )


class ProvenanceTest(unittest.TestCase):
    def test_write_and_load_manifest_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            written = provenance.write_manifest(
                path, kind="test", records=[{"path": "a", "sha256": "x" * 64}]
            )
            loaded = provenance.load_manifest(path)
            self.assertEqual(loaded["manifest_sha256"], written["manifest_sha256"])
            self.assertEqual(loaded["record_count"], 1)

    def test_load_missing_manifest_returns_none(self):
        self.assertIsNone(provenance.load_manifest("/tmp/does-not-exist-manifest.json"))


if __name__ == "__main__":
    unittest.main()
