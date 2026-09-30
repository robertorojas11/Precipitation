"""Tests for the hourly dataset validation gates
(src/data_preprocessing/validate_hourly_dataset.py), stage 3."""

from datetime import date
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
from rasterio.transform import from_origin

from src.data_preprocessing import build_hourly_dataset as B
from src.data_preprocessing import regions
from src.data_preprocessing import validate_hourly_dataset as V
from src.utils import provenance
from tests.test_build_hourly_dataset import _WithSyntheticRegion, _write_geotiff, _era5_descriptions


class ValidateRawTest(unittest.TestCase):
    def _write_manifest(self, root: Path, kind: str, region: str, days: list[str], *, write_files: bool = True, tamper: bool = False):
        records = []
        for day in days:
            rel = f"{kind}/{region}/{day[:4]}/{day[5:7]}/{'era5' if kind == 'era5_ee' else 'imerg'}_{region}_{day}.tif"
            path = root / rel
            if write_files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fake-geotiff-" + day.encode())
            sha = provenance.file_sha256(path) if write_files else "0" * 64
            if tamper:
                sha = "f" * 64
            records.append({"path": rel, "day": day, "sha256": sha, "bytes": 10})
        provenance.write_manifest(
            root / kind / region / f"manifest_{kind}_{region}.json", kind=kind, records=records,
        )

    def test_matching_complete_manifests_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            days = ["2020-01-01", "2020-01-02", "2020-01-03"]
            self._write_manifest(root, "era5_ee", "mexico_pacifico", days)
            self._write_manifest(root, "imerg", "mexico_pacifico", days)
            result = V.validate_raw("mexico_pacifico", v2_raw_root=root)
            self.assertTrue(result["accepted"])
            self.assertEqual(result["counts"], {"era5_ee": 3, "imerg": 3})

    def test_missing_manifest_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_manifest(root, "era5_ee", "mexico_pacifico", ["2020-01-01"])
            result = V.validate_raw("mexico_pacifico", v2_raw_root=root)
            self.assertFalse(result["accepted"])
            self.assertTrue(any(e["error"] == "missing_manifest" for e in result["errors"]))

    def test_missing_file_on_disk_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_manifest(root, "era5_ee", "mexico_pacifico", ["2020-01-01"], write_files=False)
            self._write_manifest(root, "imerg", "mexico_pacifico", ["2020-01-01"], write_files=False)
            result = V.validate_raw("mexico_pacifico", v2_raw_root=root)
            self.assertFalse(result["accepted"])
            self.assertTrue(any(e["error"] == "missing_file" for e in result["errors"]))

    def test_asymmetric_day_coverage_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_manifest(root, "era5_ee", "mexico_pacifico", ["2020-01-01", "2020-01-02"])
            self._write_manifest(root, "imerg", "mexico_pacifico", ["2020-01-01"])
            result = V.validate_raw("mexico_pacifico", v2_raw_root=root)
            self.assertFalse(result["accepted"])
            self.assertTrue(any(e["error"] == "era5_without_matching_imerg" for e in result["errors"]))

    def test_full_hash_check_catches_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_manifest(root, "era5_ee", "mexico_pacifico", ["2020-01-01"], tamper=True)
            self._write_manifest(root, "imerg", "mexico_pacifico", ["2020-01-01"])
            result_fast = V.validate_raw("mexico_pacifico", v2_raw_root=root, full_hash_check=False)
            self.assertTrue(result_fast["accepted"])  # existence-only misses it
            result_full = V.validate_raw("mexico_pacifico", v2_raw_root=root, full_hash_check=True)
            self.assertFalse(result_full["accepted"])
            self.assertTrue(any(e["error"] == "hash_mismatch" for e in result_full["errors"]))


class ValidateProcessedTest(_WithSyntheticRegion):
    def _build_small_region(self, tmp: Path) -> tuple[Path, Path]:
        raw_root = Path(tmp) / "raw"
        output_root = Path(tmp) / "processed"
        era5_transform = from_origin(-80.2, 20.2, 0.25, 0.25)
        imerg_transform = from_origin(-80.2, 20.2, 0.1, 0.1)
        for offset in range(2):
            day = date(2015, 6, 1 + offset)
            _write_geotiff(
                B._source_path(raw_root, "era5", self.REGION_NAME, day),
                np.ones((48, 10, 10), dtype=np.float32), era5_transform,
                descriptions=_era5_descriptions(),
            )
            _write_geotiff(
                B._source_path(raw_root, "imerg", self.REGION_NAME, day),
                np.full((48, 22, 22), 3.0, dtype=np.float32), imerg_transform,
            )
        B.build_region(self.REGION_NAME, v2_raw_root=raw_root, output_root=output_root,
                        start=date(2015, 6, 1), end=date(2015, 6, 2))
        return output_root.parent, output_root  # (root_with_metadata, processed_dir)

    def test_freshly_built_region_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self._build_small_region(Path(tmp))
            result = V.validate_processed(self.REGION_NAME, root=root)
            self.assertTrue(result["accepted"], result["errors"])
            self.assertEqual(result["counts"]["train"], 2)

    def test_missing_npz_file_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, processed = self._build_small_region(Path(tmp))
            npz_files = list(processed.rglob("*.npz"))
            npz_files[0].unlink()
            result = V.validate_processed(self.REGION_NAME, root=root)
            self.assertFalse(result["accepted"])
            self.assertTrue(any(e["error"] == "missing_file" for e in result["errors"]))

    def test_corrupted_npz_file_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, processed = self._build_small_region(Path(tmp))
            npz_files = list(processed.rglob("*.npz"))
            npz_files[0].write_bytes(b"not a real npz")
            result = V.validate_processed(self.REGION_NAME, root=root)
            self.assertFalse(result["accepted"])
            self.assertTrue(any(e["error"] == "unreadable" for e in result["errors"]))

    def test_tampered_manifest_hash_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self._build_small_region(Path(tmp))
            manifest_path = root / "metadata" / f"manifest_hourly_{self.REGION_NAME}.json"
            payload = json.loads(manifest_path.read_text())
            payload["manifest_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(payload))
            result = V.validate_processed(self.REGION_NAME, root=root)
            self.assertFalse(result["accepted"])
            self.assertTrue(any(e["error"] == "manifest_hash_mismatch" for e in result["errors"]))

    def test_sentinel_value_in_valid_pixel_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, processed = self._build_small_region(Path(tmp))
            npz_files = list(processed.rglob("*.npz"))
            with np.load(npz_files[0]) as data:
                payload = dict(data)
            valid_idx = np.argwhere(payload["target_valid_mask"])[0]
            payload["target"][tuple(valid_idx) + (0,)] = np.nan
            np.savez_compressed(npz_files[0], **payload)
            result = V.validate_processed(self.REGION_NAME, root=root)
            self.assertFalse(result["accepted"])
            self.assertTrue(any(e["error"] == "nonfinite_valid_values" for e in result["errors"]))

    def test_missing_index_reports_missing_or_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = V.validate_processed(self.REGION_NAME, root=Path(tmp))
            self.assertFalse(result["accepted"])
            self.assertEqual(result["errors"][0]["error"], "missing_or_empty_dataset_index")


class ValidateDispatchTest(unittest.TestCase):
    def test_unknown_stage_raises(self):
        with self.assertRaises(ValueError):
            V.validate("mexico_pacifico", "not_a_real_stage")


if __name__ == "__main__":
    unittest.main()
