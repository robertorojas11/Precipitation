"""Tests for the hourly pipeline orchestrator
(src/data_preprocessing/pipeline_hourly.py), using a small synthetic region
-- no real acquired data required. Exercises the full validate_raw ->
build -> validate_processed -> stats -> prepare -> validate_prepared chain
end to end, plus resumability and the hard-gate-halts-the-run behavior."""

from datetime import date
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from rasterio.transform import from_origin

from src.data_preprocessing import pipeline_hourly as P
from src.data_preprocessing import regions
from src.utils import provenance
from tests.test_build_hourly_dataset import _era5_descriptions, _write_geotiff


class _WithSyntheticRawRegion(unittest.TestCase):
    """Registers a small synthetic region and writes raw (unbuilt) ERA5/
    IMERG GeoTIFFs for it -- build/stats/prepare are left for the
    orchestrator itself to run during each test."""

    REGION_NAME = "_test_pipeline_region"

    def setUp(self):
        try:
            import geopandas
            from shapely.geometry import box
        except ImportError:
            raise unittest.SkipTest("geopandas is not installed")

        self._tmpdir = tempfile.TemporaryDirectory()
        tmp = Path(self._tmpdir.name)
        shp_path = tmp / "square_basin.shp"
        square = box(-90.0, 10.0, -70.0, 30.0)
        gdf = geopandas.GeoDataFrame({"geometry": [square]}, crs="EPSG:4326")
        gdf.to_file(shp_path)

        regions._basin_geometry.cache_clear()
        self._original_atlantico = regions.Config.ATLANTICO_SHP_PATH
        regions.Config.ATLANTICO_SHP_PATH = str(shp_path)
        regions.REGISTRY[self.REGION_NAME] = regions.Region(
            name=self.REGION_NAME, basin="atlantico",
            lon_min=-80.0, lon_max=-78.0, lat_min=18.0, lat_max=20.0,
            role="eval", description="synthetic test region for the pipeline orchestrator",
        )

        self.raw_root = tmp / "raw"
        self.build_root = tmp / "build"
        self.prepared_root = tmp / "prepared"
        self.results_root = tmp / "results"
        self.log_root = tmp / "logs"

        self.train_days = [date(2015, 6, 1), date(2015, 6, 2)]
        self.val_days = [date(2020, 6, 1)]
        for day, imerg_value in zip(self.train_days + self.val_days, (2.0, 6.0, 4.0)):
            self._write_day(day, imerg_value=imerg_value)
        self._write_acquisition_manifests(self.train_days + self.val_days)

    def _write_day(self, day: date, *, imerg_value: float) -> None:
        from src.data_preprocessing import build_hourly_dataset as B

        era5_transform = from_origin(-80.2, 20.2, 0.25, 0.25)
        era5_data = np.empty((48, 10, 10), dtype=np.float32)
        era5_data[0::2] = 1.0
        era5_data[1::2] = 0.5
        _write_geotiff(
            B._source_path(self.raw_root, "era5", self.REGION_NAME, day), era5_data, era5_transform,
            descriptions=_era5_descriptions(),
        )
        imerg_transform = from_origin(-80.2, 20.2, 0.1, 0.1)
        imerg_data = np.full((48, 22, 22), imerg_value, dtype=np.float32)
        _write_geotiff(B._source_path(self.raw_root, "imerg", self.REGION_NAME, day), imerg_data, imerg_transform)

    def _write_acquisition_manifests(self, days: list[date]) -> None:
        # validate_raw cross-checks *manifests* against disk, not a
        # directory scan -- the real acquisition stage writes one
        # alongside every export (docs/stages_v2/01-acquire.md). Mirrors
        # tests/test_validate_hourly_dataset.py::ValidateRawTest's own
        # hand-built manifest pattern.
        from src.data_preprocessing import build_hourly_dataset as B

        for kind, prefix in (("era5_ee", "era5"), ("imerg", "imerg")):
            records = []
            for day in days:
                path = B._source_path(self.raw_root, "era5" if kind == "era5_ee" else "imerg", self.REGION_NAME, day)
                rel = str(path.relative_to(self.raw_root))
                records.append({
                    "path": rel, "day": day.isoformat(), "sha256": provenance.file_sha256(path), "bytes": path.stat().st_size,
                })
            provenance.write_manifest(
                self.raw_root / kind / self.REGION_NAME / f"manifest_{kind}_{self.REGION_NAME}.json",
                kind=kind, records=records,
            )

    def _options(self, **overrides) -> P.PipelineOptions:
        defaults = dict(
            regions=(self.REGION_NAME,), stages=P.STAGE_ORDER,
            v2_raw_root=self.raw_root, build_root=self.build_root,
            prepared_root=self.prepared_root, results_root=self.results_root,
        )
        defaults.update(overrides)
        return P.PipelineOptions(**defaults)

    def tearDown(self):
        regions.Config.ATLANTICO_SHP_PATH = self._original_atlantico
        regions.REGISTRY.pop(self.REGION_NAME, None)
        regions._basin_geometry.cache_clear()
        self._tmpdir.cleanup()


class PipelineEndToEndTest(_WithSyntheticRawRegion):
    def test_full_pipeline_runs_all_stages_end_to_end(self):
        code = P.run_pipeline(self._options(), self.log_root / "run1")
        self.assertEqual(code, 0)

        prepared_artifact = json.loads(
            (self.results_root / self.REGION_NAME / "validation" / "prepared.json").read_text()
        )
        self.assertTrue(prepared_artifact["accepted"], prepared_artifact)
        self.assertEqual(prepared_artifact["counts"]["train"], 2)
        self.assertEqual(prepared_artifact["counts"]["val"], 1)

        npz_files = list((self.prepared_root / self.REGION_NAME).rglob("*.npz"))
        self.assertEqual(len(npz_files), 3)

        events = (self.log_root / "run1" / "events.jsonl").read_text().strip().splitlines()
        self.assertTrue(events)

    def test_resume_skips_already_completed_stages(self):
        first_code = P.run_pipeline(self._options(), self.log_root / "run1")
        self.assertEqual(first_code, 0)

        second_code = P.run_pipeline(self._options(), self.log_root / "run2")
        self.assertEqual(second_code, 0)
        events = [
            json.loads(line)
            for line in (self.log_root / "run2" / "events.jsonl").read_text().strip().splitlines()
        ]
        skipped_stages = {e["stage"] for e in events if e["status"] == "skipped"}
        self.assertIn("build", skipped_stages)
        self.assertIn("stats", skipped_stages)
        self.assertIn("prepare", skipped_stages)

    def test_without_resume_reruns_every_stage(self):
        P.run_pipeline(self._options(), self.log_root / "run1")
        code = P.run_pipeline(self._options(resume=False), self.log_root / "run2")
        self.assertEqual(code, 0)
        events = [
            json.loads(line)
            for line in (self.log_root / "run2" / "events.jsonl").read_text().strip().splitlines()
        ]
        self.assertFalse(any(e["status"] == "skipped" for e in events))


class PipelineHaltsOnFailureTest(_WithSyntheticRawRegion):
    def test_validate_processed_without_a_build_fails_and_halts(self):
        # No "build" stage selected -> validate_processed must fail
        # (missing/empty dataset index), and the run must stop there
        # rather than continuing to stats/prepare on nothing.
        code = P.run_pipeline(
            self._options(stages=("validate_processed", "stats", "prepare")), self.log_root / "run1",
        )
        self.assertEqual(code, 1)
        events = [
            json.loads(line)
            for line in (self.log_root / "run1" / "events.jsonl").read_text().strip().splitlines()
        ]
        self.assertTrue(any(e["stage"] == "validate_processed" and e["status"] == "failed" for e in events))
        self.assertFalse(any(e["stage"] == "stats" for e in events))

    def test_continue_on_error_proceeds_past_a_failed_gate(self):
        code = P.run_pipeline(
            self._options(stages=("validate_processed", "build"), continue_on_error=True), self.log_root / "run1",
        )
        self.assertEqual(code, 1)  # overall run still reports failure
        events = [
            json.loads(line)
            for line in (self.log_root / "run1" / "events.jsonl").read_text().strip().splitlines()
        ]
        self.assertTrue(any(e["stage"] == "build" for e in events))  # ran anyway


if __name__ == "__main__":
    unittest.main()
