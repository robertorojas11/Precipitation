"""Tests for stage 6 (training) -- src/training/{dataset,common,
eta_integration,train_baseline,train_gan,train_diffusion}.py -- using a
small synthetic prepared region (no real acquired/build/prepared data
required). All training runs use tiny model widths and 1-2 epochs on CPU,
exercising the full forward/backward/checkpoint path rather than real
convergence."""

from datetime import date
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from rasterio.transform import from_origin

from src.data_preprocessing import build_hourly_dataset as B
from src.data_preprocessing import prepare_hourly_dataset as P
from src.data_preprocessing import regions
from tests.test_build_hourly_dataset import _era5_descriptions, _write_geotiff


class _WithPreparedSyntheticRegion(unittest.TestCase):
    """Registers a small synthetic region and runs it through build ->
    compute_stats -> prepare, so stage 6 can train against a real (if tiny)
    prepared dataset end to end."""

    REGION_NAME = "_test_train_region"

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
        # 6.4x6.4 deg -> a 64x64 canonical grid at 0.1 deg: the discriminator
        # downsamples 4x (src/models/discriminator.py's hr_widths), which
        # collapses a too-small grid (e.g. 20x20) to 1x1 and then crashes on
        # reflect-padding a size-1 dimension -- 64 halves cleanly to 4 and
        # comfortably exceeds every real region's minimum grid dimension
        # (puerto_rico_caribe's 130x260 is the smallest), so this reproduces
        # a shape regime the discriminator actually has to survive.
        regions.REGISTRY[self.REGION_NAME] = regions.Region(
            name=self.REGION_NAME, basin="atlantico",
            lon_min=-84.0, lon_max=-77.6, lat_min=14.0, lat_max=20.4,
            role="eval", description="synthetic test region for stage 6 training",
        )

        raw_root = tmp / "raw"
        build_root = tmp / "build"
        self.prepared_root = tmp / "prepared"

        train_days = [date(2015, 6, 1), date(2015, 6, 2)]
        val_days = [date(2020, 6, 1)]
        for day, imerg_base in zip(train_days + val_days, (2.0, 6.0, 4.0)):
            self._write_day(raw_root, day, imerg_base=imerg_base)
        # One build_region call spanning the full range: it overwrites
        # (not appends) the CSV index on every call, so building the three
        # days across two split years in separate calls would each wipe out
        # the previous day's row. Scanning the ~1800 intervening missing
        # days this way is cheap (existence checks only, no decoding).
        B.build_region(
            self.REGION_NAME, v2_raw_root=raw_root, output_root=build_root / "processed",
            start=min(train_days + val_days), end=max(train_days + val_days),
        )
        stats = P.compute_stats((self.REGION_NAME,), build_root=build_root, output_root=self.prepared_root)
        P.prepare_region(self.REGION_NAME, stats, build_root=build_root, output_root=self.prepared_root)

    def _write_day(self, raw_root: Path, day: date, *, imerg_base: float) -> None:
        # Source rasters cover the region bbox (-84.0..-77.6, 14.0..20.4)
        # plus a margin, at a coarser (ERA5, 0.25 deg) and the canonical
        # (IMERG, 0.1 deg) resolution -- not aligned to the canonical grid,
        # same "build_day must reproject" pattern as test_build_hourly_dataset.
        era5_transform = from_origin(-84.2, 20.6, 0.25, 0.25)
        era5_data = np.empty((48, 30, 30), dtype=np.float32)
        era5_data[0::2] = 1.0
        era5_data[1::2] = 0.5
        _write_geotiff(
            B._source_path(raw_root, "era5", self.REGION_NAME, day), era5_data, era5_transform,
            descriptions=_era5_descriptions(),
        )
        imerg_transform = from_origin(-84.2, 20.6, 0.1, 0.1)
        # A spatial ramp, not a flat constant: a constant field has zero
        # variance, which makes R2's denominator (obs_sq_sum - obs_sum^2/n)
        # exactly zero and every validation epoch report R2=nan -- correct
        # behavior for genuinely information-free data, but it starves this
        # fixture of any real training signal to exercise.
        row, col = np.meshgrid(np.arange(70), np.arange(70), indexing="ij")
        spatial_ramp = (row + col) / 139.0 * 8.0  # 0..8 mm/hr across the grid
        imerg_data = np.broadcast_to(
            (imerg_base + spatial_ramp).astype(np.float32), (48, 70, 70),
        ).copy()
        _write_geotiff(B._source_path(raw_root, "imerg", self.REGION_NAME, day), imerg_data, imerg_transform)

    def tearDown(self):
        regions.Config.ATLANTICO_SHP_PATH = self._original_atlantico
        regions.REGISTRY.pop(self.REGION_NAME, None)
        regions._basin_geometry.cache_clear()
        self._tmpdir.cleanup()


class PreparedHourlyDatasetTest(_WithPreparedSyntheticRegion):
    def test_length_is_days_times_24(self):
        from src.training.dataset import PreparedHourlyDataset

        train = PreparedHourlyDataset(self.REGION_NAME, "train", prepared_root=self.prepared_root)
        val = PreparedHourlyDataset(self.REGION_NAME, "val", prepared_root=self.prepared_root)
        self.assertEqual(len(train), 2 * 24)
        self.assertEqual(len(val), 1 * 24)

    def test_item_shapes_dtypes_and_nan_sanitization(self):
        from src.training.dataset import PreparedHourlyDataset

        dataset = PreparedHourlyDataset(self.REGION_NAME, "train", prepared_root=self.prepared_root)
        item = dataset[0]
        self.assertEqual(item["inputs"].shape, (2, 64, 64))
        self.assertEqual(item["invariants"].shape, (2, 64, 64))
        self.assertEqual(item["target"].shape, (1, 64, 64))
        self.assertEqual(item["target_valid_mask"].shape, (64, 64))
        self.assertEqual(item["target_valid_mask"].dtype, torch.bool)
        for key in ("inputs", "invariants", "target"):
            self.assertTrue(torch.isfinite(item[key]).all(), key)

    def test_unknown_split_raises(self):
        from src.training.dataset import PreparedHourlyDataset

        with self.assertRaises(ValueError):
            PreparedHourlyDataset(self.REGION_NAME, "bogus", prepared_root=self.prepared_root)


class TrainBaselineTest(_WithPreparedSyntheticRegion):
    def test_one_epoch_runs_and_checkpoints(self):
        from src.training.train_baseline import BaselineTrainingConfig, train

        with tempfile.TemporaryDirectory() as out:
            config = BaselineTrainingConfig(
                region=self.REGION_NAME, epochs=1, batch_size=8, base_width=8, depth=2,
                device="cpu", patience=1,
            )
            run_dir = train(config, prepared_root=self.prepared_root, output_root=Path(out))
            self.assertTrue((run_dir / "last.pt").exists())
            self.assertTrue((run_dir / "best.pt").exists())
            self.assertTrue((run_dir / "history.json").exists())
            checkpoint = torch.load(run_dir / "last.pt", weights_only=False)
            self.assertIn("model_state", checkpoint)

    def test_eta_learning_runs_when_enabled(self):
        from src.training.train_baseline import BaselineTrainingConfig, train

        with tempfile.TemporaryDirectory() as out:
            config = BaselineTrainingConfig(
                region=self.REGION_NAME, epochs=1, batch_size=8, base_width=8, depth=2,
                device="cpu", patience=1, eta_lambda=0.1, eta_pretrain_steps=0,
                eta_refresh_every=1, eta_pool_size=8, eta_n_q=4,
            )
            run_dir = train(config, prepared_root=self.prepared_root, output_root=Path(out))
            history = (run_dir / "history.json").read_text()
            self.assertIn("train_loss", history)


class TrainGanTest(_WithPreparedSyntheticRegion):
    def test_one_epoch_runs_and_checkpoints(self):
        from src.training.train_gan import GanTrainingConfig, train

        with tempfile.TemporaryDirectory() as out:
            config = GanTrainingConfig(
                region=self.REGION_NAME, epochs=1, batch_size=4, base_width=8, depth=2, device="cpu",
            )
            run_dir = train(config, prepared_root=self.prepared_root, output_root=Path(out))
            self.assertTrue((run_dir / "last.pt").exists())
            checkpoint = torch.load(run_dir / "last.pt", weights_only=False)
            self.assertIn("generator_state", checkpoint)
            self.assertIn("discriminator_state", checkpoint)


class TrainDiffusionTest(_WithPreparedSyntheticRegion):
    def test_stage1_then_stage2_run_and_checkpoint(self):
        from src.training.train_diffusion import (
            DiffusionStageTrainingConfig,
            RegressionStageTrainingConfig,
            train_diffusion_stage,
            train_regression_stage,
        )

        with tempfile.TemporaryDirectory() as out:
            stage1_config = RegressionStageTrainingConfig(
                region=self.REGION_NAME, epochs=1, batch_size=8, base_width=8, device="cpu", patience=1,
            )
            stage1_run_dir = train_regression_stage(stage1_config, prepared_root=self.prepared_root, output_root=Path(out))
            stage1_checkpoint = stage1_run_dir / "best.pt"
            self.assertTrue(stage1_checkpoint.exists())

            stage2_config = DiffusionStageTrainingConfig(
                region=self.REGION_NAME, stage1_checkpoint=str(stage1_checkpoint),
                epochs=1, batch_size=8, base_width=8, device="cpu", patience=1,
                sampler_steps_for_validation=3,
            )
            stage2_run_dir = train_diffusion_stage(stage2_config, prepared_root=self.prepared_root, output_root=Path(out))
            self.assertTrue((stage2_run_dir / "last.pt").exists())
            checkpoint = torch.load(stage2_run_dir / "last.pt", weights_only=False)
            self.assertIn("denoiser_state", checkpoint)
