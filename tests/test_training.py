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
import unittest.mock

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

    def test_per_day_cache_avoids_reopening_the_same_file(self):
        from src.training.dataset import PreparedHourlyDataset

        dataset = PreparedHourlyDataset(self.REGION_NAME, "train", prepared_root=self.prepared_root)
        open_calls = []
        real_load = np.load

        def counting_load(path, *args, **kwargs):
            open_calls.append(path)
            return real_load(path, *args, **kwargs)

        with unittest.mock.patch("src.training.dataset.np.load", side_effect=counting_load):
            for hour in range(24):  # day 0, every hour -- sequential, same day throughout
                dataset[hour]
        self.assertEqual(len(open_calls), 1, "24 hours of the same day should open the NPZ exactly once")

        with unittest.mock.patch("src.training.dataset.np.load", side_effect=counting_load):
            dataset[24]  # day 1 -- a genuine cache miss
        self.assertEqual(len(open_calls), 2)

    def test_day_grouped_sampler_covers_every_index_grouped_by_day(self):
        from src.training.dataset import HOURS_PER_DAY, DayGroupedSampler, PreparedHourlyDataset

        dataset = PreparedHourlyDataset(self.REGION_NAME, "train", prepared_root=self.prepared_root)
        sampler = DayGroupedSampler(dataset, shuffle=True, seed=0)
        order = list(sampler)
        self.assertEqual(sorted(order), list(range(len(dataset))))  # full coverage, no duplicates

        # Every run of HOURS_PER_DAY consecutive yielded indices must share
        # the same day -- the exact property the per-day cache depends on.
        for start in range(0, len(order), HOURS_PER_DAY):
            days_in_block = {index // HOURS_PER_DAY for index in order[start:start + HOURS_PER_DAY]}
            self.assertEqual(days_in_block, {order[start] // HOURS_PER_DAY})

        sampler.set_epoch(1)
        self.assertNotEqual(list(sampler), order, "a different epoch must reshuffle the day order")

    def test_preload_loads_every_day_up_front_and_getitem_matches_non_preloaded(self):
        from src.training.dataset import PreparedHourlyDataset

        lazy = PreparedHourlyDataset(self.REGION_NAME, "train", prepared_root=self.prepared_root)
        preloaded = PreparedHourlyDataset(
            self.REGION_NAME, "train", prepared_root=self.prepared_root, preload=True,
        )
        for index in range(len(lazy)):
            for key in ("inputs", "invariants", "target", "input_valid_mask", "target_valid_mask"):
                torch.testing.assert_close(lazy[index][key], preloaded[index][key])

    def test_preload_true_does_not_require_sequential_access(self):
        from src.training.dataset import PreparedHourlyDataset

        open_calls = []
        real_load = np.load

        def counting_load(path, *args, **kwargs):
            open_calls.append(path)
            return real_load(path, *args, **kwargs)

        with unittest.mock.patch("src.training.dataset.np.load", side_effect=counting_load):
            dataset = PreparedHourlyDataset(
                self.REGION_NAME, "train", prepared_root=self.prepared_root, preload=True,
            )
        self.assertEqual(len(open_calls), 2, "preload should open each of the 2 synthetic train days exactly once")

        # Scattered, non-day-grouped access must not trigger any further loads.
        with unittest.mock.patch("src.training.dataset.np.load", side_effect=counting_load):
            for index in (23, 0, 47, 24, 1):
                dataset[index]
        self.assertEqual(len(open_calls), 2, "preloaded access must never reopen the NPZ")


class TrainCommonTest(unittest.TestCase):
    def test_total_precip_input_mm_sums_cp_and_lsp_channels_in_order(self):
        from src.data_preprocessing.prepare_hourly_dataset import ChannelMoments
        from src.training.common import denormalize_to_mm, total_precip_input_mm

        cp_moments = ChannelMoments(mean=0.1, std=0.5, count=100)
        lsp_moments = ChannelMoments(mean=-0.2, std=0.8, count=100)
        inputs_norm = torch.tensor([[[[1.0]], [[-0.5]]]])  # (1, 2, 1, 1): channel 0 = cp, channel 1 = lsp

        result = total_precip_input_mm(inputs_norm, cp_moments, lsp_moments)
        expected = denormalize_to_mm(inputs_norm[:, 0:1], cp_moments) + denormalize_to_mm(inputs_norm[:, 1:2], lsp_moments)
        torch.testing.assert_close(result, expected)

        # Swapping which moments pair with which channel must change the
        # result -- locks in that cp is channel 0 and lsp is channel 1.
        swapped = total_precip_input_mm(inputs_norm, lsp_moments, cp_moments)
        self.assertFalse(torch.allclose(result, swapped))

    def test_input_moments_reads_cp_and_lsp_from_pooled_stats(self):
        from src.training.common import input_moments

        stats = {
            "pooled": {
                "cp": {"mean": 1.0, "std": 2.0, "count": 10},
                "lsp": {"mean": 3.0, "std": 4.0, "count": 10},
                "target": {"mean": 0.0, "std": 1.0, "count": 10},
            },
        }
        cp_moments, lsp_moments = input_moments(stats)
        self.assertEqual((cp_moments.mean, cp_moments.std), (1.0, 2.0))
        self.assertEqual((lsp_moments.mean, lsp_moments.std), (3.0, 4.0))

    def test_pooled_metrics_clips_one_pathological_residual_instead_of_exploding_r2(self):
        """Regression lock: a real validation pass once produced r2=-4e18
        because one pixel's prediction blew up numerically. A handful of
        sane residuals plus one absurd one must stay in a readable range."""
        from src.training.common import PooledMetrics

        obs = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        pred_sane = obs + 0.1
        pred_sane[0] = 1e12  # one pathological prediction
        mask = torch.ones_like(obs, dtype=torch.bool)

        clipped = PooledMetrics(max_abs_residual=500.0)
        clipped.update(pred_sane.unsqueeze(0), obs.unsqueeze(0), mask.unsqueeze(0))
        clipped_r2 = clipped.compute()["r2"]

        unclipped = PooledMetrics(max_abs_residual=None)
        unclipped.update(pred_sane.unsqueeze(0), obs.unsqueeze(0), mask.unsqueeze(0))
        unclipped_r2 = unclipped.compute()["r2"]

        self.assertGreater(clipped.compute()["clipped_count"], 0)
        # unclipped blows up to ~-1e22 scale from the 1e12 prediction; clipped
        # is bounded by max_abs_residual**2, many orders of magnitude tamer.
        self.assertLess(unclipped_r2, -1e15)
        self.assertGreater(clipped_r2, unclipped_r2 / 1e6)

    def test_pooled_metrics_leaves_ordinary_residuals_untouched(self):
        from src.training.common import PooledMetrics

        unclipped = PooledMetrics(max_abs_residual=None)
        clipped = PooledMetrics(max_abs_residual=500.0)
        obs = torch.tensor([1.0, 2.0, 3.0, 4.0])
        pred = torch.tensor([1.1, 1.8, 3.3, 3.9])
        mask = torch.ones_like(obs, dtype=torch.bool)
        unclipped.update(pred.unsqueeze(0), obs.unsqueeze(0), mask.unsqueeze(0))
        clipped.update(pred.unsqueeze(0), obs.unsqueeze(0), mask.unsqueeze(0))

        self.assertEqual(clipped.compute()["clipped_count"], 0)
        self.assertAlmostEqual(unclipped.compute()["r2"], clipped.compute()["r2"], places=10)

    def test_masked_mean_matches_manual_average_when_mask_lacks_channel_dim(self):
        """Regression lock: found 2026-10-19 live in train_baseline.py's and
        diffusion stage-1's training loss. ``values`` (B,1,H,W) against
        ``mask`` (B,H,W) must reduce to the per-sample masked average, not
        silently broadcast into a (B,B,H,W) cross-batch outer product."""
        from src.training.common import masked_mean

        torch.manual_seed(0)
        batch, h, w = 3, 4, 4
        values = torch.rand(batch, 1, h, w)
        mask = torch.rand(batch, h, w) > 0.5
        self.assertTrue(mask.any(), "need at least one valid pixel for a meaningful average")

        manual = values.squeeze(1)[mask].mean()
        result = masked_mean(values, mask)
        torch.testing.assert_close(result, manual)

        # The old buggy behaviour scaled the result by ~batch_size -- a
        # tripwire in case the fix regresses back to that shape.
        self.assertLess((result - manual).abs().item(), 1e-5)

    def test_masked_mean_still_works_when_mask_already_has_channel_dim(self):
        from src.training.common import masked_mean

        torch.manual_seed(1)
        batch, h, w = 2, 4, 4
        values = torch.rand(batch, 1, h, w)
        mask = (torch.rand(batch, h, w) > 0.3).unsqueeze(1)

        manual = values[mask].mean()
        result = masked_mean(values, mask)
        torch.testing.assert_close(result, manual)


class DeterministicResNetResidualTest(unittest.TestCase):
    def test_total_precip_input_activates_residual_branch(self):
        from src.models.deterministic_resnet import DeterministicResNet, ResNetConfig

        torch.manual_seed(0)
        model = DeterministicResNet(ResNetConfig(base_width=4, depth=1))
        inputs = torch.randn(1, 2, 8, 8)
        invariants = torch.randn(1, 2, 8, 8)
        total_precip_input = torch.rand(1, 1, 8, 8) * 5.0

        without = model(inputs, invariants)
        with_residual = model(inputs, invariants, total_precip_input)
        self.assertFalse(torch.allclose(without, with_residual), "total_precip_input must reach the output")

    def test_residual_preactivation_bound_keeps_softplus_gradient_alive(self):
        """Regression lock for the dead-gradient bug measured on a real
        checkpoint (run4, epoch 11): plain L1 against mostly-zero targets
        drove the unbounded pre-activation to mean -30, where softplus'
        gradient (sigmoid(preact)) underflows to an exact 0.0 and freezes the
        whole trunk. -30 is reproduced here via an extreme head bias (the
        organic route, through thousands of Adam steps, isn't practical in a
        unit test); the bound must keep the gradient representable at it."""
        from src.models.deterministic_resnet import DeterministicResNet, ResNetConfig

        torch.manual_seed(0)
        model = DeterministicResNet(ResNetConfig(base_width=4, depth=1))
        with torch.no_grad():
            model.head.bias.fill_(-30.0)
        inputs = torch.randn(1, 2, 8, 8)
        invariants = torch.randn(1, 2, 8, 8)
        total_precip_input = torch.zeros(1, 1, 8, 8)  # base = log1p(0) = 0 -- worst case

        prediction = model(inputs, invariants, total_precip_input)
        self.assertGreater(prediction.max().item(), 0.0, "bound must prevent an exact-zero collapse")

        loss = prediction.sum()
        loss.backward()
        head_grad_norm = model.stem.weight.grad.norm().item()
        self.assertGreater(head_grad_norm, 0.0, "gradient to the trunk must not underflow to exact zero")


class TrainBaselineTest(_WithPreparedSyntheticRegion):
    def test_one_epoch_runs_and_checkpoints(self):
        from src.models.deterministic_resnet import DeterministicResNet, ResNetConfig
        from src.training.common import input_moments, load_norm_stats, to_device, total_precip_input_mm
        from src.training.dataset import PreparedHourlyDataset
        from src.training.train_baseline import BaselineTrainingConfig, _forward, train

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

            # Regression lock for the residual-over-input wiring bug: a
            # trained checkpoint's prediction must not be a flat constant
            # (the exact symptom that delated the Softplus-saturation collapse).
            model = DeterministicResNet(ResNetConfig(**checkpoint["model_config"]))
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            stats = load_norm_stats(self.prepared_root)
            cp_moments, lsp_moments = input_moments(stats)
            val_dataset = PreparedHourlyDataset(self.REGION_NAME, "val", prepared_root=self.prepared_root)
            batch = to_device(
                {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in val_dataset[0].items()},
                torch.device("cpu"),
            )
            total_precip_input = total_precip_input_mm(batch["inputs"], cp_moments, lsp_moments)
            with torch.no_grad():
                prediction = _forward(model, batch["inputs"], batch["invariants"], total_precip_input)
            self.assertGreater(prediction.std().item(), 0.0, "prediction collapsed to a constant")

    def test_rain_weight_raises_loss_contribution_of_wet_pixels_only(self):
        from src.training.train_baseline import _erm_loss

        # 1 wet pixel (target=1.0 > threshold), 3 dry pixels (target=0.0).
        # Same per-pixel error (0.5) everywhere, so unweighted loss is 0.5
        # regardless of which pixels are wet; weighting must change that.
        prediction = torch.tensor([[[[0.5, 0.5, 0.5, 0.5]]]])
        target = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
        mask = torch.ones(1, 4, dtype=torch.bool)

        unweighted = _erm_loss(prediction, target, mask, rain_weight=1.0)
        self.assertAlmostEqual(unweighted.item(), 0.5, places=6)

        weighted = _erm_loss(prediction, target, mask, rain_weight=4.0, wet_threshold_mm=0.1)
        # weights [4,1,1,1] on an identical 0.5 error everywhere still average
        # to 0.5 -- a uniform error is invariant to reweighting.
        self.assertAlmostEqual(weighted.item(), 0.5, places=6)

        dry_wrong = torch.tensor([[[[1.0, 0.5, 0.5, 0.5]]]])  # wet pixel now exact
        dry_only_unweighted = _erm_loss(dry_wrong, target, mask, rain_weight=1.0)
        dry_only_weighted = _erm_loss(dry_wrong, target, mask, rain_weight=4.0, wet_threshold_mm=0.1)
        # Error now comes only from dry pixels; the (zero-error) wet pixel's
        # extra weight mass dilutes their relative contribution.
        self.assertLess(dry_only_weighted.item(), dry_only_unweighted.item())

        wet_wrong = torch.tensor([[[[0.0, 0.0, 0.0, 0.0]]]])  # only wet pixel is wrong now
        wet_only_unweighted = _erm_loss(wet_wrong, target, mask, rain_weight=1.0)
        wet_only_weighted = _erm_loss(wet_wrong, target, mask, rain_weight=4.0, wet_threshold_mm=0.1)
        # Wet pixel is the only error source -> up-weighting it must raise
        # the loss relative to the unweighted case.
        self.assertGreater(wet_only_weighted.item(), wet_only_unweighted.item())

    def test_one_epoch_runs_with_num_workers_greater_than_zero(self):
        from src.training.train_baseline import BaselineTrainingConfig, train

        with tempfile.TemporaryDirectory() as out:
            config = BaselineTrainingConfig(
                region=self.REGION_NAME, epochs=1, batch_size=8, base_width=8, depth=2,
                device="cpu", patience=1, num_workers=2, preload_dataset=True,
            )
            run_dir = train(config, prepared_root=self.prepared_root, output_root=Path(out))
            self.assertTrue((run_dir / "last.pt").exists())

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

    def test_stage2_runs_with_nonzero_s_churn(self):
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
            stage1_checkpoint = train_regression_stage(
                stage1_config, prepared_root=self.prepared_root, output_root=Path(out),
            ) / "best.pt"

            stage2_config = DiffusionStageTrainingConfig(
                region=self.REGION_NAME, stage1_checkpoint=str(stage1_checkpoint),
                epochs=1, batch_size=8, base_width=8, device="cpu", patience=1,
                sampler_steps_for_validation=3, s_churn=7.0,
            )
            stage2_run_dir = train_diffusion_stage(stage2_config, prepared_root=self.prepared_root, output_root=Path(out))
            self.assertTrue((stage2_run_dir / "last.pt").exists())

    def test_stage2_custom_sigma_data_reaches_the_denoiser(self):
        """D2 (measured on real checkpoints): sigma_data=0.5 is Karras'
        generic default, not this dataset's measured residual scale. Locks
        in that a custom value set at the training-config level actually
        reaches the EDMPrecond wrapper the denoiser trains with, not just a
        local variable that gets silently dropped."""
        import json

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
            stage1_checkpoint = train_regression_stage(
                stage1_config, prepared_root=self.prepared_root, output_root=Path(out),
            ) / "best.pt"

            stage2_config = DiffusionStageTrainingConfig(
                region=self.REGION_NAME, stage1_checkpoint=str(stage1_checkpoint),
                epochs=1, batch_size=8, base_width=8, device="cpu", patience=1,
                sampler_steps_for_validation=3, sigma_data=0.8, p_mean=-0.22, p_std=0.9,
            )
            stage2_run_dir = train_diffusion_stage(stage2_config, prepared_root=self.prepared_root, output_root=Path(out))
            run_metadata = json.loads((stage2_run_dir / "run.json").read_text())
            self.assertEqual(run_metadata["config"]["sigma_data"], 0.8)
            self.assertEqual(run_metadata["config"]["p_mean"], -0.22)
            self.assertEqual(run_metadata["config"]["p_std"], 0.9)

    def test_stage2_same_seed_reproduces_identical_train_loss(self):
        import json

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
            stage1_checkpoint = train_regression_stage(
                stage1_config, prepared_root=self.prepared_root, output_root=Path(out),
            ) / "best.pt"

            def run_once(run_name: str) -> list[float]:
                config = DiffusionStageTrainingConfig(
                    region=self.REGION_NAME, stage1_checkpoint=str(stage1_checkpoint), run_name=run_name,
                    epochs=1, batch_size=8, base_width=8, device="cpu", patience=1,
                    sampler_steps_for_validation=3, seed=123,
                )
                run_dir = train_diffusion_stage(config, prepared_root=self.prepared_root, output_root=Path(out))
                history = json.loads((run_dir / "history.json").read_text())
                return [row["train_loss"] for row in history]

            self.assertEqual(run_once("rng_a"), run_once("rng_b"))

    def test_stage2_validation_noise_is_identical_across_repeated_calls(self):
        """Regression lock, found 2026-10-19 while investigating why val_r2
        kept swinging wildly epoch to epoch even after the sigma_data/
        s_churn fixes: ``validate_diffusion`` was called with the same
        ``rng`` object training kept advancing, so every epoch's Heun
        sampling drew different noise -- part of the swing was sampling
        luck, not the denoiser changing. Calling it twice with a generator
        re-seeded to the same value (what ``train_diffusion_stage`` now
        does every epoch) on an unchanged denoiser must give bit-identical
        metrics."""
        from src.models.diffusion_residual import EDMConfig, HeunSamplerConfig
        from src.training.common import load_norm_stats, target_moments
        from src.training.dataset import PreparedHourlyDataset
        from src.training.train_diffusion import (
            RegressionStageTrainingConfig,
            _build_denoiser,
            _load_stage1,
            train_regression_stage,
            validate_diffusion,
        )
        from torch.utils.data import DataLoader

        with tempfile.TemporaryDirectory() as out:
            stage1_config = RegressionStageTrainingConfig(
                region=self.REGION_NAME, epochs=1, batch_size=8, base_width=8, device="cpu", patience=1,
            )
            stage1_checkpoint = train_regression_stage(
                stage1_config, prepared_root=self.prepared_root, output_root=Path(out),
            ) / "best.pt"
            stage1 = _load_stage1(str(stage1_checkpoint), torch.device("cpu"))

            denoiser, _ = _build_denoiser(8, torch.device("cpu"), EDMConfig())
            sampler_config = HeunSamplerConfig(num_steps=3)
            moments = target_moments(load_norm_stats(self.prepared_root))
            val_dataset = PreparedHourlyDataset(self.REGION_NAME, "val", prepared_root=self.prepared_root)
            val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

            def run_validation() -> dict[str, float]:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(42)
                return validate_diffusion(
                    denoiser, stage1, val_loader, torch.device("cpu"), moments, sampler_config, generator=generator,
                )

            self.assertEqual(run_validation(), run_validation())
