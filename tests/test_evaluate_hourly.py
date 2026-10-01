"""Tests for stage 7 (evaluation) -- src/evaluation/evaluate_hourly.py --
reusing stage 6's synthetic prepared-region fixture to train tiny real
checkpoints and then evaluate them, exercising every metric call end to
end rather than re-testing the metric formulas themselves (those already
have their own test modules, e.g. tests/test_probabilistic_metrics.py)."""

from pathlib import Path
import tempfile
import unittest

from src.evaluation import evaluate_hourly as E
from src.evaluation.model_comparison import build_comparison_table
from src.training.train_baseline import BaselineTrainingConfig, train as train_baseline
from src.training.train_diffusion import (
    DiffusionStageTrainingConfig,
    RegressionStageTrainingConfig,
    train_diffusion_stage,
    train_regression_stage,
)
from src.training.train_gan import GanTrainingConfig, train as train_gan
from tests.test_training import _WithPreparedSyntheticRegion


def _assert_well_formed_report(case: unittest.TestCase, result: dict) -> None:
    case.assertIn("pooled", result)
    for key in ("r2", "rmse", "mae", "bias", "valid_pixels"):
        case.assertIn(key, result["pooled"])
    case.assertGreater(result["pooled"]["valid_pixels"], 0)
    case.assertIn("crps", result)
    case.assertIn("percentile_mae", result)
    case.assertIn(95.0, result["percentile_mae"])
    case.assertIn("mean_fss", result)
    case.assertIn("full_field_rmse", result)
    case.assertIn("full_field_ssim", result)
    case.assertIn("rank_histogram" if "rank_histogram" in result else "model_name", result)


class EvaluateBaselineTest(_WithPreparedSyntheticRegion):
    def test_report_is_well_formed_and_comparable(self):
        with tempfile.TemporaryDirectory() as out:
            config = BaselineTrainingConfig(
                region=self.REGION_NAME, epochs=1, batch_size=8, base_width=8, depth=2, device="cpu",
            )
            run_dir = train_baseline(config, prepared_root=self.prepared_root, output_root=Path(out))
            result = E.evaluate_baseline(
                self.REGION_NAME, "val", run_dir / "best.pt",
                prepared_root=self.prepared_root, device="cpu",
            )
            _assert_well_formed_report(self, result)
            self.assertEqual(result["model_name"], "deterministic")

            report_path = E.write_report(run_dir / "metrics_val.json", result)
            self.assertTrue(report_path.exists())

            run_result = E.to_model_run_result(result)
            table = build_comparison_table([run_result])
            self.assertEqual(len(table), 1)
            self.assertEqual(table[0]["model"], "deterministic")


class EvaluateGanTest(_WithPreparedSyntheticRegion):
    def test_report_is_well_formed(self):
        with tempfile.TemporaryDirectory() as out:
            config = GanTrainingConfig(region=self.REGION_NAME, epochs=1, batch_size=4, base_width=8, depth=2, device="cpu")
            run_dir = train_gan(config, prepared_root=self.prepared_root, output_root=Path(out))
            result = E.evaluate_gan(
                self.REGION_NAME, "val", run_dir / "best.pt",
                prepared_root=self.prepared_root, device="cpu", n_members=3,
            )
            _assert_well_formed_report(self, result)
            self.assertEqual(result["model_name"], "gan")
            self.assertEqual(len(result["rank_histogram"]), 4)  # n_members + 1


class EvaluateDiffusionTest(_WithPreparedSyntheticRegion):
    def test_report_is_well_formed(self):
        from src.models.diffusion_residual import HeunSamplerConfig

        with tempfile.TemporaryDirectory() as out:
            stage1_config = RegressionStageTrainingConfig(
                region=self.REGION_NAME, epochs=1, batch_size=8, base_width=8, device="cpu",
            )
            stage1_run_dir = train_regression_stage(stage1_config, prepared_root=self.prepared_root, output_root=Path(out))

            stage2_config = DiffusionStageTrainingConfig(
                region=self.REGION_NAME, stage1_checkpoint=str(stage1_run_dir / "best.pt"),
                epochs=1, batch_size=8, base_width=8, device="cpu", sampler_steps_for_validation=3,
            )
            stage2_run_dir = train_diffusion_stage(stage2_config, prepared_root=self.prepared_root, output_root=Path(out))

            result = E.evaluate_diffusion(
                self.REGION_NAME, "val", stage1_run_dir / "best.pt", stage2_run_dir / "best.pt",
                prepared_root=self.prepared_root, device="cpu", n_members=2,
                sampler_config=HeunSamplerConfig(num_steps=3),
            )
            _assert_well_formed_report(self, result)
            self.assertEqual(result["model_name"], "regression_diffusion")


class EvaluateBilinearBaselineTest(_WithPreparedSyntheticRegion):
    def test_report_is_well_formed(self):
        result = E.evaluate_bilinear_baseline(self.REGION_NAME, "val", build_root=self.prepared_root.parent / "build")
        self.assertIn("pooled", result)
        self.assertGreater(result["pooled"]["valid_pixels"], 0)
        self.assertEqual(result["model_name"], "bilinear_era5_baseline")
        self.assertIn("crps", result)
