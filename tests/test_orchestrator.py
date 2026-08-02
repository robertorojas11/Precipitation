import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import pipeline
from src.training.evaluate import _json_default
from src.utils.storage import probe_directory


class PipelineOrchestratorTests(unittest.TestCase):
    def test_stage_range_is_ordered(self):
        arguments = type("Arguments", (), {
            "stages": None,
            "from_stage": "build",
            "to_stage": "prepare",
        })()
        self.assertEqual(
            pipeline._selected_stages(arguments),
            ("build", "validate_processed", "prepare"),
        )

    def test_oya_acquisition_uses_requested_years(self):
        options = pipeline.PipelineOptions(
            targets=("oya",),
            stages=("acquire",),
            start_year=2010,
            end_year=2012,
            device="cpu",
            batch_size=1,
            num_workers=0,
            resume=True,
            dry_run=True,
            continue_on_error=False,
            minimum_free_gib=0.0,
        )
        command = pipeline._commands("oya", "acquire", options)[0]
        self.assertIn("2010", command)
        self.assertIn("2012", command)

    def test_chirps_acquisition_is_not_reexported(self):
        options = pipeline.PipelineOptions(
            targets=("chirps",),
            stages=("acquire",),
            start_year=2004,
            end_year=2025,
            device="cpu",
            batch_size=1,
            num_workers=0,
            resume=True,
            dry_run=True,
            continue_on_error=False,
            minimum_free_gib=0.0,
        )
        self.assertEqual(pipeline._commands("chirps", "acquire", options), [])

    def test_storage_probe_verifies_and_cleans_up(self):
        with TemporaryDirectory() as directory:
            probe = probe_directory(directory, minimum_free_gib=0.0)
            self.assertTrue(probe.healthy)
            self.assertTrue(probe.integrity_verified)

    def test_failed_validation_artifact_is_not_resumed(self):
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "raw.json"
            artifact.write_text('{"accepted": false}')
            self.assertFalse(
                pipeline._artifact_is_complete("chirps", "validate_raw", artifact)
            )
            artifact.write_text('{"accepted": true}')
            self.assertTrue(
                pipeline._artifact_is_complete("chirps", "validate_raw", artifact)
            )

    def test_truncated_evaluation_is_not_resumed(self):
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "metrics_val.json"
            artifact.write_text('{"target": "chirps",')
            self.assertFalse(
                pipeline._artifact_is_complete(
                    "chirps", "evaluate_validation", artifact
                )
            )

    def test_numpy_scalars_are_json_serializable(self):
        import numpy as np

        payload = __import__("json").dumps(
            {"accepted": np.bool_(True), "score": np.float64(0.4)},
            default=_json_default,
        )
        self.assertEqual(
            __import__("json").loads(payload),
            {"accepted": True, "score": 0.4},
        )
