import unittest
from tempfile import TemporaryDirectory

import pipeline
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
