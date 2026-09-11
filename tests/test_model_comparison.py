"""Tests for the cross-model comparison harness
(src/evaluation/model_comparison.py), PASO 8."""

import unittest

from src.evaluation.model_comparison import (
    ModelRunResult,
    build_comparison_table,
    eta_learning_contribution,
    render_markdown_table,
)


def _results():
    return [
        ModelRunResult("deterministic", False, {"crps": 2.0, "rmse": 3.0}),
        ModelRunResult("deterministic", True, {"crps": 1.8, "rmse": 3.1}),
        ModelRunResult("gan", False, {"crps": 1.5, "rmse": 2.5}),
        ModelRunResult("gan", True, {"crps": 1.2, "rmse": 2.6}),
        ModelRunResult("regression_diffusion", False, {"crps": 1.4, "rmse": 2.2}),
        ModelRunResult("regression_diffusion", True, {"crps": 1.1, "rmse": 2.3}),
    ]


class BuildComparisonTableTest(unittest.TestCase):
    def test_one_row_per_model_and_eta_flag_sorted(self):
        rows = build_comparison_table(_results())
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            [(r["model"], r["eta_enabled"]) for r in rows],
            [
                ("deterministic", False), ("deterministic", True),
                ("gan", False), ("gan", True),
                ("regression_diffusion", False), ("regression_diffusion", True),
            ],
        )


class EtaLearningContributionTest(unittest.TestCase):
    def test_positive_delta_means_eta_helped_for_lower_is_better_metric(self):
        contributions = eta_learning_contribution(_results(), metrics=["crps", "rmse"])
        # crps improves (lower) with eta for every model -> positive contribution
        for model in ("deterministic", "gan", "regression_diffusion"):
            self.assertGreater(contributions[model]["crps"], 0.0)
        # rmse got worse (higher) with eta for every model -> negative contribution
        for model in ("deterministic", "gan", "regression_diffusion"):
            self.assertLess(contributions[model]["rmse"], 0.0)

    def test_missing_pair_is_skipped(self):
        results = [ModelRunResult("gan", False, {"crps": 1.0})]  # no eta=True run
        contributions = eta_learning_contribution(results, metrics=["crps"])
        self.assertNotIn("gan", contributions)

    def test_exact_delta_values(self):
        contributions = eta_learning_contribution(_results(), metrics=["crps"])
        # deterministic: without=2.0, with=1.8 -> delta = 2.0 - 1.8 = 0.2
        self.assertAlmostEqual(contributions["deterministic"]["crps"], 0.2, places=6)


class RenderMarkdownTableTest(unittest.TestCase):
    def test_renders_header_and_rows(self):
        rows = build_comparison_table(_results())[:2]
        table = render_markdown_table(rows)
        self.assertIn("| model |", table)
        self.assertIn("deterministic", table)
        self.assertEqual(table.count("\n"), 3)  # header + separator + 2 rows - 1

    def test_empty_rows_gives_empty_string(self):
        self.assertEqual(render_markdown_table([]), "")


if __name__ == "__main__":
    unittest.main()
