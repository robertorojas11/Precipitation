# Stage 11 — Held-out test evaluation

## Scientific role

Test evaluation uses the same code, masks, formulas, baselines, ensemble, and
bootstrap procedure as validation evaluation, but applies them to dates never
used for normalization, climatology fitting, hyperparameter choice, early
stopping, architecture selection, or seed selection.

The command receives all three frozen run directories. It rejects mixed targets
and mismatched dataset-manifest hashes before inference.

## Acceptance gates

The generated `metrics_test.json` records five explicit decisions:

1. pooled model R² ≥ 0.40;
2. every test-year R² ≥ 0.20;
3. lower 95% month-block bootstrap bound ≥ 0.35;
4. model R² exceeds ERA5 R²;
5. model R² exceeds monthly-climatology R².

The first threshold is the requested project goal, while the other gates guard
against a pooled score hiding weak years, statistical uncertainty, or failure
to add value over simple predictors.

## Interpretation

CHIRPS and Oya receive separate results. Passing one target does not imply
passing the other because their observation systems, coverage, errors,
statistics, and climatologies differ. A failed gate is retained and reported;
the workflow must not tune thresholds, masks, transformations, or
post-processing against test outcomes.

Bias correction is not part of the maintained acceptance result. The evaluated
quantity is the raw arithmetic ensemble of deterministic model predictions.

## Connection to reporting

The report stage consumes these frozen predictions and metrics only to produce
spatial diagnostics and figures. It cannot change checkpoints or scores.
