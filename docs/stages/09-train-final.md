# Stage 9 — Final multi-seed training

## Purpose

One run can benefit from random initialization or minibatch order. Final
training freezes the winning candidate configuration and trains it independently
with seeds 17, 42, and 73. No hyperparameter changes are allowed after this
point.

## Reproducibility

For each seed, Python, NumPy, CPU PyTorch, and all CUDA generators are seeded.
cuDNN benchmarking is disabled and deterministic behavior enabled. The
DataLoader shuffle generator uses the same run seed. Run metadata stores the
configuration, training statistics, dataset-manifest SHA-256, Git commit, and
whether the worktree was dirty.

Each run directory contains:

- `run.json`: immutable configuration and provenance;
- `history.json`: epoch, average training loss, learning rate, validation R²,
  RMSE, MAE, and valid-pixel count;
- `last.pt`: latest complete optimizer state;
- `best.pt`: checkpoint with the highest validation R².

The run names are `final_seed17`, `final_seed42`, and `final_seed73`.
`final_runs.json` lists their paths and selected parameters.

## Ensemble definition

The three models are not selected individually on test data. During evaluation,
their deterministic physical predictions are combined pixelwise:

\[
\hat P_{ensemble}(x,t)=\frac{1}{3}
\sum_{m=1}^{3}\hat P_m(x,t).
\]

All members must name the same target and carry the same dataset-manifest hash;
otherwise evaluation refuses the ensemble.

## Connection to evaluation

Validation evaluation first confirms behavior of the frozen ensemble. Test
evaluation then opens held-out observations once for the final claim. Final
training itself produces no test metrics, bias maps, or post-processing.
