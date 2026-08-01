# Stage 8 — Full candidate training

## Purpose

Short-search rankings can favor noisy early trajectories. This stage retrains
the two best configurations for up to 80 epochs with the full early-stopping
policy, still using validation—not test—for all decisions.

## Optimization objective

All terms use the intersection of target-valid and land masks. Wet occurrence
is defined as (y\ge1) mm/day. Binary cross entropy uses a dynamic positive
weight

\[
w_+=\operatorname{clip}
\left(\frac{N_{dry}}{N_{wet}},1,20\right).
\]

Positive amount uses masked Smooth L1 in log space. Physical L1 in mm/day
weights pixels as

\[
w_i=1+eI(y_i\ge10)+2eI(y_i\ge25),
\]

where (e) is the searched event weight. A neighbor-gradient term compares
horizontal and vertical precipitation differences only when both neighboring
pixels are valid. A multiscale term computes masked area-average L1 at factors
2, 5, and 10.

The total objective is

\[
L=L_{occ}+L_{amount}+L_{physical}
+0.1L_{gradient}+0.25L_{multiscale}.
\]

## Optimizer and control

Training uses AdamW with weight decay (10^{-4}), three linear warm-up epochs,
cosine annealing toward (10^{-6}), automatic mixed precision on CUDA,
gradient-norm clipping at 1.0, deterministic seeds/backends, and patience 10 on
validation R². Checkpoints include model, optimizer, scheduler, scaler, epoch,
configuration, normalization statistics, and dataset-manifest hash.

## Temporal-context fallback

If the best one-day candidate remains below validation R² 0.40, the same
parameters are trained with three consecutive atmospheric days (54 channels).
A center date is usable only when previous, current, and next dates all exist
inside the same split; context never crosses split boundaries.

Results are ranked into `candidate_results.json`. The winner supplies the
configuration and context length for final multi-seed training.
