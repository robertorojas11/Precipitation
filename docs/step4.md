# Step 4: Bias Correction, Inference & Evaluation

**Status:** Final step — requires trained GAN-1/GAN-2 models from Step 3

## Objective
Apply stochastic bias correction (Phase 7) to raw GAN outputs, run full inference for both pipelines, and evaluate Pipeline A (CHIRPS) vs. Pipeline B (Oya) using the agreed metrics to determine the superior ground-truth target for downscaling in Mexico.

## Prerequisites
- Step 3 verification checklist fully passed
- Trained GAN-2 checkpoints available for both Pipeline A and Pipeline B
- Validation/test split data preprocessed and held out (not used during training)

---

## 4.1 De-normalization
- Before bias correction, invert any Z-score / log transforms applied in Step 2 using the saved normalization statistics
- Confirm de-normalized GAN outputs are in physical precipitation units (e.g., mm/day) and within plausible ranges

## 4.2 Phase 7: Bias Correction & Stochastic Injection

### Step 1 — Quantile Mapping
- Map the empirical CDF of raw GAN-2 output to the empirical CDF of the ground-truth target (CHIRPS for Pipeline A, Oya for Pipeline B)
- Apply across the full precipitation distribution using the **training/validation period** statistics (do not fit on test data — avoid leakage)
- Apply the resulting mapping function to test-period outputs

### Step 2 — GPD Tail Correction
- Identify the threshold for "extreme" values (95th percentile of the ground-truth distribution)
- Fit a Generalized Pareto Distribution (GPD) to exceedances above this threshold, separately for each pipeline's ground truth
- Apply GPD-based correction to GAN outputs exceeding the threshold, replacing the quantile-mapped tail with GPD-corrected values
- Document threshold and fitted GPD parameters (shape, scale) for each pipeline

### Step 3 — Stochastic Noise Injection
- Generate spatially correlated Gaussian perturbations (e.g., using a spatial covariance/kernel matching the residual error structure observed in validation)
- Add perturbations to bias-corrected output to produce **multiple ensemble members** per time step
- Define the number of ensemble members to generate (e.g., 10–20) and document the choice
- Ensure perturbations preserve non-negativity of precipitation (clip or transform as needed)

### Implementation Notes
- Implement bias correction as a standalone, reusable Python module that takes (raw GAN output, target dataset name) and returns (bias-corrected deterministic field, ensemble of stochastic fields)
- This module should also support being switched between CHIRPS-fitted and Oya-fitted correction parameters via the same `target_dataset` flag used elsewhere

## 4.3 Inference Pipeline
For the held-out test period, for each pipeline (A and B):
1. Run ERA5 inputs through GAN-1 → ERA5-Land resolution output
2. Concatenate with Phase 3/4 physics fields and DEM channels
3. Run through GAN-2 → raw high-resolution precipitation field
4. De-normalize
5. Apply Phase 7 bias correction → deterministic bias-corrected field + stochastic ensemble

Save all intermediate and final outputs for the test period for both pipelines.

## 4.4 Evaluation Metrics

### Critical Success Index (CSI)
- `CSI = TP / (TP + FP + FN)` for precipitation event detection at relevant thresholds (e.g., >1mm, >10mm, >25mm)
- Compute globally over the Mexico domain and specifically over the **Sierra Madre Occidental** subregion
- Compute for both Pipeline A (vs. CHIRPS test data) and Pipeline B (vs. Oya test data)

### Kolmogorov-Smirnov (KS) Statistic
- `D = max|F_n(x) - F(x)|` comparing the distribution of bias-corrected output to the ground-truth distribution
- Compute for both pipelines, domain-wide and over the Sierra Madre subregion

### Spatial Relative Bias
- `Bias = (P_model - P_obs) / P_obs`
- Produce spatial bias maps for both pipelines
- Specifically examine windward vs. leeward bias patterns relative to the Sierra Madre Occidental

### Probabilistic Calibration
- Using the stochastic ensemble from Phase 7, compute:
  - Reliability diagrams
  - CRPS (Continuous Ranked Probability Score)
  - Spread-skill relationship
- Compute for both pipelines

## 4.5 Comparative Analysis (Pipeline A vs. Pipeline B)
- Produce a side-by-side comparison table of all metrics for Pipeline A and Pipeline B
- Highlight performance differences specifically over the Sierra Madre Occidental, since this is the focus region per the migration notes
- Note: Pipeline A and B are evaluated against **different** ground truths (CHIRPS vs. Oya respectively) — metrics are not directly comparable in absolute terms across pipelines for the same grid cells unless CHIRPS and Oya themselves are compared for the same period. Consider including a CHIRPS-vs-Oya agreement analysis for context.
- Summarize findings: which ground truth (CHIRPS or Oya) yields a downscaling model with better skill, better calibration, and more physically consistent orographic behavior?

## Verification Checklist
- [ ] Bias correction module implemented, tested, and target-agnostic (CHIRPS/Oya switchable)
- [ ] Full inference run completed for test period, both pipelines
- [ ] All four evaluation metrics (CSI, KS, Spatial Bias, Probabilistic Calibration) computed for both pipelines
- [ ] Sierra Madre Occidental subregion analysis completed for both pipelines
- [ ] Comparative summary table produced
- [ ] CHIRPS-vs-Oya contextual comparison considered/documented
- [ ] Final written conclusion on which ground truth performs better, with supporting evidence

## Output of This Step
Final bias-corrected, ensemble precipitation downscaling outputs for both pipelines, a complete evaluation report comparing CHIRPS vs. Oya as ground-truth targets, and a documented conclusion supporting the project's central research question.

## Next Step
None — this is the final step of the methodology. Results should inform any follow-on decisions (e.g., adopting the winning ground truth for production use, extending to 1 km resolution, etc.).
