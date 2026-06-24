# Step 3: GAN Training

**Status:** Follows Step 2 — requires preprocessed target-agnostic dataset

## Objective
Train two GAN stages (GAN-1 and GAN-2) per pipeline, producing separate networks for Pipeline A (CHIRPS) and Pipeline B (Oya), using the identical architecture and training procedure with only the ground-truth target swapped.

## Prerequisites
- Step 2 verification checklist fully passed
- Preprocessed dataset available with target-agnostic loader (`target_dataset = "chirps" | "oya"`)
- Pre-computed physics fields (Phase 3 + Phase 4) available as conditioning channels

---

## 3.1 General Training Strategy
- Train **four networks total**: GAN-1 and GAN-2, each for Pipeline A and Pipeline B (2 × 2)
- GAN-1 is **target-independent** in principle (ERA5 → ERA5-Land does not depend on CHIRPS/Oya) — confirm whether a single shared GAN-1 can be reused for both pipelines, or whether separate copies are trained for consistency with the "separate networks per pipeline" requirement in the migration notes. Document the decision.
- GAN-2 is **target-dependent** — must be trained separately for Pipeline A (CHIRPS) and Pipeline B (Oya)
- Use identical hyperparameters, architecture, and training schedule across both pipelines to ensure a fair comparison — only the ground-truth tensor and any target-specific normalization stats differ

## 3.2 GAN-1: ERA5 → ERA5-Land (Phase 5)

### Architecture
- ESRGAN-style generator with Residual-in-Residual Dense Blocks (RRDB)
- Discriminator: PatchGAN-style
- Upsampling: sub-pixel convolution, 0.25° → 0.1° (~×2.5)

### Inputs
- ERA5 precipitation + wind variable stack (normalized)

### Target
- ERA5-Land precipitation (normalized)

### Loss Function
Combine the following (weights to be tuned):
- Pixel loss (L1) between generated and target ERA5-Land precipitation
- Perceptual loss (VGG-based feature comparison)
- Adversarial loss (standard or relativistic GAN loss against PatchGAN discriminator)

### Training Notes
- Train to convergence before starting GAN-2 — GAN-1 output becomes GAN-2 input
- Save model checkpoints and training curves (generator loss, discriminator loss, validation L1) for later review

## 3.3 GAN-2: Super-Resolution Stage (Phase 6)

### Architecture
- RRDB-based generator (same family as GAN-1), upsampling to final target resolution
- Upscaling factor: ERA5-Land (0.1°) → 5 km (~×2) or → 1 km (~×10), depending on experiment configuration

### Inputs (concatenated channels)
- GAN-1 output (generated ERA5-Land precipitation)
- Phase 3 output: Upslope physics field
- Phase 4 output: Spectral physics field
- DEM / terrain channels at target resolution

### Target
- **Pipeline A:** CHIRPS precipitation at 5 km
- **Pipeline B:** Oya precipitation at 5 km

### Loss Function
Same combination as GAN-1 (pixel + perceptual + adversarial), computed against the pipeline-specific ground truth

### Training Notes
- Train Pipeline A and Pipeline B GAN-2 models independently but with identical configuration
- Log per-pipeline training curves separately for direct comparison
- Save checkpoints for both pipeline variants with clear naming (e.g., `gan2_pipelineA_chirps.pt`, `gan2_pipelineB_oya.pt`)

## 3.4 Training Configuration Documentation
For reproducibility, record for each trained model:
- Architecture hyperparameters (number of RRDB blocks, channel counts, upscaling factor)
- Loss weights (pixel / perceptual / adversarial)
- Optimizer, learning rate, schedule, batch size, number of epochs
- Train/validation split (time periods used)
- Random seed
- Hardware/runtime used (for reference)

## 3.5 Sanity Checks During Training
- Visualize generated precipitation fields vs. ground truth on a validation sample every N epochs
- Confirm GAN-2 outputs show plausible orographic enhancement consistent with Phase 3/4 physics fields (e.g., higher precipitation on windward slopes of the Sierra Madre)
- Monitor for mode collapse or discriminator dominance (common GAN failure modes) — adjust loss weights or learning rates if observed

## Verification Checklist
- [ ] GAN-1 trained and checkpointed (shared or per-pipeline, per documented decision)
- [ ] GAN-2 trained and checkpointed separately for Pipeline A (CHIRPS) and Pipeline B (Oya)
- [ ] Training configuration documented for all models
- [ ] Validation losses logged and reviewed — no signs of divergence or mode collapse
- [ ] Spot-check visualizations confirm physically plausible orographic patterns
- [ ] All four model checkpoints (or GAN-1 shared + 2× GAN-2) saved with clear naming convention

## Output of This Step
Trained GAN-1 and GAN-2 models for both pipelines, with checkpoints and documented configurations, producing raw super-resolved precipitation fields (pre-bias-correction) at the target resolution for both CHIRPS and Oya ground truths.

## Next Step
Proceed to **step4.md** — Bias Correction, Inference, and Evaluation.
