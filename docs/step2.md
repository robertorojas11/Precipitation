# Step 2: Data Preprocessing & Physical Models

**Status:** Next step — start here once Step 1 verification checklist is complete

## Objective
Transform raw acquired data into a clean, normalized, target-agnostic dataset, and pre-compute the physical orographic model outputs (Phases 3 & 4 of the pipeline) that will later condition the GANs.

## Prerequisites
- Step 1 verification checklist fully passed
- Raw data archive accessible for: ERA5, ERA5-Land, CHIRPS, Oya, DEM

---

## 2.1 Spatial & Temporal Alignment

1. **Regrid all datasets to common reference grids:**
   - Low-res grid: ERA5 native 0.25°
   - Mid-res grid: ERA5-Land native 0.1°
   - High-res grid: 0.05° (5 km) — shared by CHIRPS, Oya, and final outputs
   - Optional ultra-high-res grid: 1 km (only if Pipeline targets 1 km)
2. **Temporal alignment:**
   - Resample Oya (30-min) to daily totals to match CHIRPS and ERA5 temporal resolution for primary training
   - Retain a separate sub-daily Oya dataset if sub-daily experiments are planned later
   - Align all datasets to the same calendar (UTC, consistent date boundaries)
3. **Crop all datasets to the exact Mexico bounding box** (12°N–35°N, 84°W–120°W)

## 2.2 Quality Control
- Identify and flag/interpolate missing time steps
- Remove or flag unrealistic precipitation values (negative values, extreme outliers beyond physical plausibility)
- Cross-check CHIRPS vs Oya for gross discrepancies in overlapping periods (informational only — do not reconcile, both are valid targets for their respective pipelines)

## 2.3 Normalization (Z-score)
- Compute per-variable mean and standard deviation **separately for**:
  - ERA5 input variables (precipitation, winds, geopotential)
  - ERA5-Land precipitation
  - CHIRPS precipitation (Pipeline A)
  - Oya precipitation (Pipeline B)
- Apply Z-score normalization: `z = (x - mean) / std`
- Store normalization statistics (mean, std) per variable — required for de-normalizing model outputs later
- Consider log-transform before Z-score for precipitation (heavy-tailed distribution) — document whichever choice is made, since it must be consistently inverted in Step 4

## 2.4 Target-Agnostic Dataset Structure
Per the Migration Notes, the codebase must support switching between CHIRPS and Oya via a configuration flag. Structure preprocessed data so that:
- A single loader interface accepts a `target_dataset` parameter (`"chirps"` or `"oya"`)
- Input tensors (ERA5, ERA5-Land, physics fields) are identical across both pipelines
- Only the high-res ground-truth tensor differs between Pipeline A and Pipeline B
- Train/validation/test splits are defined once and reused identically for both pipelines (same time periods) to ensure a fair comparison

## 2.5 Physical Model Pre-Computation

### Phase 3 — Upslope Physics Model
- **Inputs:** ERA5 pressure-level winds (u, v, w) + DEM gradients (∇z)
- **Computation:** Upslope moisture flux, `w_orographic = V · ∇z`, computed at each grid point and time step
- **Output:** Orographic precipitation enhancement field, regridded to 5 km (and 1 km if applicable)
- **Storage:** Save as a separate tensor/array aligned with the same spatial-temporal grid as other inputs

### Phase 4 — Spectral Physics Model
- **Inputs:** ERA5 pressure-level winds + terrain elevation (Fourier-transformed)
- **Computation:** Smith & Barstad (2004) linear theory model in the Fourier domain — wave-number decomposition of orographic forcing
- **Output:** Spectral precipitation field, regridded to 5 km (and 1 km if applicable)
- **Storage:** Save as a separate tensor/array aligned with the same spatial-temporal grid

### Combining Physical Fields
- Both physical model outputs (Phase 3 + Phase 4) should be stored as additional channels, ready to be concatenated with GAN inputs during Step 3 (used as conditioning channels for GAN-2)

## 2.6 Final Preprocessed Dataset Layout
For each time step, the following aligned tensors should exist:
- ERA5 input stack (precipitation + wind variables), 0.25°
- ERA5-Land precipitation, 0.1° (GAN-1 target)
- Upslope physics field, 5 km
- Spectral physics field, 5 km
- DEM / terrain channels, 5 km (and 1 km if applicable)
- CHIRPS precipitation, 5 km (Pipeline A target)
- Oya precipitation, 5 km (Pipeline B target)
- Normalization statistics file (JSON or similar) documenting mean/std and any transforms applied

## Verification Checklist
- [ ] All datasets share consistent spatial grids at each resolution tier
- [ ] Temporal alignment confirmed — no offset between datasets
- [ ] Z-score statistics computed and saved for all variables
- [ ] Target-agnostic loader implemented and tested with both `"chirps"` and `"oya"` flags
- [ ] Train/validation/test split defined once, identical across pipelines
- [ ] Phase 3 (Upslope) and Phase 4 (Spectral) physics fields computed and saved
- [ ] Physics fields validated against expected patterns (e.g., enhancement on windward slopes of Sierra Madre)

## Output of This Step
A preprocessed, normalized, target-agnostic dataset with pre-computed physical model fields, ready for GAN training in Step 3.

## Next Step
Proceed to **step3.md** — GAN Training.
