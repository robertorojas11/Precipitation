"""Model architectures for the v2-generative framework.

Two candidates (docs/decisions/0002-implementacion-v2.md, PASO 5), both
usable with the eta-learning regularizer (src/training/eta_learning.py):

- Candidate A (spateGAN-ERA5-inspired): generator.py + discriminator.py.
- Candidate B (Avila 2026 / Mardani et al.): regression_stage.py +
  diffusion_residual.py.

Plus a deterministic baseline (deterministic_resnet.py, PASO 5.4).
"""
