"""Reference downscaling baselines.

* ``bilinear_interpolation`` -- P2 M225 / P1 M199 lower bound: ERA5 total
  precipitation bilinearly interpolated to the target grid. Every relative
  score in ``docs/EVALUATION_PROTOCOL.md`` is quoted against this.
* ``rainfarm`` -- P1 M195 stochastic spectral downscaling (pysteps). NOT yet
  implemented; raises so it is never silently skipped. Tracked ABSENT in
  ``docs/architecture_v3.md``.
"""

from __future__ import annotations

import numpy as np


def bilinear_interpolation(lr_field: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Bilinearly resample a 2D low-resolution field to ``out_shape`` (H, W)."""
    import torch
    import torch.nn.functional as F

    tensor = torch.as_tensor(np.asarray(lr_field, dtype=np.float32))[None, None]
    up = F.interpolate(tensor, size=out_shape, mode="bilinear", align_corners=False)
    return up[0, 0].numpy()


def rainfarm(*args, **kwargs):  # pragma: no cover - intentionally unimplemented
    raise NotImplementedError(
        "rainFARM (P1 M195, pysteps.noise.fftgenerators + spectral downscaling) "
        "is not implemented yet. Do not substitute another baseline silently."
    )
