"""Land mask utility for the Mexico precipitation downscaling domain.

Generates a binary land mask (1=land, 0=ocean/water) from NASADEM elevation data
for the Mexico domain at 5km (0.05 deg) resolution: 460 rows x 720 columns.

The mask is cached in memory after the first call to avoid repeated I/O.
"""

import os
import numpy as np

_MASK_CACHE = None


def get_land_mask(device=None):
    """Returns a (1, 1, 460, 720) float32 tensor: 1.0 = land, 0.0 = ocean.

    Derived from NASADEM elevation: pixels with elevation > -100m are treated
    as land (this threshold retains below-sea-level land like coastal deltas
    while masking open ocean).

    Falls back to an all-ones mask (no masking) if NASADEM is not available.

    Args:
        device (torch.device, optional): Device to move the tensor to.

    Returns:
        torch.Tensor: Shape (1, 1, 460, 720), dtype float32.
    """
    global _MASK_CACHE
    if _MASK_CACHE is not None:
        return _MASK_CACHE.to(device) if device is not None else _MASK_CACHE

    import sys
    import torch
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

    from src.utils.config import Config

    dem_path = os.path.join(Config.RAW_DATA_DIR, "dem", "nasadem_mexico_1km.tif")

    if os.path.exists(dem_path):
        try:
            from src.data_preprocessing.physics_models import resample_dem
            elevation = resample_dem(dem_path, output_height=460, output_width=720)
            # Land = elevation > -100m (retains coastal lowlands, masks open ocean)
            mask = (elevation > -100.0).astype(np.float32)
        except Exception as e:
            import logging
            logging.getLogger("PrecipitationPipeline").warning(
                f"Failed to build land mask from NASADEM: {e}. Falling back to all-ones mask."
            )
            mask = np.ones((460, 720), dtype=np.float32)
    else:
        import logging
        logging.getLogger("PrecipitationPipeline").warning(
            f"NASADEM not found at {dem_path}. Using all-ones land mask (no ocean masking)."
        )
        mask = np.ones((460, 720), dtype=np.float32)

    # Shape: (1, 1, 460, 720) for broadcasting against (B, 1, H, W) model outputs
    _MASK_CACHE = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return _MASK_CACHE.to(device) if device is not None else _MASK_CACHE
