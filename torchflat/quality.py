"""Quality bitmask filtering."""

from __future__ import annotations

import torch

from torchflat._utils import QUALITY_BITMASK


def quality_filter(
    flux: torch.Tensor,
    time: torch.Tensor,
    quality: torch.Tensor,
    bitmask: int = QUALITY_BITMASK,
) -> torch.Tensor:
    """Apply quality bitmask and finite-value filtering.

    Args:
        flux: ``[B, L]`` flux values.
        time: ``[B, L]`` timestamps.
        quality: ``[B, L]`` integer quality flags.
        bitmask: Bitmask of bad-quality bits.  Default is the TESS bitmask.

    Returns:
        ``[B, L]`` boolean tensor.  True = good data point.
    """
    valid = (quality & bitmask) == 0
    valid &= torch.isfinite(flux)
    valid &= torch.isfinite(time)
    return valid
