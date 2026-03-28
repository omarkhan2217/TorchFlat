"""Normalization operations for Track A and Track B."""

from __future__ import annotations

import torch

from torchflat._utils import masked_median


def normalize_track_a(
    flux: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Track A normalization: median-divide, subtract 1, clamp.

    Transit dips become negative values.

    Args:
        flux: ``[B, L]`` detrended flux.
        valid_mask: ``[B, L]`` boolean mask.

    Returns:
        ``[B, L]`` normalized flux clamped to ``[-0.5, 0.5]``.
    """
    med = masked_median(flux, valid_mask)  # [B]
    normalized = (flux / med.unsqueeze(1)) - 1.0
    normalized = normalized.clamp(-0.5, 0.5)
    normalized[~valid_mask] = 0.0
    return normalized


def normalize_track_b(
    flux: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Track B normalization: MAD-based robust z-score.

    Args:
        flux: ``[B, L]`` highpass-filtered flux.
        valid_mask: ``[B, L]`` boolean mask.

    Returns:
        ``[B, L]`` robust z-score normalized flux.
    """
    center = masked_median(flux, valid_mask)  # [B]
    abs_dev = (flux - center.unsqueeze(1)).abs()
    mad = masked_median(abs_dev, valid_mask)  # [B]
    # MAD -> sigma estimate (1.4826 factor for normal distributions)
    scale = (mad * 1.4826).clamp(min=1e-8).unsqueeze(1)  # [B, 1]
    normalized = (flux - center.unsqueeze(1)) / scale
    normalized[~valid_mask] = 0.0
    return normalized
