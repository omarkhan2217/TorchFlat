"""Rolling median and sigma clipping."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from torchflat._utils import masked_median


def rolling_clip(
    flux: torch.Tensor,
    valid_mask: torch.Tensor,
    segment_id: torch.Tensor,
    sigma: float = 5.0,
    window_size: int = 25,
) -> torch.Tensor:
    """Rolling median + MAD-based sigma clipping for Track A.

    Computes a rolling median over *window_size* points, then clips points
    whose residual exceeds ``sigma * MAD / 0.6745``.

    Args:
        flux: ``[B, L]`` flux values.
        valid_mask: ``[B, L]`` boolean mask.
        segment_id: ``[B, L]`` segment labels (reserved for future use).
        sigma: Clipping threshold in sigma units.
        window_size: Rolling median window width in samples.

    Returns:
        ``[B, L]`` updated boolean mask with clipped points set to False.
    """
    # Rolling median via unfold
    windows = flux.unfold(dimension=1, size=window_size, step=1)  # [B, L-W+1, W]
    valid_windows = valid_mask.unfold(dimension=1, size=window_size, step=1)
    rolling_med = masked_median(windows, valid_windows)  # [B, L-W+1]

    # Pad edges with replicated values to restore original length
    pad_size = window_size // 2
    # F.pad expects (left, right) for last dim when input is 2-D
    rolling_med = F.pad(rolling_med, (pad_size, pad_size), mode="replicate")

    # Handle length mismatch if window_size is even
    if rolling_med.shape[1] > flux.shape[1]:
        rolling_med = rolling_med[:, : flux.shape[1]]
    elif rolling_med.shape[1] < flux.shape[1]:
        rolling_med = F.pad(rolling_med, (0, flux.shape[1] - rolling_med.shape[1]), mode="replicate")

    # Residuals and MAD
    residuals = (flux - rolling_med).abs()
    mad = masked_median(residuals, valid_mask)  # [B]

    # MAD -> sigma threshold  (MAD / 0.6745 ≈ std for normal)
    threshold = (sigma * mad.clamp(min=1e-10) / 0.6745).unsqueeze(1)  # [B, 1]

    clip_mask = residuals <= threshold
    return valid_mask & clip_mask


def conservative_clip(
    flux: torch.Tensor,
    valid_mask: torch.Tensor,
    sigma: float = 10.0,
) -> torch.Tensor:
    """Global median + MAD sigma clipping for Track B.

    Less aggressive than :func:`rolling_clip` — uses a global median
    and a wider threshold (default 10-sigma).

    Args:
        flux: ``[B, L]`` flux values.
        valid_mask: ``[B, L]`` boolean mask.
        sigma: Clipping threshold in sigma units.

    Returns:
        ``[B, L]`` updated boolean mask.
    """
    med = masked_median(flux, valid_mask)  # [B]
    residuals = (flux - med.unsqueeze(1)).abs()
    mad = masked_median(residuals, valid_mask)  # [B]
    threshold = (sigma * mad / 0.6745).unsqueeze(1)  # [B, 1]
    return valid_mask & (residuals <= threshold)
