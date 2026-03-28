"""Shared utilities: masked median, padding helpers, constants."""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# TESS quality bitmask - bits that indicate bad data
# Bits: 1(attitude tweak), 2(safe mode), 4(coarse point), 8(Earth point),
#       16(desat), 32(1-sec data), 128(manual exclude), 1024(stray light),
#       2048(impulsive outlier)
QUALITY_BITMASK: int = 0b0000110101111111  # = 3455

MIN_POINTS: int = 100           # Minimum valid points for a star to be processed
GAP_THRESHOLD: float = 5.0      # Gap ratio above this = large gap (segment boundary)
MIN_SEGMENT_LENGTH: int = 50    # Minimum valid points in a biweight window's segment


# ---------------------------------------------------------------------------
# Masked median
# ---------------------------------------------------------------------------

def masked_median(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute median along the last dimension, only over positions where *mask* is True.

    Matches numpy.median convention for even-length arrays (averages two
    middle values).

    Args:
        x: Tensor of any shape ``[..., N]``.
        mask: Boolean tensor, same shape as *x*.  True = valid.

    Returns:
        Tensor of shape ``[...]`` with the median of valid values along the
        last dimension.  Returns NaN where no valid values exist.
    """
    if x.shape[-1] == 0:
        out_shape = x.shape[:-1]
        return torch.full(out_shape, float("nan"), dtype=x.dtype, device=x.device)

    W = x.shape[-1]

    # Clone and push invalid values to +inf so they sort last
    working = x.clone()
    working[~mask] = float("inf")

    n_valid = mask.sum(dim=-1)  # [...]

    # Sort ascending: valid values first, +inf values last
    sorted_vals = torch.sort(working, dim=-1).values

    # Median indices (numpy even-length convention: average two middle values)
    mid_lo = ((n_valid - 1) // 2).clamp(min=0)
    mid_hi = (n_valid // 2).clamp(min=0)

    val_lo = sorted_vals.gather(-1, mid_lo.unsqueeze(-1)).squeeze(-1)
    val_hi = sorted_vals.gather(-1, mid_hi.unsqueeze(-1)).squeeze(-1)
    median = (val_lo + val_hi) / 2.0

    # Guard: no valid data -> NaN
    median = median.where(n_valid > 0, torch.tensor(float("nan"), dtype=median.dtype, device=median.device))

    return median


# ---------------------------------------------------------------------------
# Padding helper
# ---------------------------------------------------------------------------

def pad_to_length(
    tensors: list[torch.Tensor],
    target_len: int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Right-pad a list of 1-D tensors to *target_len* and stack into 2-D.

    Args:
        tensors: List of 1-D tensors (lengths ``<= target_len``).
        target_len: Length to pad to.
        pad_value: Fill value for padding positions.

    Returns:
        Tensor of shape ``[len(tensors), target_len]``.
    """
    B = len(tensors)
    out = torch.full(
        (B, target_len),
        fill_value=pad_value,
        dtype=tensors[0].dtype,
        device=tensors[0].device,
    )
    for i, t in enumerate(tensors):
        out[i, : t.shape[0]] = t
    return out
