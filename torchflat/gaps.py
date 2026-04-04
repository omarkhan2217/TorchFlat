"""Gap detection and linear interpolation."""

from __future__ import annotations

import torch

from torchflat._utils import GAP_THRESHOLD


def detect_gaps(
    time: torch.Tensor,
    valid_mask: torch.Tensor,
    gap_threshold: float = GAP_THRESHOLD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Detect gaps in time series and assign segment IDs.

    Args:
        time: ``[B, L]`` timestamps.
        valid_mask: ``[B, L]`` boolean mask (True = valid).
        gap_threshold: Gaps with ``dt / median_cadence >= threshold`` are
            treated as segment boundaries.

    Returns:
        segment_id: ``[B, L]`` int32 tensor (0, 0, 0, 1, 1, 2, ...).
        median_cadence: ``[B]`` per-star median cadence in the same time units
            as *time*.
    """
    B, L = time.shape
    device = time.device

    dt = time[:, 1:] - time[:, :-1]  # [B, L-1]

    # Median cadence: use torch.median directly (avoids expensive masked sort).
    # Post-quality-filter, most dt values are valid and nearly uniform.
    # torch.median on the raw dt gives the correct cadence for uniform data.
    median_cadence = torch.median(dt, dim=-1).values  # [B]

    # Avoid division by zero for stars with < 2 valid points
    safe_cadence = median_cadence.clamp(min=1e-10)
    gap_ratio = dt / safe_cadence.unsqueeze(1)  # [B, L-1]

    is_large_gap = gap_ratio >= gap_threshold  # [B, L-1]

    segment_id = torch.cat(
        [
            torch.zeros(B, 1, device=device, dtype=torch.int32),
            is_large_gap.to(torch.int32).cumsum(dim=1),
        ],
        dim=1,
    )  # [B, L]

    return segment_id, median_cadence


def interpolate_small_gaps(
    flux: torch.Tensor,
    time: torch.Tensor,
    valid_mask: torch.Tensor,
    max_gap: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linearly interpolate small gaps (<= *max_gap* consecutive invalid points).

    Fully vectorized — no per-star Python loops.

    Args:
        flux: ``[B, L]`` flux values.
        time: ``[B, L]`` timestamps.
        valid_mask: ``[B, L]`` boolean mask.
        max_gap: Maximum gap size (in cadences) to interpolate.

    Returns:
        flux_out: ``[B, L]`` with interpolated values at small-gap positions.
        valid_mask_out: ``[B, L]`` with interpolated positions set to True.
    """
    B, L = flux.shape
    device = flux.device

    flux_out = flux.clone()
    valid_out = valid_mask.clone()
    invalid = ~valid_mask  # [B, L]

    if not invalid.any():
        return flux_out, valid_out

    # For each invalid position, compute distance to the nearest valid position
    # on the left and on the right. Then we can compute gap size and interpolation
    # fraction in one vectorized pass.

    # Left boundary: for each position, index of the last valid point to the left.
    idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)  # [B, L]
    # Set invalid positions to -1 so they don't win cummax
    left_idx = idx.clone()
    left_idx[invalid] = -1
    left_idx = left_idx.cummax(dim=1).values  # [B, L]

    # Right boundary: for each position, index of the next valid point to the right.
    # Set invalid positions to L (large) so they don't win cummin, then reverse-cummin.
    right_idx = idx.clone()
    right_idx[invalid] = L
    right_idx = right_idx.flip(1).cummin(dim=1).values.flip(1)  # [B, L]

    # Gap size: right_idx - left_idx - 1 (number of consecutive invalid points)
    gap_size = right_idx - left_idx - 1  # [B, L]

    # Which positions to interpolate: invalid, with valid neighbors on both sides,
    # and gap size <= max_gap
    can_interp = invalid & (left_idx >= 0) & (right_idx < L) & (right_idx >= 0)
    can_interp = can_interp & (gap_size <= max_gap) & (gap_size > 0)

    if not can_interp.any():
        return flux_out, valid_out

    # Gather left and right flux values using the boundary indices
    # Clamp indices to valid range for gather (positions that won't be used are masked)
    left_safe = left_idx.clamp(min=0)
    right_safe = right_idx.clamp(min=0, max=L - 1)

    f_left = flux.gather(1, left_safe)   # [B, L]
    f_right = flux.gather(1, right_safe)  # [B, L]

    # Interpolation fraction: (position - left_idx) / (right_idx - left_idx)
    span = (right_idx - left_idx).float().clamp(min=1)
    frac = (idx - left_idx).float() / span  # [B, L]

    # Interpolate
    interp_vals = f_left + frac * (f_right - f_left)

    # Write only at positions that should be interpolated
    flux_out[can_interp] = interp_vals[can_interp]
    valid_out[can_interp] = True

    return flux_out, valid_out
