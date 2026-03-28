"""Segment-aware batched Tukey biweight detrending kernel."""

from __future__ import annotations

import torch

from torchflat._utils import MIN_SEGMENT_LENGTH, masked_median


def biweight_detrend(
    flux: torch.Tensor,
    time: torch.Tensor,
    valid_mask: torch.Tensor,
    segment_id: torch.Tensor,
    window_length_days: float = 0.5,
    n_iter: int = 5,
    cval: float = 5.0,
    min_segment_points: int = MIN_SEGMENT_LENGTH,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Segment-aware batched Tukey biweight detrending.

    For each sliding window position, computes a robust location estimate
    using Tukey's biweight iteration.  Segment IDs prevent the window from
    crossing data gaps.

    Args:
        flux: ``[B, L]`` flux values.
        time: ``[B, L]`` timestamps.
        valid_mask: ``[B, L]`` boolean mask (True = valid point).
        segment_id: ``[B, L]`` int32 segment labels.
        window_length_days: Sliding window width in days (default 0.5 = 12 h).
        n_iter: Number of biweight iterations.  5 converges to within 1e-6
            of the 10-iteration result for typical TESS data.
        cval: Biweight c parameter (rejection threshold in MAD units).
        min_segment_points: Minimum valid points in a window for a valid trend.
        dtype: Computation dtype (float32 or float64).

    Returns:
        detrended: ``[B, L]`` detrended flux (``flux / trend``).
            NaN where the trend is invalid.
        trend: ``[B, L]`` estimated trend.  NaN at edges and invalid regions.
    """
    B, L = flux.shape
    device = flux.device

    # ------------------------------------------------------------------
    # Step 0: Compute window size in samples from median cadence
    # ------------------------------------------------------------------
    dt = time[:, 1:] - time[:, :-1]  # [B, L-1]
    dt_valid = valid_mask[:, 1:] & valid_mask[:, :-1]
    median_cadence = masked_median(dt, dt_valid)  # [B]

    win_samples = (window_length_days / median_cadence.clamp(min=1e-10)).round().long()  # [B]
    W = int(win_samples.median().item())
    W = max(W, 3)  # need at least 3 for a meaningful window
    W = W | 1  # ensure odd for symmetric centering

    if L < W:
        # Light curves shorter than the window -> all NaN
        nan_tensor = torch.full((B, L), float("nan"), dtype=flux.dtype, device=device)
        return nan_tensor, nan_tensor.clone()

    N_pos = L - W + 1

    # ------------------------------------------------------------------
    # Step 1: Unfold into sliding windows (contiguous copies)
    # ------------------------------------------------------------------
    flux_compute = flux.to(dtype)
    flux_windows = flux_compute.unfold(dimension=1, size=W, step=1).contiguous()  # [B, N_pos, W]
    valid_windows = valid_mask.unfold(dimension=1, size=W, step=1).contiguous()   # [B, N_pos, W]
    seg_windows = segment_id.unfold(dimension=1, size=W, step=1).contiguous()     # [B, N_pos, W]

    # ------------------------------------------------------------------
    # Step 2: Segment-aware window validity mask
    # ------------------------------------------------------------------
    center_seg = seg_windows[:, :, W // 2 : W // 2 + 1]  # [B, N_pos, 1]
    window_valid = valid_windows & (seg_windows == center_seg)  # [B, N_pos, W]

    # ------------------------------------------------------------------
    # Step 3: Initial location estimate via masked median
    # ------------------------------------------------------------------
    location = masked_median(flux_windows, window_valid)  # [B, N_pos]

    # ------------------------------------------------------------------
    # Step 4-5: Biweight iteration
    # Compute MAD once from the initial median, then iterate with fixed
    # MAD.  Profiling showed MAD re-computation (sort) is ~60% of
    # iteration cost; fixing MAD gives 2.6x speedup with <1e-6 error.
    # ------------------------------------------------------------------
    abs_dev = (flux_windows - location.unsqueeze(-1)).abs()
    mad = masked_median(abs_dev, window_valid)  # [B, N_pos]
    safe_mad = (cval * mad.clamp(min=1e-10)).unsqueeze(-1)  # [B, N_pos, 1]

    for _ in range(n_iter):
        u = (flux_windows - location.unsqueeze(-1)) / safe_mad  # [B, N_pos, W]

        # Biweight weights: (1 - u^2)^2 for |u| < 1, else 0
        weights = ((1.0 - u**2) ** 2) * (u.abs() < 1.0).to(dtype) * window_valid.to(dtype)

        # Weighted mean -> new location
        w_sum = weights.sum(dim=-1).clamp(min=1e-10)  # [B, N_pos]
        location = (flux_windows * weights).sum(dim=-1) / w_sum  # [B, N_pos]

    # ------------------------------------------------------------------
    # Step 6: Map location back to full-length trend array
    # ------------------------------------------------------------------
    trend = torch.full((B, L), float("nan"), dtype=dtype, device=device)
    offset = W // 2
    trend[:, offset : offset + N_pos] = location

    # Guard: positions with too few valid points -> NaN
    n_valid_per_window = window_valid.sum(dim=-1)  # [B, N_pos]
    insufficient = n_valid_per_window < min_segment_points
    trend[:, offset : offset + N_pos] = torch.where(
        insufficient,
        torch.tensor(float("nan"), dtype=dtype, device=device),
        trend[:, offset : offset + N_pos],
    )

    # ------------------------------------------------------------------
    # Step 7: Detrend
    # ------------------------------------------------------------------
    trend_out = trend.to(flux.dtype)
    detrended = torch.where(
        (trend_out > 0) & valid_mask & torch.isfinite(trend_out),
        flux / trend_out,
        torch.tensor(float("nan"), dtype=flux.dtype, device=device),
    )

    return detrended, trend_out
