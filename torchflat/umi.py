"""UMI -- Unified Median Iterative detrending kernel.

Computes exact median + MAD, then refines with asymmetric bisquare
iterations that penalize downward deviations (transit dips) more
aggressively than upward ones.

Two execution paths (identical algorithm, identical results):
  - Fused HIP/CUDA kernel: quickselect + iterations in one GPU call
  - Fallback: torch.sort + PyTorch iterations (no compilation needed)
"""

from __future__ import annotations

import torch

from torchflat._utils import MIN_SEGMENT_LENGTH, masked_median


def umi_detrend(
    flux: torch.Tensor,
    time: torch.Tensor,
    valid_mask: torch.Tensor,
    segment_id: torch.Tensor,
    window_length_days: float = 0.5,
    n_iter: int = 5,
    cval: float = 5.0,
    min_segment_points: int = MIN_SEGMENT_LENGTH,
    dtype: torch.dtype = torch.float32,
    asymmetry: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Segment-aware UMI (Unified Median Iterative) detrending.

    Computes exact median and MAD via quickselect (GPU kernel) or
    torch.sort (fallback), then refines with asymmetric bisquare
    iterations that penalize downward deviations (transit dips) more
    aggressively than upward ones.

    Both paths produce identical results.

    Args:
        flux: ``[B, L]`` flux values.
        time: ``[B, L]`` timestamps.
        valid_mask: ``[B, L]`` boolean mask (True = valid point).
        segment_id: ``[B, L]`` int32 segment labels.
        window_length_days: Sliding window width in days (default 0.5 = 12 h).
        n_iter: Number of asymmetric bisquare iterations.
        cval: Rejection threshold in MAD units.
        min_segment_points: Minimum valid points in a window for a valid trend.
        dtype: Computation dtype (float32 or float64).
        asymmetry: Dip penalty factor.  1.0 = standard biweight.
            Values > 1 penalize downward deviations more, preserving
            transit depth.  Default 1.5 (validated on 500-star
            train/test split).

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
    dt = time[:, 1:] - time[:, :-1]
    dt_valid = valid_mask[:, 1:] & valid_mask[:, :-1]
    median_cadence = masked_median(dt, dt_valid)

    win_samples = (window_length_days / median_cadence.clamp(min=1e-10)).round().long()
    W = int(win_samples.median().item())
    W = max(W, 3)
    W = W | 1  # ensure odd

    if L < W:
        nan_tensor = torch.full((B, L), float("nan"), dtype=flux.dtype, device=device)
        return nan_tensor, nan_tensor.clone()

    N_pos = L - W + 1

    # ------------------------------------------------------------------
    # Step 1: Unfold into sliding windows
    # ------------------------------------------------------------------
    flux_compute = flux.to(dtype)
    flux_windows = flux_compute.unfold(dimension=1, size=W, step=1).contiguous()
    valid_windows = valid_mask.unfold(dimension=1, size=W, step=1).contiguous()
    seg_windows = segment_id.unfold(dimension=1, size=W, step=1).contiguous()

    # ------------------------------------------------------------------
    # Step 2: Segment-aware window validity mask
    # ------------------------------------------------------------------
    center_seg = seg_windows[:, :, W // 2 : W // 2 + 1]
    window_valid = valid_windows & (seg_windows == center_seg)

    valid_f = window_valid.to(dtype)
    n_valid = valid_f.sum(dim=-1).clamp(min=1)

    # ------------------------------------------------------------------
    # Step 3: Compute median + MAD + asymmetric bisquare iterations
    # ------------------------------------------------------------------
    _use_kernel = False
    if flux_windows.is_cuda and W <= 512:
        from torchflat._kernel_loader import _get_umi_kernel
        _umi_kern = _get_umi_kernel()
        if _umi_kern is not None:
            _use_kernel = True

    if _use_kernel:
        # Fused kernel: median -> MAD -> asymmetric bisquare iterations,
        # all in one GPU call per thread, zero global memory traffic.
        location = _umi_kern.umi_detrend(
            flux_windows, window_valid,
            float(cval), float(asymmetry), int(n_iter), int(min_segment_points),
        )
    else:
        # Fallback: torch.sort for exact median + MAD, then PyTorch
        # asymmetric bisquare iterations.  Produces identical results
        # to the fused kernel (same algorithm, different execution).
        working = flux_windows.clone()
        working[~window_valid] = float("inf")
        n_valid_long = n_valid.long()

        sorted_vals = torch.sort(working, dim=-1).values

        # Exact median (numpy even-length convention)
        mid_lo = ((n_valid_long - 1) // 2).clamp(min=0)
        mid_hi = (n_valid_long // 2).clamp(min=0)
        val_lo = sorted_vals.gather(-1, mid_lo.unsqueeze(-1)).squeeze(-1)
        val_hi = sorted_vals.gather(-1, mid_hi.unsqueeze(-1)).squeeze(-1)
        location = (val_lo + val_hi) / 2.0

        # Exact MAD (median absolute deviation)
        abs_dev = (flux_windows - location.unsqueeze(-1)).abs()
        abs_dev[~window_valid] = float("inf")
        sorted_dev = torch.sort(abs_dev, dim=-1).values
        mad_lo = sorted_dev.gather(-1, mid_lo.unsqueeze(-1)).squeeze(-1)
        mad_hi = sorted_dev.gather(-1, mid_hi.unsqueeze(-1)).squeeze(-1)
        mad = (mad_lo + mad_hi) / 2.0

        safe_scale = (cval * mad.clamp(min=1e-10)).unsqueeze(-1)

        del sorted_vals, sorted_dev, working, abs_dev

        # Asymmetric bisquare iterations (same as kernel)
        for _ in range(n_iter):
            u = (flux_windows - location.unsqueeze(-1)) / safe_scale
            u_eff = torch.where(u < 0, u * asymmetry, u)
            u_abs = u_eff.abs()
            weights = ((1.0 - u_abs ** 2) ** 2) * (u_abs < 1.0).to(dtype) * valid_f
            w_sum = weights.sum(dim=-1).clamp(min=1e-10)
            location = (flux_windows * weights).sum(dim=-1) / w_sum

    # ------------------------------------------------------------------
    # Step 4: Map location back to full-length trend array
    # ------------------------------------------------------------------
    trend = torch.full((B, L), float("nan"), dtype=dtype, device=device)
    offset = W // 2
    trend[:, offset : offset + N_pos] = location

    n_valid_per_window = window_valid.sum(dim=-1)
    insufficient = n_valid_per_window < min_segment_points
    trend[:, offset : offset + N_pos] = torch.where(
        insufficient,
        torch.tensor(float("nan"), dtype=dtype, device=device),
        trend[:, offset : offset + N_pos],
    )

    # ------------------------------------------------------------------
    # Step 5: Detrend
    # ------------------------------------------------------------------
    trend_out = trend.to(flux.dtype)
    detrended = torch.where(
        (trend_out > 0) & valid_mask & torch.isfinite(trend_out),
        flux / trend_out,
        torch.tensor(float("nan"), dtype=flux.dtype, device=device),
    )

    return detrended, trend_out
