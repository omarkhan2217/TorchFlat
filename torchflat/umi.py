"""UMI - Unified Median Iterative detrending kernel.

Computes exact median + MAD, then refines with asymmetric bisquare
iterations that penalize downward deviations (transit dips) more
aggressively than upward ones.

Two execution paths (identical algorithm, identical results):
  - Direct HIP/CUDA kernel: reads raw [B,L] arrays, no unfold needed
  - Fallback: torch.sort + PyTorch iterations (no compilation needed)
"""

from __future__ import annotations

import torch

from torchflat._utils import MIN_SEGMENT_LENGTH, masked_median

# Empirical bias lookup (ppm) by asymmetry value.
# Measured on 10,000 real TESS stars with upper-RMS scale.
# Bias is the median shift on flat (no-transit) stars.
UMI_BIAS_PPM = {
    1.0: -2,
    1.5: -209,
    2.0: -451,
    2.5: -687,
    3.0: -896,
}


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
    asymmetry: float = 2.0,
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
        asymmetry: Dip penalty factor.  2.0 = best transit accuracy
            (default), 1.5 = balanced for mixed surveys, 1.0 = zero
            bias (for variable stars).  Validated on 10,000 stars.

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

    # Auto-scale min_segment_points to window size.
    # Default MIN_SEGMENT_LENGTH=50 assumes TESS 2-min cadence (W~361).
    # Kepler long-cadence (W~25) needs a proportionally smaller threshold.
    if min_segment_points > W // 2:
        min_segment_points = max(W // 3, 5)

    if L < W:
        nan_tensor = torch.full((B, L), float("nan"), dtype=flux.dtype, device=device)
        return nan_tensor, nan_tensor.clone()

    N_pos = L - W + 1

    # ------------------------------------------------------------------
    # Step 1: Check for GPU kernel
    # ------------------------------------------------------------------
    _use_kernel = False
    if flux.is_cuda and W <= 512:
        from torchflat._kernel_loader import _get_umi_kernel
        _umi_kern = _get_umi_kernel()
        if _umi_kern is not None and hasattr(_umi_kern, "umi_detrend_direct"):
            _use_kernel = True

    if not _use_kernel and flux.is_cuda and not getattr(umi_detrend, "_warned_fallback", False):
        import logging
        logging.getLogger("torchflat").warning(
            "UMI kernel not available, using torch.sort fallback (slower). "
            "Install ROCm/CUDA toolkit to enable the fused kernel."
        )
        umi_detrend._warned_fallback = True

    if _use_kernel:
        # Direct kernel: reads raw [B, L] arrays, handles windowing and
        # segment masking internally. No unfold, no tensor copies.
        # One thread per (star, position), quickselect + biweight in registers.
        flux_compute = flux.to(dtype)
        location = _umi_kern.umi_detrend_direct(
            flux_compute, valid_mask, segment_id, int(W),
            float(cval), float(asymmetry), int(n_iter), int(min_segment_points),
        )
    else:
        # Fallback: unfold + torch.sort + PyTorch bisquare iterations.
        flux_compute = flux.to(dtype)
        flux_windows = flux_compute.unfold(dimension=1, size=W, step=1).contiguous()
        valid_windows = valid_mask.unfold(dimension=1, size=W, step=1).contiguous()
        seg_windows = segment_id.unfold(dimension=1, size=W, step=1).contiguous()

        center_seg = seg_windows[:, :, W // 2 : W // 2 + 1]
        window_valid = valid_windows & (seg_windows == center_seg)
        valid_f = window_valid.to(dtype)
        n_valid = valid_f.sum(dim=-1).clamp(min=1)

        working = flux_windows.clone()
        working[~window_valid] = float("inf")
        n_valid_long = n_valid.long()

        sorted_vals = torch.sort(working, dim=-1).values

        mid_lo = ((n_valid_long - 1) // 2).clamp(min=0)
        mid_hi = (n_valid_long // 2).clamp(min=0)
        val_lo = sorted_vals.gather(-1, mid_lo.unsqueeze(-1)).squeeze(-1)
        val_hi = sorted_vals.gather(-1, mid_hi.unsqueeze(-1)).squeeze(-1)
        location = (val_lo + val_hi) / 2.0

        # Upper-RMS scale: RMS of points above median only.
        # Transit dips (below median) never contaminate the scale estimate.
        residuals = flux_windows - location.unsqueeze(-1)
        above_mask = (residuals > 0) & window_valid
        above_f = above_mask.to(dtype)
        n_above = above_f.sum(dim=-1).clamp(min=1)
        upper_sq = ((residuals ** 2) * above_f).sum(dim=-1)
        upper_rms = (upper_sq / n_above).sqrt()
        scale = upper_rms * 0.6745  # convert to MAD-equivalent

        safe_scale = (cval * scale.clamp(min=1e-10)).unsqueeze(-1)

        del sorted_vals, working

        for _ in range(n_iter):
            u = (flux_windows - location.unsqueeze(-1)) / safe_scale
            u_eff = torch.where(u < 0, u * asymmetry, u)
            u_abs = u_eff.abs()
            weights = ((1.0 - u_abs ** 2) ** 2) * (u_abs < 1.0).to(dtype) * valid_f
            w_sum = weights.sum(dim=-1).clamp(min=1e-10)
            location = (flux_windows * weights).sum(dim=-1) / w_sum

    # ------------------------------------------------------------------
    # Step 2: Map location back to full-length trend array
    # ------------------------------------------------------------------
    trend = torch.full((B, L), float("nan"), dtype=dtype, device=device)
    offset = W // 2
    trend[:, offset : offset + N_pos] = location

    # Guard: insufficient valid points (kernel handles this internally
    # for the direct path, but we still need it for the fallback)
    if not _use_kernel:
        n_valid_per_window = window_valid.sum(dim=-1)
        insufficient = n_valid_per_window < min_segment_points
        trend[:, offset : offset + N_pos] = torch.where(
            insufficient,
            torch.tensor(float("nan"), dtype=dtype, device=device),
            trend[:, offset : offset + N_pos],
        )

    # ------------------------------------------------------------------
    # Step 3: Detrend
    # ------------------------------------------------------------------
    trend_out = trend.to(flux.dtype)
    detrended = torch.where(
        (trend_out > 0) & valid_mask & torch.isfinite(trend_out),
        flux / trend_out,
        torch.tensor(float("nan"), dtype=flux.dtype, device=device),
    )

    return detrended, trend_out
