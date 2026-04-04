"""Tapered FFT brick-wall highpass filter."""

from __future__ import annotations

import math

import torch


def _tukey_window(n: int, alpha: float, device: torch.device, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Generate a Tukey (tapered cosine) window.

    Args:
        n: Window length.
        alpha: Taper fraction (0 = rectangular, 1 = Hann).
        device: Target device.
        dtype: Output dtype.
    """
    if n <= 1:
        return torch.ones(n, device=device, dtype=dtype)
    if alpha <= 0:
        return torch.ones(n, device=device, dtype=dtype)
    if alpha >= 1:
        idx = torch.arange(n, device=device, dtype=dtype)
        return 0.5 * (1.0 - torch.cos(2.0 * math.pi * idx / (n - 1)))

    window = torch.ones(n, device=device, dtype=dtype)
    taper_len = int(alpha * n / 2)
    if taper_len == 0:
        return window

    left = torch.arange(taper_len, device=device, dtype=dtype)
    window[:taper_len] = 0.5 * (1.0 - torch.cos(math.pi * left / taper_len))
    window[n - taper_len :] = window[:taper_len].flip(0)
    return window


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def fft_highpass(
    flux: torch.Tensor,
    valid_mask: torch.Tensor,
    segment_id: torch.Tensor,
    median_cadence_days: torch.Tensor,
    cutoff_days: float = 5.0,
    taper_alpha: float = 0.1,
) -> torch.Tensor:
    """Per-segment tapered FFT brick-wall highpass filter for Track B.

    Each segment is filtered independently to avoid cross-gap spectral
    leakage.  A Tukey window suppresses Gibbs ringing at segment edges.

    For single-segment stars (the common case with clean data), processing
    is batched across all such stars in one FFT call for efficiency.

    Args:
        flux: ``[B, L]`` flux values.
        valid_mask: ``[B, L]`` boolean mask.
        segment_id: ``[B, L]`` segment labels.
        median_cadence_days: ``[B]`` per-star median cadence in days.
        cutoff_days: Cutoff period.  Frequencies below ``1 / cutoff_days``
            are removed.
        taper_alpha: Tukey window alpha (fraction of segment to taper).

    Returns:
        ``[B, L]`` highpass-filtered flux.  Invalid positions are 0.
    """
    B, L = flux.shape
    device = flux.device
    output = torch.zeros_like(flux)

    # Determine which stars have a single segment (common case)
    max_seg_per_star = segment_id.max(dim=1).values  # [B]
    single_seg = max_seg_per_star == 0  # [B] bool
    multi_seg = ~single_seg

    # --- Fast path: batch all single-segment stars ---
    single_indices = single_seg.nonzero(as_tuple=False).squeeze(-1)
    if single_indices.numel() > 0:
        cadence = median_cadence_days[single_indices[0]].item()
        if cadence > 0:
            # Stay in float32 — Track B anomaly detection doesn't need float64
            batch_flux = flux[single_indices]  # [N, L] float32
            batch_valid = valid_mask[single_indices]
            batch_flux = batch_flux * batch_valid.float()

            # Taper (float32)
            taper = _tukey_window(L, taper_alpha, device, dtype=torch.float32)
            tapered = batch_flux * taper.unsqueeze(0)

            # Batched FFT — pad to power-of-2 for rocFFT efficiency
            n_fft = _next_power_of_2(L)
            F_spec = torch.fft.rfft(tapered, n=n_fft, dim=1)
            freqs = torch.fft.rfftfreq(n_fft, d=cadence).to(device)
            cutoff_freq = 1.0 / cutoff_days
            freq_mask = (freqs >= cutoff_freq).unsqueeze(0)
            F_filtered = F_spec * freq_mask
            filtered = torch.fft.irfft(F_filtered, n=n_fft, dim=1)[:, :L]

            output[single_indices] = filtered

    # --- Slow path: per-star per-segment for multi-segment stars ---
    multi_indices = multi_seg.nonzero(as_tuple=False).squeeze(-1)
    for idx in multi_indices:
        b = idx.item()
        cadence = median_cadence_days[b].item()
        if cadence <= 0:
            continue

        seg_ids_star = segment_id[b]
        unique_segs = seg_ids_star.unique()

        for s in unique_segs:
            seg_mask = (seg_ids_star == s) & valid_mask[b]
            indices = seg_mask.nonzero(as_tuple=False).squeeze(-1)

            if indices.numel() < 2:
                continue

            seg_start = indices[0].item()
            seg_end = indices[-1].item() + 1
            seg_len = seg_end - seg_start

            seg_flux = flux[b, seg_start:seg_end]  # float32
            seg_valid = valid_mask[b, seg_start:seg_end]
            seg_flux = seg_flux * seg_valid.float()

            taper = _tukey_window(seg_len, taper_alpha, device, dtype=torch.float32)
            tapered = seg_flux * taper

            F_spec = torch.fft.rfft(tapered)
            freqs = torch.fft.rfftfreq(seg_len, d=cadence).to(device)
            cutoff_freq = 1.0 / cutoff_days
            freq_mask = (freqs >= cutoff_freq).unsqueeze(0)
            F_filtered = F_spec * freq_mask
            filtered = torch.fft.irfft(F_filtered, n=seg_len).squeeze(0)

            output[b, seg_start:seg_end] = filtered

    return output
