"""CPU pre-scan, length-bucketed batch assembly, VRAM estimation."""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import torch

from torchflat._utils import GAP_THRESHOLD, MIN_POINTS, QUALITY_BITMASK

logger = logging.getLogger("torchflat")


# ---------------------------------------------------------------------------
# CPU pre-scan
# ---------------------------------------------------------------------------

def cpu_prescan(
    times: list[np.ndarray],
    fluxes: list[np.ndarray],
    qualities: list[np.ndarray],
    bitmask: int = QUALITY_BITMASK,
    gap_threshold: float = GAP_THRESHOLD,
    min_points: int = MIN_POINTS,
    window_samples: int = 360,
) -> list[dict]:
    """CPU pre-scan: compute post-filter length and flag degenerate stars.

    Runs a fast O(N) scan per star on CPU to determine quality counts, gap
    insertions, and whether the star is degenerate (too few points or
    segments too short for the biweight window).
    """
    results: list[dict] = []

    for i, (t, f, q) in enumerate(zip(times, fluxes, qualities)):
        valid = ((q & bitmask) == 0) & np.isfinite(f) & np.isfinite(t)
        n_valid = int(valid.sum())

        if n_valid < 2:
            results.append({
                "index": i,
                "n_valid": n_valid,
                "n_insertions": 0,
                "post_filter_length": n_valid,
                "max_segment_length": n_valid,
                "degenerate": True,
                "degenerate_reason": "too_few_valid_points",
            })
            continue

        t_valid = t[valid]
        dt = np.diff(t_valid)
        med_cadence = float(np.median(dt))
        if med_cadence <= 0:
            med_cadence = 1e-10

        gap_ratio = dt / med_cadence

        # Count small-gap interpolation insertions
        small_gaps = (gap_ratio > 1.5) & (gap_ratio < gap_threshold)
        n_insertions = 0
        for gr in gap_ratio[small_gaps]:
            n_insertions += int(round(gr)) - 1

        # Longest segment
        large_gap_pos = np.where(gap_ratio >= gap_threshold)[0]
        boundaries = np.concatenate([[-1], large_gap_pos, [len(dt)]])
        segment_lengths = np.diff(boundaries)
        max_segment = int(segment_lengths.max()) if len(segment_lengths) > 0 else n_valid

        post_filter_length = n_valid + n_insertions

        degenerate = False
        reason = None
        if n_valid < min_points:
            degenerate = True
            reason = "too_few_valid_points"
        elif max_segment < window_samples:
            degenerate = True
            reason = "segment_too_short"

        results.append({
            "index": i,
            "n_valid": n_valid,
            "n_insertions": n_insertions,
            "post_filter_length": post_filter_length,
            "max_segment_length": max_segment,
            "degenerate": degenerate,
            "degenerate_reason": reason,
        })

    return results


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------

def bucket_stars(
    prescan_results: list[dict],
    bucket_width: int = 1000,
) -> list[dict]:
    """Group non-degenerate stars into length buckets for batched GPU processing."""
    buckets_map: dict[int, list[int]] = defaultdict(list)

    for info in prescan_results:
        if info["degenerate"]:
            continue
        pfl = info["post_filter_length"]
        bucket_key = (pfl // bucket_width) * bucket_width + bucket_width
        buckets_map[bucket_key].append(info["index"])

    buckets: list[dict] = []
    for pad_length in sorted(buckets_map.keys()):
        buckets.append({
            "star_indices": buckets_map[pad_length],
            "pad_length": pad_length,
        })
    return buckets


# ---------------------------------------------------------------------------
# Batch assembly
# ---------------------------------------------------------------------------

def assemble_batch(
    star_indices: list[int],
    times: list[np.ndarray],
    fluxes: list[np.ndarray],
    qualities: list[np.ndarray],
    pad_length: int,
    device: torch.device,
) -> dict:
    """Pad, stack, and transfer a batch of stars to GPU.

    Returns dict with tensors ``time``, ``flux``, ``quality``, ``lengths``,
    and ``valid_mask`` on *device*.
    """
    B = len(star_indices)

    time_batch = torch.zeros(B, pad_length, dtype=torch.float64)
    flux_batch = torch.zeros(B, pad_length, dtype=torch.float32)
    quality_batch = torch.zeros(B, pad_length, dtype=torch.int32)
    lengths = torch.zeros(B, dtype=torch.long)
    valid_mask = torch.zeros(B, pad_length, dtype=torch.bool)

    for j, idx in enumerate(star_indices):
        L = len(times[idx])
        n = min(L, pad_length)
        time_batch[j, :n] = torch.from_numpy(times[idx][:n].astype(np.float64))
        flux_batch[j, :n] = torch.from_numpy(fluxes[idx][:n].astype(np.float32))
        quality_batch[j, :n] = torch.from_numpy(qualities[idx][:n].astype(np.int32))
        lengths[j] = n
        valid_mask[j, :n] = True

    return {
        "time": time_batch.to(device),
        "flux": flux_batch.to(device),
        "quality": quality_batch.to(device),
        "lengths": lengths.to(device),
        "valid_mask": valid_mask.to(device),
    }


# ---------------------------------------------------------------------------
# VRAM estimation
# ---------------------------------------------------------------------------

def estimate_peak_vram(L: int, win: int, dtype_bytes: int = 4) -> int:
    """Estimate peak VRAM per star (bytes) during biweight detrending.

    Accounts for persistent window tensors, per-iteration intermediates,
    and temporary topk/sort int64 indices.
    """
    N_pos = L - win + 1
    window_bytes = N_pos * win * dtype_bytes
    bool_window_bytes = N_pos * win * 1
    indices_bytes = N_pos * win * 8
    seg_window_bytes = N_pos * win * 4  # int32 segment IDs

    base = L * 17  # flux(4) + time(8) + valid(1) + seg_id(4)
    persistent = window_bytes + seg_window_bytes + bool_window_bytes
    per_iter = 4 * window_bytes  # topk_clone + abs_dev + u + weights
    peak_temp = indices_bytes

    return base + persistent + per_iter + peak_temp


def compute_max_batch(
    pad_length: int,
    win: int = 360,
    device: torch.device | None = None,
    vram_budget_gb: float | None = None,
    max_batch_override: int | None = None,
    safety_factor: float = 0.8,
) -> int:
    """Dynamic max_batch with 3-tier priority: override > budget > auto-detect."""
    if max_batch_override is not None:
        return max(1, max_batch_override)

    if vram_budget_gb is not None:
        available = int(vram_budget_gb * 1024**3)
    elif device is not None and device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        total = props.total_memory
        headroom = 4 * 1024**3
        available = total - headroom
    else:
        return 1  # CPU fallback

    peak_per_star = estimate_peak_vram(pad_length, win)
    if peak_per_star <= 0:
        return 1
    max_batch = int(available * safety_factor / peak_per_star)
    return max(1, min(max_batch, 90))
