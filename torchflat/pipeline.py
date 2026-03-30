"""Track A + Track B pipeline orchestration."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from torchflat.batching import (
    assemble_batch,
    bucket_stars,
    compute_max_batch,
    cpu_prescan,
)
from torchflat.umi import umi_detrend
from torchflat.clipping import conservative_clip, rolling_clip
from torchflat.gaps import detect_gaps, interpolate_small_gaps
from torchflat.highpass import fft_highpass
from torchflat.normalize import normalize_track_a, normalize_track_b
from torchflat.quality import quality_filter
from torchflat.windows import DEFAULT_WINDOW_SCALES, extract_windows

logger = logging.getLogger("torchflat")


# ---------------------------------------------------------------------------
# Track A: PDCSAP -> Transit Windows
# ---------------------------------------------------------------------------

def preprocess_track_a(
    times: list[np.ndarray],
    fluxes: list[np.ndarray],
    qualities: list[np.ndarray],
    device: torch.device | str = "cuda",
    vram_budget_gb: float | None = None,
    max_batch: int | None = None,
    window_scales: list[tuple[int, int]] = DEFAULT_WINDOW_SCALES,
    window_length_days: float = 0.5,
    clip_sigma: float = 5.0,
    biweight_iter: int = 10,
    dtype: torch.dtype = torch.float32,
    asymmetry: float = 1.5,
    progress_callback: callable | None = None,
) -> tuple[list[dict | None], list[dict]]:
    """Full Track A preprocessing on GPU.

    Returns:
        results: Per-star dict (None for skipped stars) with window arrays
            and trend/detrended flux.
        skipped_stars: List of ``{index, reason, details}`` dicts.
    """
    if isinstance(device, str):
        device = torch.device(device)

    n_stars = len(times)
    if n_stars == 0:
        return [], []

    # CPU pre-scan
    prescan = cpu_prescan(times, fluxes, qualities)

    # Collect skipped stars
    skipped_stars: list[dict] = []
    for info in prescan:
        if info["degenerate"]:
            skipped_stars.append({
                "index": info["index"],
                "reason": info["degenerate_reason"],
                "details": {"n_valid": info["n_valid"]},
            })
            logger.warning(
                "Star %d skipped: %s (n_valid=%d)",
                info["index"], info["degenerate_reason"], info["n_valid"],
            )

    # Bucket non-degenerate stars
    buckets = bucket_stars(prescan)
    n_processable = sum(len(b["star_indices"]) for b in buckets)
    logger.info(
        "Processing %d stars across %d buckets (%d skipped)",
        n_processable, len(buckets), len(skipped_stars),
    )

    # Results array (None for skipped)
    results: list[dict | None] = [None] * n_stars

    for bucket_idx, bucket in enumerate(buckets):
        pad_length = bucket["pad_length"]
        star_indices = bucket["star_indices"]
        mb = compute_max_batch(
            pad_length, device=device,
            vram_budget_gb=vram_budget_gb, max_batch_override=max_batch,
        )
        logger.info(
            "Bucket %d: %d stars, pad_length=%d, max_batch=%d",
            bucket_idx, len(star_indices), pad_length, mb,
        )

        # Process in sub-batches
        for batch_start in range(0, len(star_indices), mb):
            batch_indices = star_indices[batch_start : batch_start + mb]
            batch = assemble_batch(
                batch_indices, times, fluxes, qualities, pad_length, device,
            )

            flux = batch["flux"]
            time_t = batch["time"]
            quality_t = batch["quality"]
            valid_mask = batch["valid_mask"]

            # Pipeline steps (gap interpolation done in assemble_batch on CPU)
            valid_mask = quality_filter(flux, time_t, quality_t) & valid_mask
            segment_id, median_cadence = detect_gaps(time_t, valid_mask)
            valid_mask = rolling_clip(flux, valid_mask, segment_id, sigma=clip_sigma)
            detrended, trend = umi_detrend(
                flux, time_t, valid_mask, segment_id,
                window_length_days=window_length_days,
                n_iter=biweight_iter, dtype=dtype, asymmetry=asymmetry,
            )
            normalized = normalize_track_a(detrended, valid_mask)
            windows_dict = extract_windows(
                normalized, valid_mask, segment_id, time_t, window_scales,
            )

            # Collect per-star results
            lengths = batch["lengths"]
            for j, orig_idx in enumerate(batch_indices):
                L = lengths[j].item()
                star_result: dict = {
                    "trend": trend[j, :L].cpu().numpy(),
                    "detrended": detrended[j, :L].cpu().numpy(),
                }
                for win_size, _stride in window_scales:
                    if win_size in windows_dict:
                        wd = windows_dict[win_size]
                        star_mask = wd["star_indices"] == j
                        star_result[f"windows_{win_size}"] = wd["windows"][star_mask].numpy()
                        star_result[f"window_times_{win_size}"] = wd["window_times"][star_mask].numpy()
                results[orig_idx] = star_result

            if progress_callback is not None:
                done = sum(1 for r in results if r is not None)
                progress_callback(done, n_processable)

    return results, skipped_stars


# ---------------------------------------------------------------------------
# Track B: SAP -> Anomaly Curve
# ---------------------------------------------------------------------------

def preprocess_track_b(
    times: list[np.ndarray],
    sap_fluxes: list[np.ndarray],
    qualities: list[np.ndarray],
    device: torch.device | str = "cuda",
    vram_budget_gb: float | None = None,
    max_batch: int | None = None,
    cutoff_days: float = 5.0,
    clip_sigma: float = 10.0,
    output_length: int = 20000,
) -> tuple[list[tuple[np.ndarray, np.ndarray] | None], list[dict]]:
    """Track B preprocessing: SAP flux -> anomaly detection curve.

    Returns:
        results: Per-star ``(curve, mask)`` tuple of np.ndarray ``[output_length]``.
            None for skipped stars.
        skipped_stars: List of ``{index, reason, details}`` dicts.
    """
    if isinstance(device, str):
        device = torch.device(device)

    n_stars = len(times)
    if n_stars == 0:
        return [], []

    prescan = cpu_prescan(times, sap_fluxes, qualities)

    skipped_stars: list[dict] = []
    for info in prescan:
        if info["degenerate"]:
            skipped_stars.append({
                "index": info["index"],
                "reason": info["degenerate_reason"],
                "details": {"n_valid": info["n_valid"]},
            })

    buckets = bucket_stars(prescan)
    results: list[tuple[np.ndarray, np.ndarray] | None] = [None] * n_stars

    for bucket in buckets:
        pad_length = bucket["pad_length"]
        star_indices = bucket["star_indices"]
        mb = compute_max_batch(
            pad_length, device=device,
            vram_budget_gb=vram_budget_gb, max_batch_override=max_batch,
        )

        for batch_start in range(0, len(star_indices), mb):
            batch_indices = star_indices[batch_start : batch_start + mb]
            batch = assemble_batch(
                batch_indices, times, sap_fluxes, qualities, pad_length, device,
            )

            flux = batch["flux"]
            time_t = batch["time"]
            quality_t = batch["quality"]
            valid_mask = batch["valid_mask"]

            valid_mask = quality_filter(flux, time_t, quality_t) & valid_mask
            segment_id, median_cadence = detect_gaps(time_t, valid_mask)
            valid_mask = conservative_clip(flux, valid_mask, sigma=clip_sigma)
            filtered = fft_highpass(
                flux, valid_mask, segment_id, median_cadence,
                cutoff_days=cutoff_days,
            )
            normalized = normalize_track_b(filtered, valid_mask)

            # Pad to output_length and collect
            lengths = batch["lengths"]
            for j, orig_idx in enumerate(batch_indices):
                L = lengths[j].item()
                curve = np.zeros(output_length, dtype=np.float32)
                mask = np.zeros(output_length, dtype=np.float32)
                n = min(L, output_length)
                curve[:n] = normalized[j, :n].cpu().numpy()
                mask[:n] = valid_mask[j, :n].cpu().float().numpy()
                results[orig_idx] = (curve, mask)

    return results, skipped_stars


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def preprocess_sector(
    star_data: list[dict],
    output_dir: Path | str | None = None,
    device: torch.device | str = "cuda",
    vram_budget_gb: float | None = None,
    max_batch: int | None = None,
    window_scales: list[tuple[int, int]] = DEFAULT_WINDOW_SCALES,
    skip_track_b: bool = False,
    progress_callback: callable | None = None,
    **kwargs,
) -> tuple[list[dict], list[dict]]:
    """Full sector preprocessing: runs Track A + Track B.

    Args:
        star_data: List of dicts with keys ``time``, ``sap_flux``,
            ``pdcsap_flux``, ``quality``.
        output_dir: If provided, save ``.npy`` files per star.
        device: Torch device string or object.
        vram_budget_gb: VRAM budget override.
        max_batch: Direct batch-size override.
        window_scales: Window extraction scales for Track A.
        skip_track_b: If True, skip Track B (FFT highpass anomaly detection).
            Saves ~60% of pipeline time when only transit search is needed.
        **kwargs: Forwarded to :func:`preprocess_track_a`.

    Returns:
        ``(results, skipped_stars)`` where *results* is a list of per-star
        dicts combining Track A and Track B outputs.
    """
    times = [s["time"] for s in star_data]
    pdcsap = [s["pdcsap_flux"] for s in star_data]
    sap = [s["sap_flux"] for s in star_data]
    qualities = [s["quality"] for s in star_data]

    track_a_results, skipped_a = preprocess_track_a(
        times, pdcsap, qualities,
        device=device, vram_budget_gb=vram_budget_gb, max_batch=max_batch,
        window_scales=window_scales, progress_callback=progress_callback,
        **kwargs,
    )

    if skip_track_b:
        track_b_results = [None] * len(star_data)
        skipped_b: list[dict] = []
    else:
        track_b_results, skipped_b = preprocess_track_b(
            times, sap, qualities,
            device=device, vram_budget_gb=vram_budget_gb, max_batch=max_batch,
        )

    # Merge results
    combined: list[dict] = []
    for i in range(len(star_data)):
        entry: dict = {}
        if track_a_results[i] is not None:
            entry.update(track_a_results[i])
        if track_b_results[i] is not None:
            curve, mask = track_b_results[i]
            entry["track_b_curve"] = curve
            entry["track_b_mask"] = mask
        combined.append(entry if entry else {})

    # Merge skipped lists (deduplicate by index)
    skipped_indices = set()
    all_skipped: list[dict] = []
    for s in skipped_a + skipped_b:
        if s["index"] not in skipped_indices:
            skipped_indices.add(s["index"])
            all_skipped.append(s)

    # Save to disk if requested
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for i, entry in enumerate(combined):
            if entry:
                for key, val in entry.items():
                    if isinstance(val, np.ndarray):
                        np.save(out / f"star_{i:05d}_{key}.npy", val)

    logger.info(
        "Sector complete: %d processed, %d skipped",
        sum(1 for e in combined if e), len(all_skipped),
    )

    return combined, all_skipped


