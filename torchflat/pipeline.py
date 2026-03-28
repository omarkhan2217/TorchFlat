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
from torchflat.biweight import biweight_detrend
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

            # Pipeline steps
            valid_mask = quality_filter(flux, time_t, quality_t) & valid_mask
            flux, valid_mask = interpolate_small_gaps(flux, time_t, valid_mask)
            segment_id, median_cadence = detect_gaps(time_t, valid_mask)
            valid_mask = rolling_clip(flux, valid_mask, segment_id, sigma=clip_sigma)
            detrended, trend = biweight_detrend(
                flux, time_t, valid_mask, segment_id,
                window_length_days=window_length_days,
                n_iter=biweight_iter, dtype=dtype,
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
            flux, valid_mask = interpolate_small_gaps(flux, time_t, valid_mask)
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
        window_scales=window_scales, **kwargs,
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


# ---------------------------------------------------------------------------
# Hybrid CPU+GPU processing
# ---------------------------------------------------------------------------

def _process_star_cpu(args: tuple) -> tuple[int, dict | None, dict | None]:
    """Process a single star on CPU using the Numba biweight kernel.

    Designed to run in a ProcessPoolExecutor worker.
    """
    idx, star, window_scales, skip_track_b = args

    from torchflat._utils import QUALITY_BITMASK, MIN_POINTS
    from torchflat.biweight_cpu import biweight_detrend_cpu_single

    t = star["time"].astype(np.float64)
    f_pdcsap = star["pdcsap_flux"].astype(np.float64)
    q = star["quality"].astype(np.int32)

    # Quality filter
    valid = ((q & QUALITY_BITMASK) == 0) & np.isfinite(f_pdcsap) & np.isfinite(t)
    if valid.sum() < MIN_POINTS:
        return idx, None, {"index": idx, "reason": "too_few_valid_points",
                           "details": {"n_valid": int(valid.sum())}}

    # Gap detection (simple CPU version)
    dt = np.diff(t)
    if len(dt[valid[1:] & valid[:-1]]) > 0:
        med_cad = np.median(dt[valid[1:] & valid[:-1]])
    else:
        return idx, None, {"index": idx, "reason": "too_few_valid_points",
                           "details": {"n_valid": int(valid.sum())}}
    if med_cad <= 0:
        med_cad = 1e-10
    gap_ratio = np.zeros(len(dt))
    gap_ratio[valid[1:] & valid[:-1]] = dt[valid[1:] & valid[:-1]] / med_cad
    is_large_gap = gap_ratio >= 5.0
    seg_id = np.concatenate([[0], np.cumsum(is_large_gap)]).astype(np.int32)

    # Small gap interpolation
    f_interp = f_pdcsap.copy()
    v_interp = valid.copy()
    i = 0
    while i < len(f_interp):
        if not v_interp[i]:
            gs = i
            while i < len(f_interp) and not v_interp[i]:
                i += 1
            ge = i
            gap_sz = ge - gs
            if gs > 0 and ge < len(f_interp) and gap_sz <= 4:
                for k in range(gap_sz):
                    frac = (k + 1) / (gap_sz + 1)
                    f_interp[gs + k] = f_interp[gs - 1] + frac * (f_interp[ge] - f_interp[gs - 1])
                    v_interp[gs + k] = True
        else:
            i += 1

    # Rolling median sigma clipping (matches GPU rolling_clip)
    L = len(f_interp)
    win_clip = 25
    if v_interp.sum() > win_clip:
        from scipy.ndimage import median_filter
        rolling_med = median_filter(f_interp, size=win_clip, mode='nearest')

        residuals = np.abs(f_interp - rolling_med)
        mad = np.median(residuals[v_interp])
        if mad > 1e-10:
            threshold = 5.0 * mad / 0.6745
            v_interp &= residuals <= threshold

    # Biweight detrend (Numba kernel)
    detrended, trend = biweight_detrend_cpu_single(
        f_interp, t, v_interp, seg_id,
    )

    result = {
        "trend": trend.astype(np.float32),
        "detrended": detrended.astype(np.float32),
    }

    return idx, result, None


def preprocess_sector_hybrid(
    star_data: list[dict],
    output_dir: Path | str | None = None,
    gpu_device: torch.device | str = "cuda",
    vram_budget_gb: float | None = None,
    max_batch: int | None = None,
    window_scales: list[tuple[int, int]] = DEFAULT_WINDOW_SCALES,
    skip_track_b: bool = False,
    cpu_fraction: float = 0.65,
    cpu_workers: int = 12,
    **kwargs,
) -> tuple[list[dict], list[dict]]:
    """Hybrid CPU+GPU sector preprocessing.

    Splits stars between GPU and CPU, processing both simultaneously.
    The GPU runs the full TorchFlat pipeline. CPU workers run a fast
    Numba JIT biweight kernel (8+ stars/sec per thread) in parallel.

    Args:
        star_data: List of dicts with keys ``time``, ``sap_flux``,
            ``pdcsap_flux``, ``quality``.
        output_dir: If provided, save ``.npy`` files per star.
        gpu_device: GPU device string or object.
        vram_budget_gb: VRAM budget override.
        max_batch: Direct batch-size override.
        window_scales: Window extraction scales for Track A.
        skip_track_b: If True, skip Track B.
        cpu_fraction: Fraction of stars to process on CPU (default 0.25).
        cpu_workers: Number of CPU worker processes (default 12).
        **kwargs: Forwarded to :func:`preprocess_track_a`.

    Returns:
        ``(results, skipped_stars)``
    """
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    n_total = len(star_data)
    if n_total == 0:
        return [], []

    cpu_fraction = max(0.0, min(1.0, cpu_fraction))
    n_cpu = int(n_total * cpu_fraction)
    n_gpu = n_total - n_cpu

    if n_cpu == 0:
        return preprocess_sector(
            star_data, output_dir=output_dir, device=gpu_device,
            vram_budget_gb=vram_budget_gb, max_batch=max_batch,
            window_scales=window_scales, skip_track_b=skip_track_b, **kwargs,
        )

    gpu_data = star_data[:n_gpu]
    cpu_data = star_data[n_gpu:]

    logger.info(
        "Hybrid mode: %d stars on GPU, %d stars on CPU (%d workers)",
        n_gpu, n_cpu, cpu_workers,
    )

    # Prepare CPU work items
    cpu_args = [
        (n_gpu + i, cpu_data[i], window_scales, skip_track_b)
        for i in range(n_cpu)
    ]

    gpu_result = [None, None]

    def run_gpu():
        gpu_result[0], gpu_result[1] = preprocess_sector(
            gpu_data, device=gpu_device,
            vram_budget_gb=vram_budget_gb, max_batch=max_batch,
            window_scales=window_scales, skip_track_b=skip_track_b, **kwargs,
        )

    # Run GPU in a thread, CPU workers in processes, simultaneously
    combined = [None] * n_total
    all_skipped: list[dict] = []

    with ThreadPoolExecutor(max_workers=1) as gpu_pool:
        gpu_future = gpu_pool.submit(run_gpu)

        # CPU workers run in parallel with GPU
        with ProcessPoolExecutor(max_workers=cpu_workers) as cpu_pool:
            for idx, result, skip_info in cpu_pool.map(
                _process_star_cpu, cpu_args, chunksize=max(1, n_cpu // cpu_workers),
            ):
                combined[idx] = result
                if skip_info is not None:
                    all_skipped.append(skip_info)

        gpu_future.result()

    # Merge GPU results into combined
    for i, entry in enumerate(gpu_result[0]):
        combined[i] = entry
    all_skipped.extend(gpu_result[1])

    # Replace None with empty dict
    combined = [c if c is not None else {} for c in combined]

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
        "Hybrid sector complete: %d processed, %d skipped",
        sum(1 for e in combined if e), len(all_skipped),
    )

    return combined, all_skipped
