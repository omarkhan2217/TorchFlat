#!/usr/bin/env python
"""Profile TorchFlat pipeline to find bottlenecks.

Logs detailed timing for every pipeline step and the biweight kernel internals.
Results are saved to logs/ with timestamps.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

CADENCE = 2.0 / 1440.0
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def _sync():
    """Synchronize GPU for accurate timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _make_stars(n_stars: int, n_points: int = 18000, seed: int = 42):
    rng = np.random.default_rng(seed)
    times, fluxes, qualities = [], [], []
    for _ in range(n_stars):
        t = np.arange(n_points, dtype=np.float64) * CADENCE
        f = (1.0 + rng.normal(0, 0.001, n_points)).astype(np.float32)
        q = np.zeros(n_points, dtype=np.int32)
        times.append(t)
        fluxes.append(f)
        qualities.append(q)
    return times, fluxes, qualities


def profile_pipeline_steps(n_stars: int = 20, n_points: int = 18000):
    """Profile each pipeline step individually."""
    from torchflat.batching import assemble_batch, bucket_stars, cpu_prescan
    from torchflat.umi import umi_detrend as biweight_detrend
    from torchflat.clipping import rolling_clip
    from torchflat.gaps import detect_gaps, interpolate_small_gaps
    from torchflat.highpass import fft_highpass
    from torchflat.normalize import normalize_track_a, normalize_track_b
    from torchflat.quality import quality_filter
    from torchflat.windows import extract_windows

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    times, fluxes, qualities = _make_stars(n_stars, n_points)

    results = {
        "config": {
            "n_stars": n_stars,
            "n_points": n_points,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "torch_version": torch.__version__,
        },
        "steps": {},
    }

    # CPU prescan
    t0 = time.perf_counter()
    prescan = cpu_prescan(times, fluxes, qualities)
    results["steps"]["cpu_prescan"] = time.perf_counter() - t0

    # Bucketing
    t0 = time.perf_counter()
    buckets = bucket_stars(prescan)
    results["steps"]["bucket_stars"] = time.perf_counter() - t0

    # Assemble batch (use first bucket)
    bucket = buckets[0]
    t0 = time.perf_counter()
    batch = assemble_batch(
        bucket["star_indices"], times, fluxes, qualities,
        bucket["pad_length"], device,
    )
    _sync()
    results["steps"]["assemble_batch"] = time.perf_counter() - t0

    flux = batch["flux"]
    time_t = batch["time"]
    quality_t = batch["quality"]
    valid_mask = batch["valid_mask"]
    B = flux.shape[0]

    # Quality filter
    _sync()
    t0 = time.perf_counter()
    valid_mask = quality_filter(flux, time_t, quality_t) & valid_mask
    _sync()
    results["steps"]["quality_filter"] = time.perf_counter() - t0

    # Gap interpolation
    _sync()
    t0 = time.perf_counter()
    flux, valid_mask = interpolate_small_gaps(flux, time_t, valid_mask)
    _sync()
    results["steps"]["interpolate_small_gaps"] = time.perf_counter() - t0

    # Gap detection
    _sync()
    t0 = time.perf_counter()
    segment_id, median_cadence = detect_gaps(time_t, valid_mask)
    _sync()
    results["steps"]["detect_gaps"] = time.perf_counter() - t0

    # Rolling clip
    _sync()
    t0 = time.perf_counter()
    valid_mask = rolling_clip(flux, valid_mask, segment_id)
    _sync()
    results["steps"]["rolling_clip"] = time.perf_counter() - t0

    # Biweight detrend (THE main bottleneck)
    _sync()
    t0 = time.perf_counter()
    detrended, trend = biweight_detrend(flux, time_t, valid_mask, segment_id)
    _sync()
    results["steps"]["biweight_detrend"] = time.perf_counter() - t0

    # Normalize Track A
    _sync()
    t0 = time.perf_counter()
    normalized = normalize_track_a(detrended, valid_mask)
    _sync()
    results["steps"]["normalize_track_a"] = time.perf_counter() - t0

    # Window extraction
    _sync()
    t0 = time.perf_counter()
    windows = extract_windows(normalized, valid_mask, segment_id, time_t,
                              window_scales=[(256, 128)])
    _sync()
    results["steps"]["extract_windows"] = time.perf_counter() - t0

    # Track B: highpass
    _sync()
    t0 = time.perf_counter()
    filtered = fft_highpass(flux, valid_mask, segment_id, median_cadence)
    _sync()
    results["steps"]["fft_highpass"] = time.perf_counter() - t0

    # Track B: normalize
    _sync()
    t0 = time.perf_counter()
    norm_b = normalize_track_b(filtered, valid_mask)
    _sync()
    results["steps"]["normalize_track_b"] = time.perf_counter() - t0

    # Total
    total = sum(results["steps"].values())
    results["total_time"] = total
    results["stars_per_sec"] = n_stars / total
    results["ms_per_star"] = total / n_stars * 1000

    return results


def profile_biweight_internals(n_stars: int = 20, n_points: int = 18000):
    """Profile inside the biweight kernel: unfold, median, iterations."""
    from torchflat._utils import masked_median

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(42)

    # Prepare tensors as the pipeline would
    time_t = (torch.arange(n_points, dtype=torch.float64) * CADENCE).unsqueeze(0).expand(n_stars, -1).contiguous().to(device)
    flux = torch.tensor(
        1.0 + rng.normal(0, 0.001, (n_stars, n_points)),
        dtype=torch.float32, device=device,
    )
    valid_mask = torch.ones(n_stars, n_points, dtype=torch.bool, device=device)
    segment_id = torch.zeros(n_stars, n_points, dtype=torch.int32, device=device)

    results = {
        "config": {
            "n_stars": n_stars,
            "n_points": n_points,
            "device": str(device),
        },
        "steps": {},
    }

    # Step 0: Cadence + window size
    _sync()
    t0 = time.perf_counter()
    dt = time_t[:, 1:] - time_t[:, :-1]
    dt_valid = valid_mask[:, 1:] & valid_mask[:, :-1]
    median_cadence = masked_median(dt, dt_valid)
    win_samples = (0.5 / median_cadence.clamp(min=1e-10)).round().long()
    W = int(win_samples.median().item()) | 1
    _sync()
    results["steps"]["cadence_calc"] = time.perf_counter() - t0
    results["window_size"] = W

    B, L = flux.shape
    N_pos = L - W + 1

    # Step 1: Unfold + contiguous
    _sync()
    t0 = time.perf_counter()
    flux_windows = flux.unfold(dimension=1, size=W, step=1).contiguous()
    valid_windows = valid_mask.unfold(dimension=1, size=W, step=1).contiguous()
    seg_windows = segment_id.unfold(dimension=1, size=W, step=1).contiguous()
    _sync()
    results["steps"]["unfold_contiguous"] = time.perf_counter() - t0
    results["window_tensor_shape"] = list(flux_windows.shape)
    results["window_tensor_MB"] = flux_windows.nelement() * 4 / 1024**2

    # Step 2: Segment mask
    _sync()
    t0 = time.perf_counter()
    center_seg = seg_windows[:, :, W // 2: W // 2 + 1]
    window_valid = valid_windows & (seg_windows == center_seg)
    _sync()
    results["steps"]["segment_mask"] = time.perf_counter() - t0

    # Step 3: Initial median
    _sync()
    t0 = time.perf_counter()
    location = masked_median(flux_windows, window_valid)
    _sync()
    results["steps"]["initial_median"] = time.perf_counter() - t0

    # Step 4-5: Single biweight iteration (time one iteration)
    _sync()
    t0 = time.perf_counter()
    abs_dev = (flux_windows - location.unsqueeze(-1)).abs()
    _sync()
    results["steps"]["iter_abs_dev"] = time.perf_counter() - t0

    _sync()
    t0 = time.perf_counter()
    mad = masked_median(abs_dev, window_valid)
    _sync()
    results["steps"]["iter_mad_median"] = time.perf_counter() - t0

    _sync()
    t0 = time.perf_counter()
    safe_mad = (5.0 * mad.clamp(min=1e-10)).unsqueeze(-1)
    u = (flux_windows - location.unsqueeze(-1)) / safe_mad
    weights = ((1.0 - u**2) ** 2) * (u.abs() < 1.0).float() * window_valid.float()
    w_sum = weights.sum(dim=-1).clamp(min=1e-10)
    location = (flux_windows * weights).sum(dim=-1) / w_sum
    _sync()
    results["steps"]["iter_weights_update"] = time.perf_counter() - t0

    # Full 10 iterations
    location = masked_median(flux_windows, window_valid)
    _sync()
    t0 = time.perf_counter()
    for _ in range(10):
        abs_dev = (flux_windows - location.unsqueeze(-1)).abs()
        mad = masked_median(abs_dev, window_valid)
        safe_mad = (5.0 * mad.clamp(min=1e-10)).unsqueeze(-1)
        u = (flux_windows - location.unsqueeze(-1)) / safe_mad
        weights = ((1.0 - u**2) ** 2) * (u.abs() < 1.0).float() * window_valid.float()
        w_sum = weights.sum(dim=-1).clamp(min=1e-10)
        location = (flux_windows * weights).sum(dim=-1) / w_sum
    _sync()
    results["steps"]["full_10_iterations"] = time.perf_counter() - t0
    results["steps"]["per_iteration_avg"] = results["steps"]["full_10_iterations"] / 10

    # Breakdown: how much of each iteration is median vs weights
    # Time 10x median calls
    location = masked_median(flux_windows, window_valid)
    _sync()
    t0 = time.perf_counter()
    for _ in range(10):
        abs_dev = (flux_windows - location.unsqueeze(-1)).abs()
        mad = masked_median(abs_dev, window_valid)
    _sync()
    results["steps"]["10x_mad_median_only"] = time.perf_counter() - t0

    # Time 10x weight update calls (no median)
    mad_fixed = mad.clone()
    _sync()
    t0 = time.perf_counter()
    for _ in range(10):
        safe_mad = (5.0 * mad_fixed.clamp(min=1e-10)).unsqueeze(-1)
        u = (flux_windows - location.unsqueeze(-1)) / safe_mad
        weights = ((1.0 - u**2) ** 2) * (u.abs() < 1.0).float() * window_valid.float()
        w_sum = weights.sum(dim=-1).clamp(min=1e-10)
        location = (flux_windows * weights).sum(dim=-1) / w_sum
    _sync()
    results["steps"]["10x_weight_update_only"] = time.perf_counter() - t0

    # Time topk directly
    working = flux_windows.clone()
    working[~window_valid] = float("inf")
    _sync()
    t0 = time.perf_counter()
    sorted_vals = working.topk(k=W, dim=-1, largest=False, sorted=True).values
    _sync()
    results["steps"]["single_topk"] = time.perf_counter() - t0

    # Time torch.sort for comparison
    _sync()
    t0 = time.perf_counter()
    sorted_vals2 = torch.sort(working, dim=-1).values
    _sync()
    results["steps"]["single_sort"] = time.perf_counter() - t0

    return results


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print(f"TorchFlat Profiler - {timestamp}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 60)

    # Warmup
    print("\nWarming up GPU...")
    if torch.cuda.is_available():
        x = torch.randn(100, 100, device="cuda")
        _ = x @ x.T
        torch.cuda.synchronize()

    # Profile pipeline steps
    print("\n--- Pipeline Step Profiling (20 stars x 18000 points) ---")
    pipeline_results = profile_pipeline_steps(n_stars=20, n_points=18000)
    total = pipeline_results["total_time"]

    print(f"\n{'Step':<25} {'Time (s)':>10} {'% of total':>12}")
    print("-" * 50)
    for step, t in sorted(pipeline_results["steps"].items(), key=lambda x: -x[1]):
        pct = t / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {step:<23} {t:>10.3f} {pct:>10.1f}%  {bar}")
    print("-" * 50)
    print(f"  {'TOTAL':<23} {total:>10.3f}")
    print(f"  Stars/sec: {pipeline_results['stars_per_sec']:.2f}")
    print(f"  ms/star: {pipeline_results['ms_per_star']:.1f}")

    # Profile biweight internals
    print("\n--- Biweight Kernel Internals (20 stars x 18000 points) ---")
    bw_results = profile_biweight_internals(n_stars=20, n_points=18000)

    print(f"\n  Window size: {bw_results['window_size']}")
    print(f"  Window tensor shape: {bw_results['window_tensor_shape']}")
    print(f"  Window tensor size: {bw_results['window_tensor_MB']:.1f} MB")

    print(f"\n{'Step':<25} {'Time (s)':>10}")
    print("-" * 40)
    for step, t in bw_results["steps"].items():
        print(f"  {step:<23} {t:>10.4f}")

    iter_total = bw_results["steps"]["full_10_iterations"]
    med_time = bw_results["steps"]["10x_mad_median_only"]
    wt_time = bw_results["steps"]["10x_weight_update_only"]
    print(f"\n  Iteration breakdown:")
    print(f"    MAD median (topk):  {med_time:.3f}s ({med_time/iter_total*100:.0f}%)")
    print(f"    Weight update:      {wt_time:.3f}s ({wt_time/iter_total*100:.0f}%)")
    print(f"    topk vs sort:       topk={bw_results['steps']['single_topk']:.4f}s, sort={bw_results['steps']['single_sort']:.4f}s")

    # Save to log file
    log_data = {
        "timestamp": timestamp,
        "pipeline": pipeline_results,
        "biweight_internals": bw_results,
    }
    log_file = LOGS_DIR / f"profile_{timestamp}.json"
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2, default=str)
    print(f"\nLog saved to: {log_file}")


if __name__ == "__main__":
    main()
