#!/usr/bin/env python
"""Benchmarks for torchflat vs CPU baseline.

Usage:
    python benchmarks/bench_vs_wotan.py                 # all benchmarks
    python benchmarks/bench_vs_wotan.py --bench biweight
    python benchmarks/bench_vs_wotan.py --bench pipeline
    python benchmarks/bench_vs_wotan.py --bench scaling
    python benchmarks/bench_vs_wotan.py --bench precision
    python benchmarks/bench_vs_wotan.py --bench vram
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

CADENCE = 2.0 / 1440.0


def _make_stars(n_stars: int, n_points: int = 18000, seed: int = 0):
    """Generate synthetic star data."""
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


def _make_sector_data(n_stars: int, n_points: int = 18000, seed: int = 0):
    """Generate synthetic sector data for preprocess_sector."""
    rng = np.random.default_rng(seed)
    star_data = []
    for _ in range(n_stars):
        t = np.arange(n_points, dtype=np.float64) * CADENCE
        star_data.append({
            "time": t,
            "pdcsap_flux": (1.0 + rng.normal(0, 0.001, n_points)).astype(np.float32),
            "sap_flux": (1.0 + rng.normal(0, 0.001, n_points)).astype(np.float32),
            "quality": np.zeros(n_points, dtype=np.int32),
        })
    return star_data


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_biweight_kernel(n_stars: int = 50, n_points: int = 18000):
    """Benchmark biweight kernel only."""
    from torchflat.biweight import biweight_detrend

    print(f"\n=== Biweight Kernel: {n_stars} stars x {n_points} points ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(42)

    time_t = torch.arange(n_points, dtype=torch.float64).unsqueeze(0).expand(n_stars, -1).contiguous() * CADENCE
    flux_t = torch.tensor(
        1.0 + rng.normal(0, 0.001, (n_stars, n_points)),
        dtype=torch.float32,
    )
    valid = torch.ones(n_stars, n_points, dtype=torch.bool)
    seg_id = torch.zeros(n_stars, n_points, dtype=torch.int32)

    time_t = time_t.to(device)
    flux_t = flux_t.to(device)
    valid = valid.to(device)
    seg_id = seg_id.to(device)

    # Warmup
    biweight_detrend(flux_t[:1], time_t[:1], valid[:1], seg_id[:1])
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    biweight_detrend(flux_t, time_t, valid, seg_id)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    print(f"  Device: {device}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Stars/sec: {n_stars / elapsed:.1f}")
    print(f"  ms/star: {elapsed / n_stars * 1000:.1f}")


def bench_full_pipeline(n_stars: int = 50, n_points: int = 18000):
    """Benchmark full Track A pipeline."""
    from torchflat.pipeline import preprocess_track_a

    print(f"\n=== Full Pipeline (Track A): {n_stars} stars x {n_points} points ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    times, fluxes, qualities = _make_stars(n_stars, n_points)

    # Warmup
    preprocess_track_a(times[:1], fluxes[:1], qualities[:1],
                       device=device, window_scales=[(256, 128)])

    t0 = time.perf_counter()
    results, skipped = preprocess_track_a(
        times, fluxes, qualities,
        device=device, window_scales=[(256, 128)],
    )
    elapsed = time.perf_counter() - t0

    n_processed = sum(1 for r in results if r is not None)
    print(f"  Device: {device}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Stars processed: {n_processed}, skipped: {len(skipped)}")
    print(f"  Stars/sec: {n_processed / elapsed:.1f}")


def bench_scaling_stars():
    """Throughput vs star count."""
    from torchflat.pipeline import preprocess_track_a

    device = "cuda" if torch.cuda.is_available() else "cpu"
    counts = [1, 5, 10, 50, 100]
    if torch.cuda.is_available():
        counts.extend([500, 1000])

    print(f"\n=== Scaling: Stars (device={device}) ===")
    print(f"  {'Stars':>8} {'Time (s)':>10} {'Stars/sec':>12}")
    print(f"  {'-'*8} {'-'*10} {'-'*12}")

    for n in counts:
        times, fluxes, qualities = _make_stars(n, 5000)
        t0 = time.perf_counter()
        preprocess_track_a(
            times, fluxes, qualities,
            device=device, window_scales=[(256, 128)],
        )
        elapsed = time.perf_counter() - t0
        print(f"  {n:>8} {elapsed:>10.3f} {n / elapsed:>12.1f}")


def bench_scaling_length():
    """Throughput vs light curve length."""
    from torchflat.pipeline import preprocess_track_a

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lengths = [2000, 5000, 10000, 15000]
    n_stars = 20

    print(f"\n=== Scaling: LC Length ({n_stars} stars, device={device}) ===")
    print(f"  {'Length':>8} {'Time (s)':>10} {'ms/star':>10}")
    print(f"  {'-'*8} {'-'*10} {'-'*10}")

    for L in lengths:
        times, fluxes, qualities = _make_stars(n_stars, L)
        t0 = time.perf_counter()
        preprocess_track_a(
            times, fluxes, qualities,
            device=device, window_scales=[(256, 128)],
        )
        elapsed = time.perf_counter() - t0
        print(f"  {L:>8} {elapsed:>10.3f} {elapsed / n_stars * 1000:>10.1f}")


def bench_precision_profile():
    """Float32 vs float64 accuracy and performance."""
    from torchflat.biweight import biweight_detrend

    print("\n=== Precision Profile ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(42)
    n_stars, n_points = 20, 5000

    time_t = torch.arange(n_points, dtype=torch.float64).unsqueeze(0).expand(n_stars, -1).contiguous() * CADENCE
    flux_np = 1.0 + rng.normal(0, 0.001, (n_stars, n_points))
    flux_t = torch.tensor(flux_np, dtype=torch.float64).to(device)
    valid = torch.ones(n_stars, n_points, dtype=torch.bool, device=device)
    seg_id = torch.zeros(n_stars, n_points, dtype=torch.int32, device=device)
    time_t = time_t.to(device)

    # float64
    t0 = time.perf_counter()
    _, trend_f64 = biweight_detrend(flux_t, time_t, valid, seg_id, dtype=torch.float64)
    if device.type == "cuda":
        torch.cuda.synchronize()
    time_f64 = time.perf_counter() - t0

    # float32
    t0 = time.perf_counter()
    _, trend_f32 = biweight_detrend(flux_t.float(), time_t, valid, seg_id, dtype=torch.float32)
    if device.type == "cuda":
        torch.cuda.synchronize()
    time_f32 = time.perf_counter() - t0

    # Compare
    both_finite = torch.isfinite(trend_f64) & torch.isfinite(trend_f32)
    if both_finite.sum() > 0:
        t64 = trend_f64[both_finite].double()
        t32 = trend_f32[both_finite].double()
        rel_err = ((t32 - t64) / t64.clamp(min=1e-10)).abs()
        print(f"  Relative error: max={rel_err.max().item():.2e}, "
              f"mean={rel_err.mean().item():.2e}, "
              f"p99={rel_err.quantile(0.99).item():.2e}")

    print(f"  float64 time: {time_f64:.3f}s")
    print(f"  float32 time: {time_f32:.3f}s")
    print(f"  Speedup: {time_f64 / max(time_f32, 1e-6):.2f}x")


def bench_vram_profile():
    """Peak VRAM usage vs batch size (GPU only)."""
    if not torch.cuda.is_available():
        print("\n=== VRAM Profile: SKIPPED (no GPU) ===")
        return

    from torchflat.batching import estimate_peak_vram
    from torchflat.biweight import biweight_detrend

    print("\n=== VRAM Profile ===")
    batch_sizes = [1, 5, 10, 20]
    n_points = 5000

    print(f"  {'Batch':>6} {'Estimated (MB)':>15} {'Actual (MB)':>13} {'Ratio':>8}")
    print(f"  {'-'*6} {'-'*15} {'-'*13} {'-'*8}")

    device = torch.device("cuda")

    for bs in batch_sizes:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        time_t = torch.arange(n_points, dtype=torch.float64, device=device).unsqueeze(0).expand(bs, -1).contiguous() * CADENCE
        flux_t = torch.ones(bs, n_points, dtype=torch.float32, device=device)
        valid = torch.ones(bs, n_points, dtype=torch.bool, device=device)
        seg_id = torch.zeros(bs, n_points, dtype=torch.int32, device=device)

        biweight_detrend(flux_t, time_t, valid, seg_id)
        torch.cuda.synchronize()

        actual_mb = torch.cuda.max_memory_allocated() / 1024**2
        estimated_mb = estimate_peak_vram(n_points, 361) * bs / 1024**2

        ratio = actual_mb / max(estimated_mb, 1)
        print(f"  {bs:>6} {estimated_mb:>15.1f} {actual_mb:>13.1f} {ratio:>8.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TorchFlat benchmarks")
    parser.add_argument("--bench", type=str, default="all",
                        choices=["all", "biweight", "pipeline", "scaling", "precision", "vram"],
                        help="Which benchmark to run")
    args = parser.parse_args()

    benchmarks = {
        "biweight": bench_biweight_kernel,
        "pipeline": bench_full_pipeline,
        "scaling": lambda: (bench_scaling_stars(), bench_scaling_length()),
        "precision": bench_precision_profile,
        "vram": bench_vram_profile,
    }

    if args.bench == "all":
        for name, fn in benchmarks.items():
            fn()
    else:
        benchmarks[args.bench]()


if __name__ == "__main__":
    main()
