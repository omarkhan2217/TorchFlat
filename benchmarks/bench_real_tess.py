#!/usr/bin/env python
"""Benchmark TorchFlat on real TESS sector data.

Usage:
    python benchmarks/bench_real_tess.py --data-dir <path> [--n-stars 500] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def load_tess_lc(fits_path: Path) -> dict | None:
    """Load a TESS light curve from FITS."""
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[1].data
            return {
                "time": np.array(data["TIME"], dtype=np.float64),
                "pdcsap_flux": np.array(data["PDCSAP_FLUX"], dtype=np.float32),
                "sap_flux": np.array(data["SAP_FLUX"], dtype=np.float32),
                "quality": np.array(data["QUALITY"], dtype=np.int32),
            }
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--n-stars", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-track-b", action="store_true", help="Skip Track B (FFT highpass)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    fits_files = sorted(data_dir.glob("*.fits"))[: args.n_stars]
    print(f"Loading {len(fits_files)} FITS files from {data_dir}...")

    t0 = time.perf_counter()
    star_data = []
    for f in fits_files:
        lc = load_tess_lc(f)
        if lc is not None:
            star_data.append(lc)
    load_time = time.perf_counter() - t0
    print(f"Loaded {len(star_data)} stars in {load_time:.1f}s")

    # Show data stats
    lengths = [len(s["time"]) for s in star_data]
    print(f"LC lengths: min={min(lengths)}, max={max(lengths)}, median={np.median(lengths):.0f}")

    import torchflat

    device = args.device
    print(f"\nProcessing on {device}...")
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(f"Track B: {'SKIPPED' if args.skip_track_b else 'enabled'}")

    t0 = time.perf_counter()
    results, skipped = torchflat.preprocess_sector(
        star_data,
        device=device,
        window_scales=[(256, 128), (2048, 512)],
        skip_track_b=args.skip_track_b,
    )
    elapsed = time.perf_counter() - t0

    n_processed = sum(1 for r in results if r)
    rate = n_processed / elapsed

    print(f"\n=== Results ===")
    print(f"  Stars:     {n_processed} processed, {len(skipped)} skipped")
    print(f"  Time:      {elapsed:.1f}s")
    print(f"  Rate:      {rate:.1f} stars/sec")
    print(f"  ms/star:   {elapsed / max(n_processed, 1) * 1000:.1f}")

    if device == "cuda" and torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak VRAM: {peak_mb:.0f} MB")

    # Project to full sector
    total_stars = 19618
    projected = total_stars / rate
    print(f"\n  Full sector ({total_stars} stars): {projected / 60:.1f} min")
    print(f"  vs Celix wotan 12-worker: ~78 min")
    print(f"  Speedup: {78 * 60 / projected:.1f}x")

    # Save log
    log = {
        "timestamp": datetime.now().isoformat(),
        "n_stars_loaded": len(star_data),
        "n_processed": n_processed,
        "n_skipped": len(skipped),
        "elapsed_s": elapsed,
        "stars_per_sec": rate,
        "device": device,
        "peak_vram_mb": peak_mb if device == "cuda" else None,
        "sector_projection_min": projected / 60,
    }
    log_file = LOGS_DIR / f"bench_real_tess_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nLog: {log_file}")


if __name__ == "__main__":
    main()
