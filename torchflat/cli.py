"""TorchFlat CLI -- command-line interface for UMI detrending.

Usage:
    torchflat umi_detrend --input <fits_dir> [--output <dir>] [options]
    torchflat benchmark --input <fits_dir> [--n-stars N] [options]
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def _progress_bar(done, total):
    """Simple print-based progress bar."""
    width = 40
    frac = done / max(total, 1)
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r  [{bar}] {done}/{total} ({frac*100:.0f}%)", end="", flush=True)
    if done >= total:
        print()


def _load_fits(fits_dir: Path, n_stars: int = 0):
    """Load TESS light curves from FITS files."""
    from astropy.io import fits as pyfits

    files = sorted(fits_dir.glob("*.fits"))
    if n_stars > 0:
        files = files[:n_stars]

    star_data = []
    for f in files:
        try:
            with pyfits.open(f) as h:
                d = h[1].data
                star_data.append({
                    "time": np.array(d["TIME"], dtype=np.float64),
                    "pdcsap_flux": np.array(d["PDCSAP_FLUX"], dtype=np.float32),
                    "sap_flux": np.array(d["SAP_FLUX"], dtype=np.float32),
                    "quality": np.array(d["QUALITY"], dtype=np.int32),
                    "filename": f.name,
                })
        except Exception:
            continue
    return star_data


def _add_common_args(parser):
    """Add arguments shared between umi_detrend and benchmark."""
    # Input/output
    parser.add_argument("--input", "-i", required=True,
                        help="Directory with FITS files")
    parser.add_argument("--n-stars", "-n", type=int, default=0,
                        help="Max stars to process (0=all)")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda or cpu (default: cuda)")

    # Batch/memory
    parser.add_argument("--max-batch", type=int, default=None,
                        help="Override batch size (auto-detected from VRAM by default)")
    parser.add_argument("--vram-budget", type=float, default=None,
                        help="VRAM budget in GB (auto-detected by default)")

    # UMI detrending
    parser.add_argument("--asymmetry", type=float, default=1.5,
                        help="Dip penalty factor. 1.0=symmetric biweight, 1.5=default UMI (default: 1.5)")
    parser.add_argument("--window-length", type=float, default=0.5,
                        help="Sliding window width in days (default: 0.5)")
    parser.add_argument("--cval", type=float, default=5.0,
                        help="Rejection threshold in MAD units (default: 5.0)")
    parser.add_argument("--n-iter", type=int, default=5,
                        help="Number of bisquare iterations (default: 5)")

    # Clipping
    parser.add_argument("--clip-sigma", type=float, default=5.0,
                        help="Sigma clipping threshold for Track A (default: 5.0)")

    # Track B
    parser.add_argument("--skip-track-b", action="store_true",
                        help="Skip Track B (FFT highpass)")
    parser.add_argument("--cutoff-days", type=float, default=5.0,
                        help="Track B FFT highpass cutoff period in days (default: 5.0)")

    # Window extraction
    parser.add_argument("--window-scales", type=str, default=None,
                        help="Window scales as 'size:stride,size:stride,...' "
                             "(default: 256:128,512:256,2048:512,8192:2048)")


def _parse_window_scales(s):
    """Parse '256:128,2048:512' into [(256,128),(2048,512)]."""
    if s is None:
        return None
    scales = []
    for pair in s.split(","):
        size, stride = pair.strip().split(":")
        scales.append((int(size), int(stride)))
    return scales


def _build_kwargs(args):
    """Build kwargs dict for preprocess_sector from parsed args."""
    kwargs = {
        "device": args.device,
        "max_batch": args.max_batch,
        "vram_budget_gb": args.vram_budget,
        "skip_track_b": args.skip_track_b,
        "window_length_days": args.window_length,
        "clip_sigma": args.clip_sigma,
        "biweight_iter": args.n_iter,
        "asymmetry": args.asymmetry,
        "progress_callback": _progress_bar,
    }
    scales = _parse_window_scales(args.window_scales)
    if scales is not None:
        kwargs["window_scales"] = scales
    return kwargs


def cmd_detrend(args):
    """Detrend TESS light curves and save results."""
    import torch
    import torchflat

    fits_dir = Path(args.input)
    if not fits_dir.exists():
        print(f"Error: {fits_dir} does not exist")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else fits_dir.parent / "torchflat_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"TorchFlat v{torchflat.__version__}")
    print(f"Input:  {fits_dir}")
    print(f"Output: {output_dir}")

    # Load
    print(f"\nLoading FITS files...")
    t0 = time.perf_counter()
    star_data = _load_fits(fits_dir, args.n_stars)
    print(f"Loaded {len(star_data)} stars in {time.perf_counter()-t0:.1f}s")

    if not star_data:
        print("No valid FITS files found.")
        sys.exit(1)

    # Check GPU
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        from torchflat._kernel_loader import _get_umi_kernel
        kern = _get_umi_kernel()
        print(f"UMI kernel: {'LOADED' if kern else 'fallback (torch.sort)'}")

    # Show config
    print(f"\nConfig: asymmetry={args.asymmetry}, window={args.window_length}d, "
          f"cval={args.cval}, clip_sigma={args.clip_sigma}")

    # Process
    print(f"\nProcessing {len(star_data)} stars...")
    t0 = time.perf_counter()
    kwargs = _build_kwargs(args)
    kwargs["output_dir"] = str(output_dir)
    results, skipped = torchflat.preprocess_sector(star_data, **kwargs)
    elapsed = time.perf_counter() - t0

    n_ok = sum(1 for r in results if r)
    rate = n_ok / elapsed if elapsed > 0 else 0
    print(f"\nDone: {n_ok} processed, {len(skipped)} skipped in {elapsed:.1f}s ({rate:.1f}/sec)")

    # Save results as npz
    for i, result in enumerate(results):
        if not result:
            continue
        fname = star_data[i].get("filename", f"star_{i:05d}")
        stem = Path(fname).stem
        out_file = output_dir / f"{stem}_detrended.npz"
        save_dict = {}
        for key, val in result.items():
            if isinstance(val, np.ndarray):
                save_dict[key] = val
            elif hasattr(val, "numpy"):
                save_dict[key] = val.numpy()
        np.savez_compressed(out_file, **save_dict)

    print(f"Results saved to {output_dir}/")

    # Summary
    summary = {
        "version": torchflat.__version__,
        "input": str(fits_dir),
        "output": str(output_dir),
        "n_loaded": len(star_data),
        "n_processed": n_ok,
        "n_skipped": len(skipped),
        "elapsed_s": round(elapsed, 1),
        "rate": round(rate, 1),
        "device": device,
        "config": {
            "asymmetry": args.asymmetry,
            "window_length_days": args.window_length,
            "cval": args.cval,
            "n_iter": args.n_iter,
            "clip_sigma": args.clip_sigma,
            "skip_track_b": args.skip_track_b,
        },
        "date": datetime.now().isoformat(),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def cmd_benchmark(args):
    """Run speed benchmark on TESS data."""
    import torch
    import torchflat

    fits_dir = Path(args.input)
    if not fits_dir.exists():
        print(f"Error: {fits_dir} does not exist")
        sys.exit(1)

    print(f"TorchFlat v{torchflat.__version__}")

    # Load
    print(f"\nLoading FITS files...")
    t0 = time.perf_counter()
    star_data = _load_fits(fits_dir, args.n_stars)
    load_time = time.perf_counter() - t0
    print(f"Loaded {len(star_data)} stars in {load_time:.1f}s")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        from torchflat._kernel_loader import _get_umi_kernel
        kern = _get_umi_kernel()
        print(f"UMI kernel: {'LOADED' if kern else 'fallback (torch.sort)'}")

    print(f"\nConfig: asymmetry={args.asymmetry}, window={args.window_length}d, "
          f"cval={args.cval}, clip_sigma={args.clip_sigma}")

    # Benchmark
    print(f"\nBenchmarking ({len(star_data)} stars)...")
    t0 = time.perf_counter()
    kwargs = _build_kwargs(args)
    results, skipped = torchflat.preprocess_sector(star_data, **kwargs)
    elapsed = time.perf_counter() - t0

    n_ok = sum(1 for r in results if r)
    rate = n_ok / elapsed if elapsed > 0 else 0

    print(f"\n=== Results ===")
    print(f"  Stars:     {n_ok} processed, {len(skipped)} skipped")
    print(f"  Time:      {elapsed:.1f}s")
    print(f"  Rate:      {rate:.1f} stars/sec")
    print(f"  ms/star:   {elapsed / max(n_ok, 1) * 1000:.1f}")

    if device == "cuda" and torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak VRAM: {peak_mb:.0f} MB")

    total_stars = 19618
    projected = total_stars / rate if rate > 0 else float("inf")
    print(f"\n  Full sector ({total_stars} stars): {projected / 60:.1f} min")
    print(f"  vs Celix wotan 12-worker: ~78 min")
    print(f"  Speedup: {78 * 60 / projected:.1f}x")


def main():
    parser = argparse.ArgumentParser(
        prog="torchflat",
        description="TorchFlat: GPU-accelerated photometric preprocessing with UMI detrending",
    )
    sub = parser.add_subparsers(dest="command")

    # umi_detrend
    p_det = sub.add_parser("umi_detrend", help="Detrend TESS light curves with UMI")
    _add_common_args(p_det)
    p_det.add_argument("--output", "-o",
                        help="Output directory (default: <input>/../torchflat_output)")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Run speed benchmark")
    _add_common_args(p_bench)
    # Override n-stars default for benchmark
    for action in p_bench._actions:
        if hasattr(action, "dest") and action.dest == "n_stars":
            action.default = 500
            break

    args = parser.parse_args()

    if args.command == "umi_detrend":
        cmd_detrend(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
