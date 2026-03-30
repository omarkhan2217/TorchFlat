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


# Mission presets: column names and quality bitmasks
MISSION_PRESETS = {
    "tess": {
        "time": "TIME",
        "flux": "PDCSAP_FLUX",
        "flux_raw": "SAP_FLUX",
        "quality": "QUALITY",
        "bitmask": 0b0000110101111111,  # 3455
    },
    "kepler": {
        "time": "TIME",
        "flux": "PDCSAP_FLUX",
        "flux_raw": "SAP_FLUX",
        "quality": "SAP_QUALITY",
        "bitmask": 0b0001111111111111,  # 8191
    },
    "k2": {
        "time": "TIME",
        "flux": "PDCSAP_FLUX",
        "flux_raw": "SAP_FLUX",
        "quality": "SAP_QUALITY",
        "bitmask": 0b0001111111111111,  # 8191
    },
}


def _load_fits(fits_dir: Path, n_stars: int = 0, mission: str = "tess",
               col_time: str = None, col_flux: str = None,
               col_flux_raw: str = None, col_quality: str = None):
    """Load light curves from FITS files.

    Supports TESS, Kepler, and K2 via mission presets, or custom column
    names for any mission.
    """
    from astropy.io import fits as pyfits

    preset = MISSION_PRESETS.get(mission, MISSION_PRESETS["tess"])
    time_col = col_time or preset["time"]
    flux_col = col_flux or preset["flux"]
    raw_col = col_flux_raw or preset["flux_raw"]
    qual_col = col_quality or preset["quality"]

    files = sorted(fits_dir.glob("*.fits"))
    if n_stars > 0:
        files = files[:n_stars]

    star_data = []
    for f in files:
        try:
            with pyfits.open(f) as h:
                d = h[1].data
                star_data.append({
                    "time": np.array(d[time_col], dtype=np.float64),
                    "pdcsap_flux": np.array(d[flux_col], dtype=np.float32),
                    "sap_flux": np.array(d[raw_col], dtype=np.float32),
                    "quality": np.array(d[qual_col], dtype=np.int32),
                    "filename": f.name,
                })
        except (KeyError, ValueError, OSError) as e:
            print(f"  Warning: skipping {f.name}: {e}", file=sys.stderr)
            continue
    return star_data


def _add_common_args(parser):
    """Add arguments shared between umi_detrend and benchmark."""
    # Input/output
    parser.add_argument("--input", "-i", required=True,
                        help="Directory with FITS files, or a single FITS file")
    parser.add_argument("--n-stars", "-n", type=int, default=0,
                        help="Max stars to process (0=all)")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda or cpu (default: cuda)")

    # Mission/data format
    parser.add_argument("--mission", default="tess", choices=["tess", "kepler", "k2"],
                        help="Mission preset for column names and quality bitmask (default: tess)")
    parser.add_argument("--col-time", default=None,
                        help="Override time column name (default: from mission preset)")
    parser.add_argument("--col-flux", default=None,
                        help="Override flux column name (default: from mission preset)")
    parser.add_argument("--col-quality", default=None,
                        help="Override quality column name (default: from mission preset)")

    # Batch/memory
    parser.add_argument("--max-batch", type=int, default=None,
                        help="Override batch size (auto-detected from VRAM by default)")
    parser.add_argument("--vram-budget", type=float, default=None,
                        help="VRAM budget in GB (auto-detected by default)")

    # UMI detrending
    parser.add_argument("--asymmetry", type=float, default=2.0,
                        help="Dip penalty factor. 2.0=best accuracy (default), 1.5=mixed, 1.0=variable stars")
    parser.add_argument("--window-length", type=float, default=0.5,
                        help="Sliding window width in days (default: 0.5)")
    parser.add_argument("--cval", type=float, default=5.0,
                        help="Rejection threshold in MAD units (default: 5.0)")
    parser.add_argument("--n-iter", type=int, default=5,
                        help="Number of bisquare iterations (default: 5)")

    # Bias correction
    parser.add_argument("--bias-correct", action="store_true",
                        help="Correct the asymmetry-induced bias on detrended flux "
                             "(recommended for population studies)")

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

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        sys.exit(1)

    # Single file or directory
    if input_path.is_file() and input_path.suffix == ".fits":
        fits_dir = input_path.parent
        output_dir = Path(args.output) if args.output else fits_dir / "torchflat_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        star_data = _load_fits(fits_dir, 0, mission=args.mission,
                                   col_time=args.col_time, col_flux=args.col_flux,
                                   col_quality=args.col_quality)
        # Filter to just the requested file
        star_data = [s for s in star_data if s.get("filename") == input_path.name]
    else:
        fits_dir = input_path
        output_dir = Path(args.output) if args.output else fits_dir.parent / "torchflat_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nLoading FITS files...")
        t0 = time.perf_counter()
        star_data = _load_fits(fits_dir, args.n_stars, mission=args.mission,
                                   col_time=args.col_time, col_flux=args.col_flux,
                                   col_quality=args.col_quality)
        print(f"Loaded {len(star_data)} stars in {time.perf_counter()-t0:.1f}s")

    print(f"TorchFlat v{torchflat.__version__}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")

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

    # Bias correction
    if args.bias_correct:
        from torchflat.umi import UMI_BIAS_PPM
        asym = args.asymmetry
        keys = sorted(UMI_BIAS_PPM.keys())
        if asym in UMI_BIAS_PPM:
            bias = UMI_BIAS_PPM[asym] / 1e6
        elif asym < keys[0]:
            bias = UMI_BIAS_PPM[keys[0]] / 1e6
        elif asym > keys[-1]:
            bias = UMI_BIAS_PPM[keys[-1]] / 1e6
        else:
            lo = max(k for k in keys if k <= asym)
            hi = min(k for k in keys if k >= asym)
            frac = (asym - lo) / (hi - lo) if hi != lo else 0
            bias = (UMI_BIAS_PPM[lo] + frac * (UMI_BIAS_PPM[hi] - UMI_BIAS_PPM[lo])) / 1e6
        print(f"Bias correction: {bias*1e6:+.0f} ppm (asymmetry={asym})")
        for r in results:
            if r and "detrended" in r:
                r["detrended"] = r["detrended"] - bias

    n_ok = sum(1 for r in results if r)
    rate = n_ok / elapsed if elapsed > 0 else 0
    print(f"\nDone: {n_ok} processed, {len(skipped)} skipped in {elapsed:.1f}s ({rate:.1f}/sec)")

    # Save results
    fmt = getattr(args, "output_format", "npz")
    for i, result in enumerate(results):
        if not result:
            continue
        fname = star_data[i].get("filename", f"star_{i:05d}")
        stem = Path(fname).stem

        if fmt == "fits":
            from astropy.io import fits as pyfits
            from astropy.table import Table

            table_data = {}
            for key, val in result.items():
                if isinstance(val, np.ndarray) and val.ndim == 1:
                    table_data[key] = val
                elif hasattr(val, "numpy") and val.ndim == 1:
                    table_data[key] = val.numpy()
            if table_data:
                t = Table(table_data)
                out_file = output_dir / f"{stem}_detrended.fits"
                t.write(out_file, format="fits", overwrite=True)
        else:
            out_file = output_dir / f"{stem}_detrended.npz"
            save_dict = {}
            for key, val in result.items():
                if isinstance(val, np.ndarray):
                    save_dict[key] = val
                elif hasattr(val, "numpy"):
                    save_dict[key] = val.numpy()
            np.savez_compressed(out_file, **save_dict)

    print(f"Results saved to {output_dir}/ ({fmt} format)")

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
    star_data = _load_fits(fits_dir, args.n_stars, mission=args.mission,
                               col_time=args.col_time, col_flux=args.col_flux,
                               col_quality=args.col_quality)
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
    p_det.add_argument("--output-format", default="fits", choices=["fits", "npz"],
                        help="Output format: fits (FITS table) or npz (numpy) (default: fits)")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Run speed benchmark")
    _add_common_args(p_bench)
    # Override n-stars default for benchmark
    for action in p_bench._actions:
        if hasattr(action, "dest") and action.dest == "n_stars":
            action.default = 500
            break

    # plot
    p_plot = sub.add_parser("plot", help="Plot detrended light curve for a single FITS file")
    p_plot.add_argument("--fits", "-f", required=True, help="Path to a single FITS file")
    p_plot.add_argument("--mission", default="tess", choices=["tess", "kepler", "k2"])
    p_plot.add_argument("--col-time", default=None)
    p_plot.add_argument("--col-flux", default=None)
    p_plot.add_argument("--col-quality", default=None)
    p_plot.add_argument("--asymmetry", type=float, default=2.0)
    p_plot.add_argument("--device", default="cuda")
    p_plot.add_argument("--save", "-s", default=None, help="Save plot to file instead of showing")

    args = parser.parse_args()

    if args.command == "umi_detrend":
        cmd_detrend(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "plot":
        cmd_plot(args)
    else:
        parser.print_help()


def cmd_plot(args):
    """Plot a single star: raw flux, trend, detrended."""
    import torch
    import matplotlib.pyplot as plt
    from astropy.io import fits as pyfits

    preset = MISSION_PRESETS.get(args.mission, MISSION_PRESETS["tess"])
    time_col = args.col_time or preset["time"]
    flux_col = args.col_flux or preset["flux"]
    qual_col = args.col_quality or preset["quality"]
    bitmask = preset["bitmask"]

    fits_path = Path(args.fits)
    if not fits_path.exists():
        print(f"Error: {fits_path} does not exist")
        sys.exit(1)

    with pyfits.open(fits_path) as h:
        d = h[1].data
        t = np.array(d[time_col], dtype=np.float64)
        fl = np.array(d[flux_col], dtype=np.float64)
        q = np.array(d[qual_col], dtype=np.int32)

    v = ((q & bitmask) == 0) & np.isfinite(fl) & np.isfinite(t)
    t_v, f_v = t[v], fl[v]
    print(f"Star: {fits_path.name}, {len(t_v)} valid points")

    # Check GPU
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    from torchflat._kernel_loader import _get_umi_kernel
    kern = _get_umi_kernel() if device == "cuda" else None

    # Detrend
    cad = np.median(np.diff(t_v))
    W = int(round(0.5 / cad)) | 1
    L = len(f_v)
    min_valid = min(50, W // 2)

    if kern and device == "cuda":
        ft = torch.tensor(f_v, dtype=torch.float32, device="cuda").unsqueeze(0)
        vt = torch.ones(1, L, dtype=torch.bool, device="cuda")
        st = torch.zeros(1, L, dtype=torch.int32, device="cuda")
        loc = kern.umi_detrend_direct(ft, vt, st, W, 5.0, args.asymmetry, 5, min_valid)
        torch.cuda.synchronize()
        N_pos = L - W + 1
        off = W // 2
        trend = np.full(L, np.nan)
        trend[off:off + N_pos] = loc[0].cpu().numpy()
    else:
        from torchflat.umi import umi_detrend
        ft = torch.tensor(f_v, dtype=torch.float32).unsqueeze(0)
        tt = torch.tensor(t_v, dtype=torch.float64).unsqueeze(0)
        vt = torch.ones(1, L, dtype=torch.bool)
        st = torch.zeros(1, L, dtype=torch.int32)
        _, trend_t = umi_detrend(ft, tt, vt, st, asymmetry=args.asymmetry)
        trend = trend_t[0].numpy()

    detrended = np.where((trend > 0) & np.isfinite(trend), f_v / trend, np.nan)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"TorchFlat UMI - {fits_path.name}", fontsize=13)

    # Panel 1: Raw flux + trend
    axes[0].plot(t_v, f_v, ".", ms=0.5, color="gray", alpha=0.5, label="Raw flux")
    finite_t = np.isfinite(trend)
    axes[0].plot(t_v[finite_t], trend[finite_t], "-", color="red", lw=1.2, label="UMI trend")
    axes[0].set_ylabel("Flux")
    axes[0].legend(loc="upper right", fontsize=9)

    # Panel 2: Detrended
    finite_d = np.isfinite(detrended)
    axes[1].plot(t_v[finite_d], detrended[finite_d], ".", ms=0.5, color="black", alpha=0.5)
    axes[1].axhline(1.0, color="red", ls="--", lw=0.8, alpha=0.5)
    axes[1].set_ylabel("Detrended (flux/trend)")
    med = np.nanmedian(detrended[finite_d])
    std = np.nanstd(detrended[finite_d])
    axes[1].set_ylim(med - 5 * std, med + 5 * std)

    # Panel 3: Detrended zoomed (transit depth scale)
    axes[2].plot(t_v[finite_d], (detrended[finite_d] - 1) * 1e6, ".", ms=0.5, color="black", alpha=0.5)
    axes[2].axhline(0, color="red", ls="--", lw=0.8, alpha=0.5)
    axes[2].set_ylabel("Residual (ppm)")
    axes[2].set_xlabel("Time (days)")
    axes[2].set_ylim(-5000, 5000)

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
