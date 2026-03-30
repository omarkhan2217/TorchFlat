"""UMI-only injection grid: 10 periods x 5 durations x 7 depths x 500 stars.

No wotan comparison (runs 10x faster without it).
Saves results to results/injection_grid_umi.json.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
device = torch.device("cuda")

from torchflat._kernel_loader import _get_umi_kernel
kern = _get_umi_kernel()
assert kern is not None


def load_stars(data_dir, n):
    stars = []
    for f in sorted(data_dir.glob("*.fits")):
        if len(stars) >= n: break
        try:
            with fits.open(f) as h:
                d = h[1].data
                t = np.array(d["TIME"], dtype=np.float64)
                fl = np.array(d["PDCSAP_FLUX"], dtype=np.float64)
                q = np.array(d["QUALITY"], dtype=np.int32)
                v = ((q & 3455) == 0) & np.isfinite(fl) & np.isfinite(t)
                if v.sum() > 500: stars.append((t[v], fl[v]))
        except: continue
    return stars


def recover_depth(det, t, period, duration):
    finite = np.isfinite(det)
    if finite.sum() < 100: return np.nan
    d, tt = det[finite], t[finite]
    phase = ((tt - t[0]) % period) / period
    hd = duration / period / 2.0
    in_tr = (phase < hd) | (phase > 1.0 - hd)
    out_tr = ~in_tr
    if in_tr.sum() < 3 or out_tr.sum() < 50: return np.nan
    return float(np.median(d[out_tr]) - np.median(d[in_tr]))


def gpu_detrend(t_v, f_v):
    cadence = np.median(np.diff(t_v))
    W = int(round(0.5 / cadence)) | 1
    L = len(f_v)
    if L < W: return np.full(L, np.nan)
    ft = torch.tensor(f_v, dtype=torch.float32, device=device).unsqueeze(0)
    vt = torch.ones(1, L, dtype=torch.bool, device=device)
    st = torch.zeros(1, L, dtype=torch.int32, device=device)
    loc = kern.umi_detrend_direct(ft, vt, st, W, 5.0, 1.5, 5, 50)
    N_pos = L - W + 1; offset = W // 2
    trend = np.full(L, np.nan)
    trend[offset:offset+N_pos] = loc[0].cpu().numpy()
    return np.where((trend > 0) & np.isfinite(trend), f_v / trend, np.nan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-stars", type=int, default=500)
    args = parser.parse_args()

    data_dir = Path("C:/Users/USER/Desktop/Website/trips-section/Celix/data/fits/sector_6")
    print(f"Loading {args.n_stars} stars...")
    stars = load_stars(data_dir, args.n_stars)
    print(f"Loaded {len(stars)} valid stars")

    periods = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 27]
    durations = [1, 2, 3, 5, 8]
    depths = [0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05]

    total = 0
    for p in periods:
        for dur in durations:
            if dur / 24.0 > p / 2: continue
            total += len(depths)
    print(f"\nGrid: {len(periods)} periods x {len(durations)} durations x {len(depths)} depths")
    print(f"Valid configs: {total} (skipping impossible period/duration combos)")

    results = {}
    done = 0
    t_start = time.perf_counter()

    for period in periods:
        for dur_h in durations:
            if dur_h / 24.0 > period / 2: continue
            duration = dur_h / 24.0

            for depth in depths:
                recs = []
                for t_v, f_v in stars:
                    span = t_v[-1] - t_v[0]
                    if period > span * 0.8: continue

                    phase = ((t_v - t_v[0]) % period) / period
                    hd = duration / period / 2.0
                    if hd >= 0.5: continue
                    in_tr = (phase < hd) | (phase > 1.0 - hd)
                    if in_tr.sum() < 3: continue

                    fi = f_v.copy()
                    fi[in_tr] -= depth * np.median(f_v)
                    det = gpu_detrend(t_v, fi)
                    torch.cuda.synchronize()
                    r = recover_depth(det, t_v, period, duration)
                    if np.isfinite(r) and r > 0: recs.append(r)

                med_err = abs(np.median(recs) - depth) / depth * 100 if len(recs) > 10 else None
                key = f"p{period}_d{dur_h}h_dep{depth}"
                results[key] = {
                    "period": period, "duration_h": dur_h, "depth": depth,
                    "median_error_pct": round(med_err, 1) if med_err is not None else None,
                    "n_valid": len(recs),
                }
                done += 1

                elapsed = time.perf_counter() - t_start
                if done % 20 == 0 or done == total:
                    eta = elapsed / done * (total - done)
                    print(f"  [{done}/{total}] {elapsed:.0f}s, ~{eta/60:.0f}min left", flush=True)

    # Summary tables
    print(f"\n{'='*70}")
    print(f"UMI INJECTION GRID ({len(stars)} stars)")
    print(f"{'='*70}")

    # By depth (across all periods/durations)
    print(f"\nBy depth (median across all periods/durations):")
    print(f"{'Depth':>8}  {'Error':>7}  {'Configs':>8}")
    for depth in depths:
        errs = [r["median_error_pct"] for r in results.values()
                if r["depth"] == depth and r["median_error_pct"] is not None]
        med = np.median(errs) if errs else np.nan
        print(f"{depth:>8.4f}  {med:>5.1f}%  {len(errs):>8}")

    # By period (across all durations/depths)
    print(f"\nBy period (median across all durations/depths):")
    print(f"{'Period':>8}  {'Error':>7}  {'Configs':>8}")
    for period in periods:
        errs = [r["median_error_pct"] for r in results.values()
                if r["period"] == period and r["median_error_pct"] is not None]
        med = np.median(errs) if errs else np.nan
        print(f"{period:>6.1f}d  {med:>5.1f}%  {len(errs):>8}")

    # By duration (across all periods/depths)
    print(f"\nBy duration (median across all periods/depths):")
    print(f"{'Dur':>8}  {'Error':>7}  {'Configs':>8}")
    for dur in durations:
        errs = [r["median_error_pct"] for r in results.values()
                if r["duration_h"] == dur and r["median_error_pct"] is not None]
        med = np.median(errs) if errs else np.nan
        print(f"{dur:>6}h  {med:>5.1f}%  {len(errs):>8}")

    # Heatmap: depth vs period (averaged over durations)
    print(f"\nHeatmap: median error by depth x period (avg over durations)")
    header = f"{'':>8}"
    for p in periods:
        header += f"  {p:>5.1f}d"
    print(header)
    for depth in depths:
        row = f"{depth:>8.4f}"
        for period in periods:
            errs = [r["median_error_pct"] for r in results.values()
                    if r["depth"] == depth and r["period"] == period
                    and r["median_error_pct"] is not None]
            med = np.median(errs) if errs else np.nan
            if np.isfinite(med):
                row += f"  {med:>5.1f}%"
            else:
                row += f"  {'N/A':>5}"
        print(row)

    # Save
    doc = {
        "title": "UMI Injection Recovery Grid",
        "date": datetime.now().isoformat(),
        "config": {
            "n_stars": len(stars),
            "periods": periods, "durations": durations, "depths": depths,
            "asymmetry": 1.5, "cval": 5.0, "window_length": 0.5,
        },
        "per_config": results,
    }
    with open(RESULTS_DIR / "injection_grid_umi.json", "w") as f:
        json.dump(doc, f, indent=2)
    print(f"\nSaved to results/injection_grid_umi.json")
