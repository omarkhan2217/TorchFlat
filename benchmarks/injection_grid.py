"""Injection grid: 10 periods x 5 durations x 7 depths on 500 stars.

Tests TorchFlat UMI vs wotan across a comprehensive grid of transit
configurations. Saves results to results/injection_grid.json.

Usage:
    python benchmarks/injection_grid.py [--n-stars 500]
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wotan
from astropy.io import fits

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

device = torch.device("cuda")

from torchflat._kernel_loader import _get_umi_kernel
kern = _get_umi_kernel()
assert kern is not None, "UMI kernel required"


def load_stars(data_dir, n):
    stars = []
    for f in sorted(data_dir.glob("*.fits")):
        if len(stars) >= n:
            break
        try:
            with fits.open(f) as h:
                d = h[1].data
                t = np.array(d["TIME"], dtype=np.float64)
                fl = np.array(d["PDCSAP_FLUX"], dtype=np.float64)
                q = np.array(d["QUALITY"], dtype=np.int32)
                v = ((q & 3455) == 0) & np.isfinite(fl) & np.isfinite(t)
                if v.sum() > 500:
                    stars.append((t[v], fl[v]))
        except Exception:
            continue
    return stars


def recover_depth(det, t, period, duration):
    finite = np.isfinite(det)
    if finite.sum() < 100:
        return np.nan
    d, tt = det[finite], t[finite]
    phase = ((tt - t[0]) % period) / period
    hd = duration / period / 2.0
    in_tr = (phase < hd) | (phase > 1.0 - hd)
    out_tr = ~in_tr
    if in_tr.sum() < 3 or out_tr.sum() < 50:
        return np.nan
    return float(np.median(d[out_tr]) - np.median(d[in_tr]))


def gpu_detrend(t_v, f_v, asym=1.5, cval=5.0, n_iter=5):
    cadence = np.median(np.diff(t_v))
    W = int(round(0.5 / cadence)) | 1
    L = len(f_v)
    if L < W:
        return np.full(L, np.nan)
    ft = torch.tensor(f_v, dtype=torch.float32, device=device).unsqueeze(0)
    vt = torch.ones(1, L, dtype=torch.bool, device=device)
    st = torch.zeros(1, L, dtype=torch.int32, device=device)
    loc = kern.umi_detrend_direct(ft, vt, st, W, cval, asym, n_iter, 50)
    N_pos = L - W + 1
    offset = W // 2
    trend = np.full(L, np.nan)
    trend[offset:offset + N_pos] = loc[0].cpu().numpy()
    return np.where((trend > 0) & np.isfinite(trend), f_v / trend, np.nan)


def run_one_config(stars, period, duration, depth):
    """Run injection recovery for one (period, duration, depth) config."""
    res_w, res_u = [], []
    for t_v, f_v in stars:
        # Check if period makes sense for this star's time span
        span = t_v[-1] - t_v[0]
        if period > span * 0.8:
            continue  # skip configs where we'd get <2 transits

        phase = ((t_v - t_v[0]) % period) / period
        hd = (duration / 24.0) / period / 2.0
        if hd >= 0.5:
            continue  # duration longer than half the period
        in_tr = (phase < hd) | (phase > 1.0 - hd)
        if in_tr.sum() < 3:
            continue

        f_inj = f_v.copy()
        f_inj[in_tr] -= depth * np.median(f_v)

        # Wotan
        try:
            flat_w, _ = wotan.flatten(t_v, f_inj, method="biweight",
                                       window_length=0.5, return_trend=True)
            r = recover_depth(flat_w, t_v, period, duration / 24.0)
            if np.isfinite(r) and r > 0:
                res_w.append(r)
        except Exception:
            pass

        # UMI
        try:
            det_u = gpu_detrend(t_v, f_inj)
            torch.cuda.synchronize()
            r = recover_depth(det_u, t_v, period, duration / 24.0)
            if np.isfinite(r) and r > 0:
                res_u.append(r)
        except Exception:
            pass

    w_med = abs(np.median(res_w) - depth) / depth * 100 if len(res_w) > 10 else np.nan
    u_med = abs(np.median(res_u) - depth) / depth * 100 if len(res_u) > 10 else np.nan
    return w_med, u_med, len(res_w), len(res_u)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-stars", type=int, default=500)
    args = parser.parse_args()

    data_dir = Path("C:/Users/USER/Desktop/Website/trips-section/Celix/data/fits/sector_6")
    print(f"Loading {args.n_stars} stars...")
    stars = load_stars(data_dir, args.n_stars)
    print(f"Loaded {len(stars)} valid stars")

    periods = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 27]       # days
    durations = [1, 3, 5]                                     # hours
    depths = [0.001, 0.005, 0.05]                             # shallow, mid, deep

    total_configs = len(periods) * len(durations) * len(depths)
    print(f"\nGrid: {len(periods)} periods x {len(durations)} durations x {len(depths)} depths = {total_configs} configs")
    print(f"Each config tests {len(stars)} stars with wotan + UMI")

    results = {}
    umi_wins = 0
    wotan_wins = 0
    ties = 0
    done = 0
    t_start = time.perf_counter()

    for period in periods:
        for duration in durations:
            # Skip impossible configs (duration > period/2)
            if duration / 24.0 > period / 2:
                continue

            for depth in depths:
                w_err, u_err, n_w, n_u = run_one_config(stars, period, duration, depth)
                done += 1

                key = f"p{period}_d{duration}h_dep{depth}"
                results[key] = {
                    "period_days": period,
                    "duration_hours": duration,
                    "depth": depth,
                    "wotan_median_error_pct": round(w_err, 1) if np.isfinite(w_err) else None,
                    "umi_median_error_pct": round(u_err, 1) if np.isfinite(u_err) else None,
                    "n_wotan": n_w,
                    "n_umi": n_u,
                }

                if np.isfinite(w_err) and np.isfinite(u_err):
                    if u_err < w_err:
                        umi_wins += 1
                    elif w_err < u_err:
                        wotan_wins += 1
                    else:
                        ties += 1

                elapsed = time.perf_counter() - t_start
                eta = elapsed / done * (total_configs - done) if done > 0 else 0
                if done % 10 == 0 or done == total_configs:
                    print(f"  [{done}/{total_configs}] {elapsed:.0f}s elapsed, ~{eta/60:.0f}min remaining", flush=True)

    # Summary by depth
    print(f"\n{'='*60}")
    print(f"SUMMARY BY DEPTH (median error across all periods/durations)")
    print(f"{'='*60}")
    print(f"{'Depth':>8}  {'wotan':>7}  {'UMI':>7}  {'configs':>8}")
    print("-" * 35)
    for depth in depths:
        w_errs = [r["wotan_median_error_pct"] for r in results.values()
                  if r["depth"] == depth and r["wotan_median_error_pct"] is not None]
        u_errs = [r["umi_median_error_pct"] for r in results.values()
                  if r["depth"] == depth and r["umi_median_error_pct"] is not None]
        w_med = np.median(w_errs) if w_errs else np.nan
        u_med = np.median(u_errs) if u_errs else np.nan
        print(f"{depth:>8.4f}  {w_med:>5.1f}%  {u_med:>5.1f}%  {len(w_errs):>8}")

    print(f"\nUMI wins: {umi_wins}, wotan wins: {wotan_wins}, ties: {ties}")
    total_compared = umi_wins + wotan_wins + ties
    if total_compared > 0:
        print(f"UMI win rate: {umi_wins/total_compared*100:.0f}%")

    # Save
    doc = {
        "title": "Injection Recovery Grid",
        "date": datetime.now().isoformat(),
        "config": {
            "n_stars": len(stars),
            "periods_days": periods,
            "durations_hours": durations,
            "depths": depths,
            "total_configs": total_configs,
            "umi_asymmetry": 1.5,
        },
        "summary": {
            "umi_wins": umi_wins,
            "wotan_wins": wotan_wins,
            "ties": ties,
            "umi_win_rate": round(umi_wins / max(total_compared, 1), 2),
        },
        "per_config": results,
    }
    out = RESULTS_DIR / "injection_grid.json"
    with open(out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"\nSaved to {out}")
