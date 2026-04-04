"""Compare UMI against all detrending methods, 1000 stars.

Saves results to results/method_comparison_1k.json.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wotan
from astropy.io import fits
from scipy.signal import savgol_filter

RESULTS_DIR = Path(__file__).parent.parent / "results"
device = torch.device("cuda")

from torchflat._kernel_loader import _get_umi_kernel
kern = _get_umi_kernel()
assert kern is not None

data_dir = Path("C:/Users/USER/Desktop/Website/trips-section/Celix/data/fits/sector_6")
stars = []
print("Loading 1000 stars...")
for f in sorted(data_dir.glob("*.fits"))[:1100]:
    try:
        with fits.open(f) as h:
            d = h[1].data
            t = np.array(d["TIME"], dtype=np.float64)
            fl = np.array(d["PDCSAP_FLUX"], dtype=np.float64)
            q = np.array(d["QUALITY"], dtype=np.int32)
            v = ((q & 3455) == 0) & np.isfinite(fl) & np.isfinite(t)
            if v.sum() > 500:
                stars.append((t[v], fl[v].copy()))
    except:
        continue
    if len(stars) >= 1000:
        break
print(f"Loaded {len(stars)} stars")


def rd(det, t, p, dur):
    fi = np.isfinite(det)
    if fi.sum() < 100:
        return np.nan
    d, tt = det[fi], t[fi]
    ph = ((tt - t[0]) % p) / p
    hd = dur / p / 2
    itr = (ph < hd) | (ph > 1 - hd)
    otr = ~itr
    if itr.sum() < 3 or otr.sum() < 50:
        return np.nan
    return float(np.median(d[otr]) - np.median(d[itr]))


def gpu_umi(t_v, f_v):
    cad = np.median(np.diff(t_v))
    W = int(round(0.5 / cad)) | 1
    L = len(f_v)
    if L < W:
        return np.full(L, np.nan)
    ft = torch.tensor(f_v, dtype=torch.float32, device=device).unsqueeze(0)
    vt = torch.ones(1, L, dtype=torch.bool, device=device)
    st = torch.zeros(1, L, dtype=torch.int32, device=device)
    loc = kern.umi_detrend_direct(ft, vt, st, W, 5.0, 2.0, 5, 50)
    Np = L - W + 1
    off = W // 2
    trend = np.full(L, np.nan)
    trend[off:off + Np] = loc[0].cpu().numpy()
    return np.where((trend > 0) & np.isfinite(trend), f_v / trend, np.nan)


def savgol_detrend(t_v, f_v):
    cad = np.median(np.diff(t_v))
    W = int(round(0.5 / cad)) | 1
    if W <= 2:
        return np.full(len(f_v), np.nan)
    # Iterative savgol with 3-sigma clipping
    mask = np.ones(len(f_v), dtype=bool)
    trend = None
    for _ in range(3):
        f_filled = f_v.copy()
        if not mask.all():
            good = np.where(mask)[0]
            bad = np.where(~mask)[0]
            if len(good) > 2:
                f_filled[bad] = np.interp(bad, good, f_v[good])
        trend = savgol_filter(f_filled, W, 2)
        res = f_v - trend
        med_r = np.median(res[mask])
        mad_r = np.median(np.abs(res[mask] - med_r))
        if mad_r < 1e-10:
            break
        threshold = 3.0 * mad_r / 0.6745
        mask = np.abs(res - med_r) < threshold
        if mask.sum() < 100:
            break
    if trend is None:
        return np.full(len(f_v), np.nan)
    return np.where(trend > 0, f_v / trend, np.nan)


period, duration = 3.0, 3.0 / 24.0
depths = [0.001, 0.005, 0.01, 0.05]

# Methods: name, function
wotan_methods = [
    ("biweight", {"method": "biweight", "window_length": 0.5}),
    ("median", {"method": "median", "window_length": 0.5}),
    ("mean", {"method": "mean", "window_length": 0.5}),
    ("trim_mean", {"method": "trim_mean", "window_length": 0.5}),
    ("welsch", {"method": "welsch", "window_length": 0.5}),
    ("huber", {"method": "huber", "window_length": 0.5}),
    ("lowess", {"method": "lowess", "window_length": 0.5}),
]

all_methods = ["UMI"] + [name for name, _ in wotan_methods] + ["savgol"]

print(f"\nperiod={period}d, duration={duration*24:.0f}h, {len(stars)} stars")
print(f"Methods: {', '.join(all_methods)}")

results = {}
header = f"{'Depth':>8}"
for m in all_methods:
    header += f"  {m:>10}"
print(f"\n{header}")
print("-" * len(header))

for depth in depths:
    row_data = {m: [] for m in all_methods}

    for si, (t_v, f_v) in enumerate(stars):
        fi = f_v.copy()
        ph = ((t_v - t_v[0]) % period) / period
        hd = duration / period / 2
        itr = (ph < hd) | (ph > 1 - hd)
        fi[itr] -= depth * np.median(f_v)

        # UMI
        det = gpu_umi(t_v, fi)
        torch.cuda.synchronize()
        r = rd(det, t_v, period, duration)
        if np.isfinite(r) and r > 0:
            row_data["UMI"].append(r)

        # Wotan methods
        for name, kwargs in wotan_methods:
            try:
                flat, _ = wotan.flatten(t_v, fi, return_trend=True, **kwargs)
                r = rd(flat, t_v, period, duration)
                if np.isfinite(r) and r > 0:
                    row_data[name].append(r)
            except Exception:
                pass

        # Savgol
        det = savgol_detrend(t_v, fi)
        r = rd(det, t_v, period, duration)
        if np.isfinite(r) and r > 0:
            row_data["savgol"].append(r)

        if (si + 1) % 200 == 0:
            print(f"  ... {si+1}/{len(stars)} stars", flush=True)

    row = f"{depth:>8.4f}"
    for m in all_methods:
        recs = row_data[m]
        if len(recs) > 10:
            err = abs(np.median(recs) - depth) / depth * 100
            row += f"  {err:>8.1f}%"
            if depth not in results:
                results[depth] = {}
            results[depth][m] = round(err, 1)
        else:
            row += f"  {'N/A':>9}"
    print(row, flush=True)

# Speed comparison
print(f"\n=== SPEED (per star, avg of 20 runs) ===")
t_v, f_v = stars[0]
speed_results = {}

# UMI
gpu_umi(t_v, f_v); torch.cuda.synchronize()
ts = []
for _ in range(20):
    t0 = time.perf_counter(); gpu_umi(t_v, f_v); torch.cuda.synchronize()
    ts.append(time.perf_counter() - t0)
ms = np.mean(ts) * 1000
print(f"  {'UMI (GPU)':>12}: {ms:.1f}ms")
speed_results["UMI"] = round(ms, 1)

for name, kwargs in wotan_methods:
    try:
        wotan.flatten(t_v, f_v, return_trend=True, **kwargs)
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            wotan.flatten(t_v, f_v, return_trend=True, **kwargs)
            ts.append(time.perf_counter() - t0)
        ms = np.mean(ts) * 1000
        print(f"  {name:>12}: {ms:.0f}ms")
        speed_results[name] = round(ms)
    except Exception as e:
        print(f"  {name:>12}: FAILED ({type(e).__name__})")

# Savgol
ts = []
for _ in range(20):
    t0 = time.perf_counter(); savgol_detrend(t_v, f_v); ts.append(time.perf_counter() - t0)
ms = np.mean(ts) * 1000
print(f"  {'savgol':>12}: {ms:.1f}ms")
speed_results["savgol"] = round(ms, 1)

# Save
doc = {
    "title": "Method Comparison (1000 stars)",
    "date": datetime.now().isoformat(),
    "n_stars": len(stars),
    "period_days": period,
    "duration_hours": duration * 24,
    "accuracy": {str(d): results.get(d, {}) for d in depths},
    "speed_ms_per_star": speed_results,
}
out = RESULTS_DIR / "method_comparison_1k.json"
with open(out, "w") as f:
    json.dump(doc, f, indent=2)
print(f"\nSaved to {out}")

# Uninstall statsmodels after
print("\nTo uninstall statsmodels: pip uninstall statsmodels -y")
