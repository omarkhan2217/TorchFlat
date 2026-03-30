"""Compare UMI against all major detrending methods via wotan."""
import numpy as np
import torch
import wotan
import time
from pathlib import Path
from astropy.io import fits
from torchflat._kernel_loader import _get_umi_kernel

device = torch.device("cuda")
kern = _get_umi_kernel()
data_dir = Path("C:/Users/USER/Desktop/Website/trips-section/Celix/data/fits/sector_6")

stars = []
for f in sorted(data_dir.glob("*.fits"))[:100]:
    with fits.open(f) as h:
        d = h[1].data
        t = np.array(d["TIME"], dtype=np.float64)
        fl = np.array(d["PDCSAP_FLUX"], dtype=np.float64)
        q = np.array(d["QUALITY"], dtype=np.int32)
        v = ((q & 3455) == 0) & np.isfinite(fl) & np.isfinite(t)
        if v.sum() > 500:
            stars.append((t[v], fl[v]))
print(f"Loaded {len(stars)} stars")


def rd(det, t, p, dur):
    finite = np.isfinite(det)
    if finite.sum() < 100: return np.nan
    d, tt = det[finite], t[finite]
    phase = ((tt - t[0]) % p) / p; hd = dur / p / 2
    in_tr = (phase < hd) | (phase > 1 - hd); out_tr = ~in_tr
    if in_tr.sum() < 3 or out_tr.sum() < 50: return np.nan
    return float(np.median(d[out_tr]) - np.median(d[in_tr]))


def gpu_umi(t_v, f_v):
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


period, duration = 3.0, 3.0 / 24.0
depths = [0.001, 0.005, 0.01, 0.05]

methods = [
    ("biweight", {"method": "biweight", "window_length": 0.5}),
    ("median", {"method": "median", "window_length": 0.5}),
    ("mean", {"method": "mean", "window_length": 0.5}),
    ("trimmed_mean", {"method": "trim_mean", "window_length": 0.5}),
    ("welsch", {"method": "welsch", "window_length": 0.5}),
    ("huber", {"method": "huber", "window_length": 0.5}),
    ("lowess", {"method": "lowess", "window_length": 0.5}),
    ("savgol", {"method": "savgol", "window_length": 0.5}),
]

print(f"\n=== DETRENDING METHOD COMPARISON (100 stars) ===")
print(f"    period={period}d, duration={duration*24:.0f}h")
print(f"    median per-star error (lower = better)")

header = f"{'Depth':>8}  {'UMI':>7}"
for name, _ in methods:
    header += f"  {name:>12}"
print(f"\n{header}")
print("-" * len(header))

for depth in depths:
    row_data = {"UMI": []}
    for name, _ in methods:
        row_data[name] = []

    for t_v, f_v in stars:
        phase = ((t_v - t_v[0]) % period) / period
        hd = duration / period / 2
        in_tr = (phase < hd) | (phase > 1 - hd)
        fi = f_v.copy()
        fi[in_tr] -= depth * np.median(f_v)

        # UMI
        det = gpu_umi(t_v, fi)
        torch.cuda.synchronize()
        r = rd(det, t_v, period, duration)
        if np.isfinite(r) and r > 0: row_data["UMI"].append(r)

        # Wotan methods
        for name, kwargs in methods:
            try:
                flat, _ = wotan.flatten(t_v, fi, return_trend=True, **kwargs)
                r = rd(flat, t_v, period, duration)
                if np.isfinite(r) and r > 0:
                    row_data[name].append(r)
            except Exception:
                pass

    row = f"{depth:>8.4f}"
    for key in ["UMI"] + [name for name, _ in methods]:
        recs = row_data[key]
        if len(recs) > 10:
            err = abs(np.median(recs) - depth) / depth * 100
            row += f"  {err:>10.1f}%"
        else:
            row += f"  {'N/A':>11}"
    print(row)

# Speed comparison
print(f"\n=== SPEED (1 star, avg of 10 runs) ===")
t_v, f_v = stars[0]

# UMI
gpu_umi(t_v, f_v); torch.cuda.synchronize()
times = []
for _ in range(10):
    t0 = time.perf_counter()
    gpu_umi(t_v, f_v); torch.cuda.synchronize()
    times.append(time.perf_counter()-t0)
print(f"  {'UMI (GPU)':>15}: {np.mean(times)*1000:.1f}ms")

for name, kwargs in methods:
    try:
        wotan.flatten(t_v, f_v, return_trend=True, **kwargs)
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            wotan.flatten(t_v, f_v, return_trend=True, **kwargs)
            times.append(time.perf_counter()-t0)
        print(f"  {name:>15}: {np.mean(times)*1000:.1f}ms")
    except Exception as e:
        print(f"  {name:>15}: FAILED ({e})")
