"""P2: Re-run asymmetry validation and multi-sector with 2000 stars.

Saves results to results/asymmetry_validation_2k.json and
results/multisector_validation_2k.json.
"""
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wotan
from astropy.io import fits

RESULTS_DIR = Path(__file__).parent.parent / "results"
device = torch.device("cuda")

from torchflat._kernel_loader import _get_umi_kernel
kern = _get_umi_kernel()
assert kern is not None

N_STARS = 2000

sectors = {
    6: Path("C:/Users/USER/Desktop/Website/trips-section/Celix/data/fits/sector_6"),
    7: Path("C:/Users/USER/Desktop/Website/trips-section/Celix/data/fits/sector_7"),
    12: Path("C:/Users/USER/Desktop/Website/trips-section/Celix/data/fits/sector_12"),
}


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


def extract_tic_id(filename):
    m = re.match(r"tic(\d+)_s\d+\.fits", filename)
    return int(m.group(1)) if m else None


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


def gpu_detrend(t_v, f_v, asym=1.5, cval=5.0, n_iter=5):
    cadence = np.median(np.diff(t_v))
    W = int(round(0.5 / cadence)) | 1
    L = len(f_v)
    if L < W: return np.full(L, np.nan)
    ft = torch.tensor(f_v, dtype=torch.float32, device=device).unsqueeze(0)
    vt = torch.ones(1, L, dtype=torch.bool, device=device)
    st = torch.zeros(1, L, dtype=torch.int32, device=device)
    loc = kern.umi_detrend_direct(ft, vt, st, W, cval, asym, n_iter, 50)
    N_pos = L - W + 1; offset = W // 2
    trend = np.full(L, np.nan)
    trend[offset:offset+N_pos] = loc[0].cpu().numpy()
    return np.where((trend > 0) & np.isfinite(trend), f_v / trend, np.nan)


def run_injection(stars, depth, period, duration, asym=1.5):
    res_w, res_u = [], []
    for t_v, f_v in stars:
        phase = ((t_v - t_v[0]) % period) / period
        hd = duration / period / 2.0
        in_tr = (phase < hd) | (phase > 1.0 - hd)
        fi = f_v.copy(); fi[in_tr] -= depth * np.median(f_v)

        try:
            fw, _ = wotan.flatten(t_v, fi, method="biweight", window_length=0.5, return_trend=True)
            r = recover_depth(fw, t_v, period, duration)
            if np.isfinite(r) and r > 0: res_w.append(r)
        except: pass

        try:
            det = gpu_detrend(t_v, fi, asym=asym)
            torch.cuda.synchronize()
            r = recover_depth(det, t_v, period, duration)
            if np.isfinite(r) and r > 0: res_u.append(r)
        except: pass

    w = abs(np.median(res_w) - depth) / depth * 100 if len(res_w) > 10 else np.nan
    u = abs(np.median(res_u) - depth) / depth * 100 if len(res_u) > 10 else np.nan
    return w, u, len(res_w), len(res_u)


period = 3.0
duration = 3.0 / 24.0
depths = [0.001, 0.003, 0.005, 0.01, 0.05]

# ============================================================
# PART 1: Asymmetry validation with 2000 stars
# ============================================================
print("=" * 60)
print(f"ASYMMETRY VALIDATION ({N_STARS} stars, train/test split)")
print("=" * 60)

data_dir = sectors[6]
all_files = sorted(data_dir.glob("*.fits"))
train_files = [f for f in all_files if extract_tic_id(f.name) and extract_tic_id(f.name) % 2 == 1]
test_files = [f for f in all_files if extract_tic_id(f.name) and extract_tic_id(f.name) % 2 == 0]
print(f"Train files: {len(train_files)}, Test files: {len(test_files)}")

train_stars = load_stars(Path("dummy"), 0)  # empty
# Load from train files directly
train_stars = []
for f in train_files:
    if len(train_stars) >= N_STARS: break
    try:
        with fits.open(f) as h:
            d = h[1].data
            t = np.array(d["TIME"], dtype=np.float64)
            fl = np.array(d["PDCSAP_FLUX"], dtype=np.float64)
            q = np.array(d["QUALITY"], dtype=np.int32)
            v = ((q & 3455) == 0) & np.isfinite(fl) & np.isfinite(t)
            if v.sum() > 500: train_stars.append((t[v], fl[v]))
    except: continue

test_stars = []
for f in test_files:
    if len(test_stars) >= N_STARS: break
    try:
        with fits.open(f) as h:
            d = h[1].data
            t = np.array(d["TIME"], dtype=np.float64)
            fl = np.array(d["PDCSAP_FLUX"], dtype=np.float64)
            q = np.array(d["QUALITY"], dtype=np.int32)
            v = ((q & 3455) == 0) & np.isfinite(fl) & np.isfinite(t)
            if v.sum() > 500: test_stars.append((t[v], fl[v]))
    except: continue

print(f"Loaded {len(train_stars)} train, {len(test_stars)} test stars")

# Train: test asymmetry values
asymmetries = [1.0, 1.2, 1.3, 1.5, 1.7, 2.0]
train_results = {}
t0 = time.perf_counter()
for asym in asymmetries:
    train_results[asym] = {}
    for depth in depths:
        _, u_err, _, n = run_injection(train_stars, depth, period, duration, asym=asym)
        train_results[asym][depth] = u_err
    print(f"  asym={asym}: {[f'{train_results[asym][d]:.1f}%' for d in depths]}", flush=True)
train_time = time.perf_counter() - t0
print(f"Train phase: {train_time:.0f}s")

# Find best
avg_errors = {a: np.nanmean([train_results[a][d] for d in depths]) for a in asymmetries}
best_asym = min(avg_errors, key=avg_errors.get)
print(f"Optimal: {best_asym}")

# Test
print(f"\nTest phase ({len(test_stars)} stars)...")
test_results = {}
for depth in depths:
    w, u, nw, nu = run_injection(test_stars, depth, period, duration, asym=best_asym)
    test_results[depth] = {"wotan": w, "umi": u}
    print(f"  depth={depth}: wotan={w:.1f}%, UMI={u:.1f}%", flush=True)

# Save
asym_doc = {
    "title": "Asymmetry Validation (2000 stars)",
    "date": datetime.now().isoformat(),
    "n_train": len(train_stars), "n_test": len(test_stars),
    "optimal_asymmetry": best_asym,
    "train": {str(a): {str(d): train_results[a][d] for d in depths} for a in asymmetries},
    "test": {str(d): test_results[d] for d in depths},
}
with open(RESULTS_DIR / "asymmetry_validation_2k.json", "w") as f:
    json.dump(asym_doc, f, indent=2)

# ============================================================
# PART 2: Multi-sector validation with 2000 stars
# ============================================================
print(f"\n{'='*60}")
print(f"MULTI-SECTOR VALIDATION ({N_STARS} stars per sector)")
print(f"{'='*60}")

sector_results = {}
for sec_num, sec_dir in sectors.items():
    print(f"\nSector {sec_num}...")
    stars = load_stars(sec_dir, N_STARS)
    print(f"  Loaded {len(stars)} stars")

    sec_data = {}
    t0 = time.perf_counter()
    for depth in depths:
        w, u, nw, nu = run_injection(stars, depth, period, duration, asym=best_asym)
        sec_data[depth] = {"wotan": w, "umi": u}
        winner = "UMI" if u < w else "wotan" if w < u else "tie"
        print(f"  depth={depth}: wotan={w:.1f}%, UMI={u:.1f}% ({winner})", flush=True)
    print(f"  Completed in {time.perf_counter()-t0:.0f}s")
    sector_results[sec_num] = sec_data

# Win count
umi_wins = sum(1 for s in sector_results.values() for d in depths
               if s[d]["umi"] < s[d]["wotan"] and np.isfinite(s[d]["umi"]))
total = sum(1 for s in sector_results.values() for d in depths
            if np.isfinite(s[d]["umi"]) and np.isfinite(s[d]["wotan"]))
print(f"\nUMI wins: {umi_wins}/{total} ({umi_wins/max(total,1)*100:.0f}%)")

# Save
multi_doc = {
    "title": "Multi-Sector Validation (2000 stars)",
    "date": datetime.now().isoformat(),
    "n_stars_per_sector": N_STARS,
    "optimal_asymmetry": best_asym,
    "sectors": {str(s): {str(d): v for d, v in data.items()} for s, data in sector_results.items()},
    "umi_wins": umi_wins, "total": total,
}
with open(RESULTS_DIR / "multisector_validation_2k.json", "w") as f:
    json.dump(multi_doc, f, indent=2)

print(f"\nResults saved to results/asymmetry_validation_2k.json and multisector_validation_2k.json")
