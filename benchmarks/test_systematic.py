"""T1: Systematic error analysis on different stellar types.

Tests UMI on eclipsing binaries, variable stars, and normal stars.
Checks if asymmetric weight causes problems on non-transit signals.
Saves results to results/systematic_analysis.json.
"""
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

RESULTS_DIR = Path(__file__).parent.parent / "results"
device = torch.device("cuda")

from torchflat._kernel_loader import _get_umi_kernel
kern = _get_umi_kernel()
assert kern is not None

data_dirs = {
    6: Path("C:/Users/USER/Desktop/Website/trips-section/Celix/data/fits/sector_6"),
    7: Path("C:/Users/USER/Desktop/Website/trips-section/Celix/data/fits/sector_7"),
    12: Path("C:/Users/USER/Desktop/Website/trips-section/Celix/data/fits/sector_12"),
}
toi_file = Path("C:/Users/USER/Desktop/Website/trips-section/Celix/data/catalogs/toi_catalog.csv")
eb_file = Path("C:/Users/USER/Desktop/Website/trips-section/Celix/data/catalogs/eb_catalog.csv")


def load_star(fits_path):
    with fits.open(fits_path) as h:
        d = h[1].data
        t = np.array(d["TIME"], dtype=np.float64)
        fl = np.array(d["PDCSAP_FLUX"], dtype=np.float64)
        q = np.array(d["QUALITY"], dtype=np.int32)
    v = ((q & 3455) == 0) & np.isfinite(fl) & np.isfinite(t)
    if v.sum() < 500:
        return None, None
    return t[v], fl[v]


def gpu_umi(t_v, f_v, asym=2.0):
    cad = np.median(np.diff(t_v))
    W = int(round(0.5 / cad)) | 1
    L = len(f_v)
    if L < W:
        return np.full(L, np.nan), np.full(L, np.nan)
    ft = torch.tensor(f_v, dtype=torch.float32, device=device).unsqueeze(0)
    vt = torch.ones(1, L, dtype=torch.bool, device=device)
    st = torch.zeros(1, L, dtype=torch.int32, device=device)
    loc = kern.umi_detrend_direct(ft, vt, st, W, 5.0, asym, 5, 50)
    Np = L - W + 1
    off = W // 2
    trend = np.full(L, np.nan)
    trend[off:off + Np] = loc[0].cpu().numpy()
    det = np.where((trend > 0) & np.isfinite(trend), f_v / trend, np.nan)
    return det, trend


def analyze_detrended(det, f_v):
    """Compute stats on detrended flux."""
    finite = np.isfinite(det)
    if finite.sum() < 100:
        return None
    d = det[finite]
    return {
        "median": float(np.median(d)),
        "std": float(np.std(d)),
        "min": float(np.min(d)),
        "max": float(np.max(d)),
        "mad": float(np.median(np.abs(d - np.median(d)))),
        "skew": float(np.mean(((d - np.mean(d)) / np.std(d)) ** 3)) if np.std(d) > 0 else 0,
        "n_below_3sigma": int(np.sum(d < np.median(d) - 3 * np.median(np.abs(d - np.median(d))) / 0.6745)),
        "n_above_3sigma": int(np.sum(d > np.median(d) + 3 * np.median(np.abs(d - np.median(d))) / 0.6745)),
        "flux_range_pct": float((np.max(f_v) - np.min(f_v)) / np.median(f_v) * 100),
    }


# ==================================================================
# Find EBs in our sectors
# ==================================================================
print("Finding eclipsing binaries...")
eb_tics = set()
with open(eb_file, "r", encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f)
    for row in reader:
        eb_tics.add(row.get("tess_id", "").strip())

# Also from TOI catalog
with open(toi_file, "r", encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f)
    for row in reader:
        disp = row.get("TESS Disposition", "")
        tfop = row.get("TFOPWG Disposition", "")
        if "EB" in disp or "EB" in tfop:
            eb_tics.add(row.get("TIC ID", "").strip())

print(f"  Known EBs: {len(eb_tics)} TIC IDs")

# ==================================================================
# Classify stars in sector 6
# ==================================================================
print("\nClassifying sector 6 stars...")
normal_stars = []
eb_stars = []
variable_stars = []

for f in sorted(data_dirs[6].glob("*.fits"))[:2000]:
    tic = f.stem.replace("tic", "").split("_")[0]
    t_v, f_v = load_star(f)
    if t_v is None:
        continue

    # Check variability: std/median > 1%
    rel_std = np.std(f_v) / np.median(f_v)

    if tic in eb_tics:
        eb_stars.append((t_v, f_v, tic, rel_std))
    elif rel_std > 0.01:
        variable_stars.append((t_v, f_v, tic, rel_std))
    else:
        normal_stars.append((t_v, f_v, tic, rel_std))

    if len(normal_stars) >= 200 and len(eb_stars) >= 20 and len(variable_stars) >= 50:
        break

print(f"  Normal (quiet): {len(normal_stars)}")
print(f"  Eclipsing binaries: {len(eb_stars)}")
print(f"  Variable (rel_std > 1%): {len(variable_stars)}")

# ==================================================================
# Test each category
# ==================================================================
results = {}

for cat_name, cat_stars in [
    ("normal", normal_stars[:200]),
    ("eclipsing_binary", eb_stars[:50]),
    ("variable", variable_stars[:100]),
]:
    print(f"\n{'='*50}")
    print(f"{cat_name.upper()} ({len(cat_stars)} stars)")
    print(f"{'='*50}")

    stats_a10 = []  # symmetric biweight
    stats_a20 = []  # UMI asymmetric

    for t_v, f_v, tic, rel_std in cat_stars:
        det10, _ = gpu_umi(t_v, f_v, asym=1.0)
        det20, _ = gpu_umi(t_v, f_v, asym=2.0)
        torch.cuda.synchronize()

        s10 = analyze_detrended(det10, f_v)
        s20 = analyze_detrended(det20, f_v)
        if s10 and s20:
            stats_a10.append(s10)
            stats_a20.append(s20)

    if not stats_a10:
        print("  No valid stars")
        continue

    def med(stats, key):
        return np.median([s[key] for s in stats])

    print(f"  {'Metric':>20}  {'a=1.0':>10}  {'a=2.0':>10}")
    print(f"  {'-'*45}")
    for key in ["median", "std", "mad", "skew", "n_below_3sigma", "n_above_3sigma"]:
        v10 = med(stats_a10, key)
        v20 = med(stats_a20, key)
        if key == "median":
            print(f"  {key:>20}  {v10:>10.6f}  {v20:>10.6f}")
        elif key in ("n_below_3sigma", "n_above_3sigma"):
            print(f"  {key:>20}  {v10:>10.0f}  {v20:>10.0f}")
        else:
            print(f"  {key:>20}  {v10:>10.4f}  {v20:>10.4f}")

    # Bias
    bias10 = np.median([s["median"] - 1.0 for s in stats_a10]) * 1e6
    bias20 = np.median([s["median"] - 1.0 for s in stats_a20]) * 1e6
    print(f"  {'bias (ppm)':>20}  {bias10:>+10.0f}  {bias20:>+10.0f}")

    # Detrending quality: is std reduced?
    raw_std = np.median([s["flux_range_pct"] for s in stats_a10])
    det_std10 = med(stats_a10, "std") * 100
    det_std20 = med(stats_a20, "std") * 100
    print(f"  {'raw flux range %':>20}  {raw_std:>10.2f}")
    print(f"  {'detrended std %':>20}  {det_std10:>10.4f}  {det_std20:>10.4f}")

    results[cat_name] = {
        "n_stars": len(stats_a10),
        "bias_a10_ppm": round(bias10),
        "bias_a20_ppm": round(bias20),
        "median_std_a10": round(med(stats_a10, "std"), 6),
        "median_std_a20": round(med(stats_a20, "std"), 6),
        "median_skew_a10": round(med(stats_a10, "skew"), 4),
        "median_skew_a20": round(med(stats_a20, "skew"), 4),
    }

# ==================================================================
# Flare detection: inject upward spikes and check if UMI preserves them
# ==================================================================
print(f"\n{'='*50}")
print(f"FLARE PRESERVATION TEST")
print(f"{'='*50}")

flare_results = []
for t_v, f_v, tic, _ in normal_stars[:100]:
    fi = f_v.copy()
    # Inject 3 random flares: 5% upward spikes, 10 cadences each
    rng = np.random.default_rng(int(tic) % 2**31)
    for _ in range(3):
        start = rng.integers(100, len(fi) - 100)
        fi[start:start + 10] *= 1.05  # 5% flare

    det, _ = gpu_umi(t_v, fi, asym=2.0)
    torch.cuda.synchronize()
    finite = np.isfinite(det)
    if finite.sum() < 100:
        continue
    # Check if flares are preserved (should be above 1.0 in detrended)
    max_det = np.max(det[finite])
    flare_results.append(max_det)

flare_preserved = np.median(flare_results)
print(f"  Injected 5% flares on 100 stars")
print(f"  Median max detrended: {flare_preserved:.4f}")
print(f"  Flares {'PRESERVED' if flare_preserved > 1.03 else 'ABSORBED'}")
results["flare_test"] = {
    "n_stars": len(flare_results),
    "median_max_detrended": round(float(flare_preserved), 4),
    "flares_preserved": bool(flare_preserved > 1.03),
}

# Save
doc = {
    "title": "Systematic Error Analysis",
    "date": datetime.now().isoformat(),
    "categories": results,
}
out = RESULTS_DIR / "systematic_analysis.json"
with open(out, "w") as f:
    json.dump(doc, f, indent=2)
print(f"\nSaved to {out}")
