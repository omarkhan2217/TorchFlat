"""Known planet recovery: detrend with TorchFlat UMI and wotan, compare to published depths.

Cross-references TOI catalog against FITS files in sectors 6, 7, 12.
Saves results to results/known_planet_recovery.json.

Usage:
    python benchmarks/validate_known_planets.py
"""
import csv
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
toi_file = Path("C:/Users/USER/Desktop/Website/trips-section/Celix/data/catalogs/toi_catalog.csv")
data_dirs = {
    6: Path("C:/Users/USER/Desktop/Website/trips-section/Celix/data/fits/sector_6"),
    7: Path("C:/Users/USER/Desktop/Website/trips-section/Celix/data/fits/sector_7"),
    12: Path("C:/Users/USER/Desktop/Website/trips-section/Celix/data/fits/sector_12"),
}

from torchflat._kernel_loader import _get_umi_kernel
kern = _get_umi_kernel()
assert kern is not None, "UMI kernel required"


def find_planets():
    """Find confirmed planets in our sectors with FITS files."""
    planets = []
    with open(toi_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            disp = row.get("TESS Disposition", "")
            tfop = row.get("TFOPWG Disposition", "")
            if disp != "KP" and tfop != "KP":
                continue
            sectors = row.get("Sectors", "")
            if not sectors:
                continue
            sec_list = [int(s.strip()) for s in sectors.split(",") if s.strip().isdigit()]
            in_our = set(sec_list) & {6, 7, 12}
            if not in_our:
                continue
            tic = row.get("TIC ID", "")
            try:
                d_ppm = float(row.get("Depth (ppm)", ""))
                p = float(row.get("Period (days)", ""))
                dur = float(row.get("Duration (hours)", ""))
                epoch = float(row.get("Epoch (BJD)", ""))
            except (ValueError, TypeError):
                continue

            for s in sorted(in_our):
                fits_path = data_dirs[s] / f"tic{tic}_s{s:04d}.fits"
                if fits_path.exists():
                    planets.append({
                        "tic": tic,
                        "sector": s,
                        "name": row.get("Planet Name", "") or row.get("TOI", ""),
                        "published_depth_ppm": d_ppm,
                        "period": p,
                        "duration": dur,
                        "epoch_bjd": epoch,
                        "fits": str(fits_path),
                    })
                    break
    return planets


TESS_BJD_OFFSET = 2457000.0  # TESS BTJD = BJD - 2457000


def recover_depth(det, t, period, duration, epoch_bjd):
    """Recover transit depth using published ephemeris.

    t is in BTJD (BJD - 2457000), epoch_bjd is in full BJD.
    """
    finite = np.isfinite(det)
    if finite.sum() < 100:
        return np.nan
    d, tt = det[finite], t[finite]
    # Convert epoch to BTJD
    epoch = epoch_bjd - TESS_BJD_OFFSET
    # Phase-fold
    phase = ((tt - epoch) % period) / period
    # Wrap to [-0.5, 0.5]
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    hd = (duration / 24.0) / period / 2.0
    in_tr = np.abs(phase) < hd
    out_tr = ~in_tr
    if in_tr.sum() < 3 or out_tr.sum() < 50:
        return np.nan
    return float(np.median(d[out_tr]) - np.median(d[in_tr]))


def gpu_detrend(t_v, f_v, asym=1.5, cval=5.0, n_iter=10):
    """Detrend with UMI kernel."""
    cadence = np.median(np.diff(t_v))
    W = int(round(0.5 / cadence)) | 1
    L = len(f_v)
    if L < W:
        return np.full(L, np.nan)
    ft = torch.tensor(f_v, dtype=torch.float32, device=device).unsqueeze(0)
    vt = torch.ones(1, L, dtype=torch.bool, device=device)
    fw = ft.unfold(1, W, 1).contiguous()
    vw = vt.unfold(1, W, 1).contiguous()
    loc = kern.umi_detrend(fw, vw, cval, asym, n_iter, 50)
    N_pos = L - W + 1
    offset = W // 2
    trend = np.full(L, np.nan)
    trend[offset:offset + N_pos] = loc[0].cpu().numpy()
    return np.where((trend > 0) & np.isfinite(trend), f_v / trend, np.nan)


# ==================================================================
# Find planets
# ==================================================================
planets = find_planets()
print(f"Found {len(planets)} confirmed planets with FITS files")

# ==================================================================
# Process each planet
# ==================================================================
results = []
print(f"\n{'#':>3} {'Name':>15} {'Sec':>4} {'Published':>10} {'wotan':>10} {'UMI':>10} {'w_err%':>8} {'u_err%':>8}")
print("-" * 78)

for i, pl in enumerate(sorted(planets, key=lambda x: x["published_depth_ppm"])):
    # Load data
    try:
        with fits.open(pl["fits"]) as h:
            d = h[1].data
            t = np.array(d["TIME"], dtype=np.float64)
            fl = np.array(d["PDCSAP_FLUX"], dtype=np.float64)
            q = np.array(d["QUALITY"], dtype=np.int32)
    except Exception as e:
        print(f"  SKIP {pl['name']}: {e}")
        continue

    v = ((q & 3455) == 0) & np.isfinite(fl) & np.isfinite(t)
    if v.sum() < 500:
        continue
    t_v, f_v = t[v], fl[v]

    pub_depth = pl["published_depth_ppm"] / 1e6  # convert ppm to fraction

    # Wotan detrend
    try:
        flat_w, _ = wotan.flatten(t_v, f_v, method="biweight", window_length=0.5, return_trend=True)
        depth_w = recover_depth(flat_w, t_v, pl["period"], pl["duration"], pl["epoch_bjd"])
    except Exception:
        depth_w = np.nan

    # UMI detrend
    try:
        det_u = gpu_detrend(t_v, f_v)
        torch.cuda.synchronize()
        depth_u = recover_depth(det_u, t_v, pl["period"], pl["duration"], pl["epoch_bjd"])
    except Exception:
        depth_u = np.nan

    # Depths are fractional (flux/trend ~ 1.0, so depth = median_out - median_in)
    # Convert to ppm for comparison with published values
    pub_ppm = pl["published_depth_ppm"]
    w_ppm = depth_w * 1e6 if np.isfinite(depth_w) else np.nan
    u_ppm = depth_u * 1e6 if np.isfinite(depth_u) else np.nan

    # Error: how close is recovered depth to published depth (in % of published)
    w_err = abs(w_ppm - pub_ppm) / pub_ppm * 100 if np.isfinite(w_ppm) and pub_ppm > 0 else np.nan
    u_err = abs(u_ppm - pub_ppm) / pub_ppm * 100 if np.isfinite(u_ppm) and pub_ppm > 0 else np.nan

    results.append({
        "tic": pl["tic"],
        "name": pl["name"],
        "sector": pl["sector"],
        "published_depth_ppm": pub_ppm,
        "period_days": pl["period"],
        "duration_hours": pl["duration"],
        "wotan_depth_ppm": round(w_ppm, 1) if np.isfinite(w_ppm) else None,
        "umi_depth_ppm": round(u_ppm, 1) if np.isfinite(u_ppm) else None,
        "wotan_error_pct": round(w_err, 1) if np.isfinite(w_err) else None,
        "umi_error_pct": round(u_err, 1) if np.isfinite(u_err) else None,
    })

    def fmt_ppm(v):
        return f"{v:>8.0f}" if np.isfinite(v) else f"{'N/A':>8}"
    def fmt_err(v):
        return f"{v:>6.1f}%" if np.isfinite(v) else f"{'N/A':>7}"

    print(f"{i+1:>3} {pl['name']:>15} {pl['sector']:>4} {pub_ppm:>8.0f}pp {fmt_ppm(w_ppm)}pp {fmt_ppm(u_ppm)}pp {fmt_err(w_err)} {fmt_err(u_err)}")

# ==================================================================
# Summary statistics
# ==================================================================
valid = [r for r in results if r["wotan_error_pct"] is not None and r["umi_error_pct"] is not None]
w_errs = [r["wotan_error_pct"] for r in valid]
u_errs = [r["umi_error_pct"] for r in valid]

print(f"\n{'='*70}")
print(f"SUMMARY ({len(valid)} planets with both methods)")
print(f"{'='*70}")
print(f"  wotan median error:     {np.median(w_errs):.1f}%")
print(f"  UMI median error:       {np.median(u_errs):.1f}%")
print(f"  wotan mean error:       {np.mean(w_errs):.1f}%")
print(f"  UMI mean error:         {np.mean(u_errs):.1f}%")

# By depth category
for label, lo, hi in [("Shallow (<1000 ppm)", 0, 1000), ("Mid (1000-10000)", 1000, 10000), ("Deep (>10000)", 10000, 1e9)]:
    sub = [r for r in valid if lo <= r["published_depth_ppm"] < hi]
    if sub:
        sw = [r["wotan_error_pct"] for r in sub]
        su = [r["umi_error_pct"] for r in sub]
        print(f"\n  {label} ({len(sub)} planets):")
        print(f"    wotan: median={np.median(sw):.1f}%  mean={np.mean(sw):.1f}%")
        print(f"    UMI:   median={np.median(su):.1f}%  mean={np.mean(su):.1f}%")

umi_wins = sum(1 for r in valid if r["umi_error_pct"] < r["wotan_error_pct"])
print(f"\n  UMI better on {umi_wins}/{len(valid)} planets ({umi_wins/len(valid)*100:.0f}%)")

# ==================================================================
# Save results
# ==================================================================
result_doc = {
    "title": "Known Planet Transit Depth Recovery",
    "date": datetime.now().isoformat(),
    "data": {
        "toi_catalog": str(toi_file),
        "sectors": [6, 7, 12],
        "n_planets_found": len(planets),
        "n_planets_valid": len(valid),
    },
    "config": {
        "umi_asymmetry": 1.5,
        "umi_cval": 5.0,
        "umi_n_iter": 10,
        "wotan_method": "biweight",
        "wotan_window_length": 0.5,
    },
    "summary": {
        "wotan_median_error_pct": round(np.median(w_errs), 1),
        "umi_median_error_pct": round(np.median(u_errs), 1),
        "wotan_mean_error_pct": round(np.mean(w_errs), 1),
        "umi_mean_error_pct": round(np.mean(u_errs), 1),
        "umi_wins_count": umi_wins,
        "umi_wins_fraction": round(umi_wins / len(valid), 2),
    },
    "per_planet": results,
}

out_file = RESULTS_DIR / "known_planet_recovery.json"
with open(out_file, "w") as f:
    json.dump(result_doc, f, indent=2)
print(f"\nResults saved to {out_file}")
