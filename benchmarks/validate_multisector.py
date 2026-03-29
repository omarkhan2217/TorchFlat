"""Multi-sector validation: run injection recovery on sectors 6, 7, 12.

Shows TorchFlat UMI accuracy is consistent across different spacecraft orientations.
Saves results to results/multisector_validation.json.

Usage:
    python benchmarks/validate_multisector.py
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
RESULTS_DIR.mkdir(exist_ok=True)

device = torch.device("cuda")
sectors = {
    6: Path("C:/Users/USER/Desktop/Website/trips-section/Celix/data/fits/sector_6"),
    7: Path("C:/Users/USER/Desktop/Website/trips-section/Celix/data/fits/sector_7"),
    12: Path("C:/Users/USER/Desktop/Website/trips-section/Celix/data/fits/sector_12"),
}

from torchflat._kernel_loader import _get_umi_kernel
kern = _get_umi_kernel()
assert kern is not None, "UMI kernel required"

N_STARS = 500  # per sector


def load_stars(data_dir, n):
    """Load valid flux/time from first n FITS files."""
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


def gpu_detrend(t_v, f_v, asym=1.5, cval=5.0, n_iter=10):
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


period = 3.0
duration = 3.0 / 24.0
depths = [0.001, 0.003, 0.005, 0.01, 0.05]

all_results = {}

for sec_num, data_dir in sectors.items():
    print(f"\n{'='*60}")
    print(f"SECTOR {sec_num} ({data_dir})")
    print(f"{'='*60}")

    stars = load_stars(data_dir, N_STARS)
    print(f"Loaded {len(stars)} stars")

    sec_results = {"wotan": {}, "umi": {}}

    t0 = time.perf_counter()
    for depth in depths:
        res_w, res_u = [], []
        for t_v, f_v in stars:
            phase = ((t_v - t_v[0]) % period) / period
            hd = duration / period / 2.0
            in_tr = (phase < hd) | (phase > 1.0 - hd)
            f_inj = f_v.copy()
            f_inj[in_tr] -= depth * np.median(f_v)

            # Wotan
            try:
                flat_w, _ = wotan.flatten(t_v, f_inj, method="biweight",
                                           window_length=0.5, return_trend=True)
                r = recover_depth(flat_w, t_v, period, duration)
                if np.isfinite(r) and r > 0:
                    res_w.append(r)
            except Exception:
                pass

            # UMI
            try:
                det_u = gpu_detrend(t_v, f_inj)
                torch.cuda.synchronize()
                r = recover_depth(det_u, t_v, period, duration)
                if np.isfinite(r) and r > 0:
                    res_u.append(r)
            except Exception:
                pass

        w_med = abs(np.median(res_w) - depth) / depth * 100 if res_w else np.nan
        u_med = abs(np.median(res_u) - depth) / depth * 100 if res_u else np.nan
        sec_results["wotan"][str(depth)] = round(w_med, 1)
        sec_results["umi"][str(depth)] = round(u_med, 1)

    elapsed = time.perf_counter() - t0
    print(f"Completed in {elapsed:.0f}s")

    # Print sector results
    print(f"\n{'Depth':>8}  {'wotan':>7}  {'UMI':>7}  {'Winner':>10}")
    print("-" * 38)
    for depth in depths:
        w = sec_results["wotan"][str(depth)]
        u = sec_results["umi"][str(depth)]
        winner = "UMI" if u < w else ("wotan" if w < u else "tie")
        print(f"{depth:>8.4f}  {w:>5.1f}%  {u:>5.1f}%  {winner:>10}")

    all_results[sec_num] = sec_results

# ==================================================================
# Cross-sector summary
# ==================================================================
print(f"\n{'='*60}")
print(f"CROSS-SECTOR SUMMARY")
print(f"{'='*60}")

header = f"{'Depth':>8}"
for s in sectors:
    header += f"  {'S'+str(s)+' w':>7}  {'S'+str(s)+' u':>7}"
print(header)
print("-" * len(header))

for depth in depths:
    row = f"{depth:>8.4f}"
    for s in sectors:
        w = all_results[s]["wotan"][str(depth)]
        u = all_results[s]["umi"][str(depth)]
        row += f"  {w:>5.1f}%  {u:>5.1f}%"
    print(row)

# Consistency check: UMI error std across sectors
print(f"\nConsistency (std of UMI error across sectors):")
for depth in depths:
    vals = [all_results[s]["umi"][str(depth)] for s in sectors]
    print(f"  depth={depth}: {np.std(vals):.1f}% (values: {', '.join(f'{v:.1f}%' for v in vals)})")

# Win count
umi_wins = 0
total = 0
for s in sectors:
    for depth in depths:
        w = all_results[s]["wotan"][str(depth)]
        u = all_results[s]["umi"][str(depth)]
        if u < w:
            umi_wins += 1
        total += 1
print(f"\nUMI wins: {umi_wins}/{total} depth-sector combinations ({umi_wins/total*100:.0f}%)")

# ==================================================================
# Save results
# ==================================================================
result_doc = {
    "title": "Multi-Sector Injection Recovery Validation",
    "date": datetime.now().isoformat(),
    "config": {
        "sectors": list(sectors.keys()),
        "n_stars_per_sector": N_STARS,
        "period_days": period,
        "duration_hours": duration * 24,
        "depths": depths,
        "umi_asymmetry": 1.5,
    },
    "per_sector": {str(s): all_results[s] for s in sectors},
    "summary": {
        "umi_wins": umi_wins,
        "total_comparisons": total,
        "umi_win_rate": round(umi_wins / total, 2),
        "consistency": {
            str(d): {
                "umi_values": [all_results[s]["umi"][str(d)] for s in sectors],
                "umi_std": round(np.std([all_results[s]["umi"][str(d)] for s in sectors]), 1),
                "wotan_values": [all_results[s]["wotan"][str(d)] for s in sectors],
            }
            for d in depths
        },
    },
}

out_file = RESULTS_DIR / "multisector_validation.json"
with open(out_file, "w") as f:
    json.dump(result_doc, f, indent=2)
print(f"\nResults saved to {out_file}")
