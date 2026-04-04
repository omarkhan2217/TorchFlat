"""Kepler multi-quarter validation: injection recovery on Q2, Q5, Q9, Q17.

Tests UMI consistency across different Kepler quarters, analogous to
the multi-sector TESS validation. 1000 stars per quarter, 5 depths.

Saves results to results/kepler_multi_quarter.json
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

from torchflat.umi import umi_detrend

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("D:/TorchFlat/data/kepler")
device = torch.device("cuda")

QUARTERS = [2, 5, 9, 17]
N_STARS = 1000
PERIOD = 3.0        # days
DURATION = 3.0 / 24 # 3 hours
DEPTHS = [0.001, 0.003, 0.005, 0.01, 0.05]  # 0.1% to 5%


def load_quarter_stars(quarter, n):
    """Load n stars from a specific Kepler quarter."""
    stars = []
    for f in sorted(DATA_DIR.glob("*.fits")):
        if len(stars) >= n:
            break
        try:
            with fits.open(f) as h:
                if h[0].header.get("QUARTER") != quarter:
                    continue
                d = h[1].data
                t = np.array(d["TIME"], dtype=np.float64)
                fl = np.array(d["PDCSAP_FLUX"], dtype=np.float64)
                q = np.array(d["SAP_QUALITY"], dtype=np.int32)
                v = (q == 0) & np.isfinite(fl) & np.isfinite(t)
                if v.sum() > 500:
                    stars.append((t[v], fl[v]))
        except Exception:
            continue
    return stars


def detrend_star(t_v, f_v):
    """Detrend a single star with UMI."""
    cadence = np.median(np.diff(t_v))
    W = int(round(0.5 / cadence)) | 1
    L = len(f_v)
    if L < W:
        return np.full(L, np.nan)

    flux_t = torch.tensor(f_v, dtype=torch.float32, device=device).unsqueeze(0)
    time_t = torch.tensor(t_v, dtype=torch.float64, device=device).unsqueeze(0)
    valid_t = torch.ones(1, L, dtype=torch.bool, device=device)
    seg_t = torch.zeros(1, L, dtype=torch.int32, device=device)

    det, trend = umi_detrend(flux_t, time_t, valid_t, seg_t,
                             window_length_days=0.5, asymmetry=2.0)
    det_np = det[0].cpu().numpy()
    return det_np


def recover_depth(det, t, period, duration):
    """Measure recovered transit depth from detrended light curve."""
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


def run_injection(stars, depth):
    """Run injection-recovery for one depth across all stars."""
    errors = []
    for t_v, f_v in stars:
        span = t_v[-1] - t_v[0]
        if PERIOD > span * 0.8:
            continue

        phase = ((t_v - t_v[0]) % PERIOD) / PERIOD
        hd = DURATION / PERIOD / 2.0
        in_tr = (phase < hd) | (phase > 1.0 - hd)
        if in_tr.sum() < 3:
            continue

        fi = f_v.copy()
        fi[in_tr] -= depth * np.median(f_v)

        det = detrend_star(t_v, fi)
        torch.cuda.synchronize()

        r = recover_depth(det, t_v, PERIOD, DURATION)
        if np.isfinite(r) and r > 0:
            err = abs(r - depth) / depth * 100
            errors.append(err)

    if len(errors) > 10:
        return round(float(np.median(errors)), 1), len(errors)
    return None, len(errors)


if __name__ == "__main__":
    results = {"quarters": {}, "config": {
        "n_stars_target": N_STARS, "period_days": PERIOD,
        "duration_hours": DURATION * 24, "depths": DEPTHS,
        "asymmetry": 2.0, "window_days": 0.5,
    }}

    t_total = time.perf_counter()

    for quarter in QUARTERS:
        print(f"\n{'='*50}")
        print(f"Quarter {quarter}")
        print(f"{'='*50}")

        stars = load_quarter_stars(quarter, N_STARS)
        print(f"Loaded {len(stars)} stars")

        if len(stars) < 50:
            print(f"  Skipping Q{quarter}: too few stars")
            continue

        q_results = {"n_stars": len(stars), "depths": {}}

        for depth in DEPTHS:
            t0 = time.perf_counter()
            med_err, n_valid = run_injection(stars, depth)
            elapsed = time.perf_counter() - t0
            pct = f"{depth*100:.1f}%"
            print(f"  depth={pct:>5}: error={med_err}%, n={n_valid}, {elapsed:.0f}s")
            q_results["depths"][str(depth)] = {
                "median_error_pct": med_err,
                "n_valid": n_valid,
            }

        results["quarters"][str(quarter)] = q_results

    elapsed_total = time.perf_counter() - t_total
    results["total_time_sec"] = round(elapsed_total, 1)

    # Summary table
    print(f"\n{'='*60}")
    print(f"KEPLER MULTI-QUARTER VALIDATION (UMI, asymmetry=2.0)")
    print(f"{'='*60}")
    header = f"{'Depth':>8}"
    for q in QUARTERS:
        header += f"  {'Q'+str(q):>8}"
    print(header)

    for depth in DEPTHS:
        row = f"{depth*100:.1f}%".rjust(8)
        for q in QUARTERS:
            qk = str(q)
            if qk in results["quarters"]:
                e = results["quarters"][qk]["depths"].get(str(depth), {}).get("median_error_pct")
                row += f"  {e:>7.1f}%" if e is not None else "      N/A"
            else:
                row += "     skip"
        print(row)

    print(f"\nTotal time: {elapsed_total/60:.1f} min")

    out = RESULTS_DIR / "kepler_multi_quarter.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out}")
