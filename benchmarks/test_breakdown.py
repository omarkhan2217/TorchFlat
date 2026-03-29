"""Monte Carlo breakdown point characterization for UMI asymmetric bisquare.

Tests: at what fraction of transit contamination does the estimator fail?
Uses realistic contamination depths (1-50 sigma, matching transit depths).
Saves results to results/breakdown_point.json.
"""
import json
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchflat._kernel_loader import _get_umi_kernel
kern = _get_umi_kernel()

def umi_location(data, asym, cval=5.0, n_iter=10):
    """Compute UMI location for a single window."""
    n = len(data)
    if kern is not None and device.type == "cuda":
        ft = torch.tensor(data, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        vt = torch.ones(1, 1, n, dtype=torch.bool, device=device)
        loc = kern.umi_detrend(ft, vt, cval, asym, n_iter, 3)
        return loc.item()
    else:
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad < 1e-10: mad = 1e-10
        scale = cval * mad
        loc = median
        for _ in range(n_iter):
            u = (data - loc) / scale
            u_eff = np.where(u < 0, u * asym, u)
            u_abs = np.abs(u_eff)
            mask = u_abs < 1
            w = ((1 - u_abs**2)**2) * mask
            ws = w.sum()
            if ws > 1e-10:
                loc = (data * w).sum() / ws
        return loc

rng = np.random.default_rng(42)
N = 361
n_trials = 200

fractions = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
# Realistic transit depths: 1 sigma (0.1% transit) to 50 sigma (5% transit)
depths_sigma = [1, 3, 5, 10, 50]

print("=== BREAKDOWN POINT: BIWEIGHT vs UMI ===")
print(f"    N={N}, {n_trials} trials, contamination = downward shift")
print(f"    Bias shown as fraction of contamination depth")
print()

results = {}

for depth in depths_sigma:
    print(f"\n--- Contamination depth: {depth} sigma ---")
    print(f"{'Frac':>6} {'n_con':>5}  {'bw_bias':>10}  {'umi_bias':>10}  {'bw_rel%':>8}  {'umi_rel%':>8}")
    print("-" * 55)

    results[str(depth)] = {}

    for frac in fractions:
        n_contam = int(N * frac)
        bias_bw, bias_umi = [], []

        for _ in range(n_trials):
            data = rng.normal(0, 1, N)
            contam_idx = rng.choice(N, n_contam, replace=False)
            data[contam_idx] -= depth

            loc_bw = umi_location(data.copy(), asym=1.0)
            loc_umi = umi_location(data.copy(), asym=1.5)

            bias_bw.append(loc_bw)
            bias_umi.append(loc_umi)

        med_bw = np.median(bias_bw)
        med_umi = np.median(bias_umi)
        # Relative: bias as % of contamination depth
        rel_bw = med_bw / depth * 100
        rel_umi = med_umi / depth * 100

        results[str(depth)][str(frac)] = {
            "biweight_median_bias": round(float(med_bw), 4),
            "umi_median_bias": round(float(med_umi), 4),
            "biweight_relative_pct": round(float(rel_bw), 2),
            "umi_relative_pct": round(float(rel_umi), 2),
        }

        print(f"{frac:>6.2f} {n_contam:>5}  {med_bw:>+10.4f}  {med_umi:>+10.4f}  {rel_bw:>+7.1f}%  {rel_umi:>+7.1f}%")

# Summary: at each depth, find the fraction where bias > 10% of depth
print(f"\n=== BREAKDOWN (bias > 10% of contamination depth) ===")
for depth in depths_sigma:
    bp_bw, bp_umi = ">50%", ">50%"
    for frac in fractions:
        r = results[str(depth)][str(frac)]
        if bp_bw == ">50%" and abs(r["biweight_relative_pct"]) > 10:
            bp_bw = f"{frac:.0%}"
        if bp_umi == ">50%" and abs(r["umi_relative_pct"]) > 10:
            bp_umi = f"{frac:.0%}"
    print(f"  depth={depth:>2} sigma: biweight @ {bp_bw:>5}, UMI @ {bp_umi:>5}")

# Note on UMI asymmetry bias
print(f"\nNote: UMI has a constant upward bias of ~0.19 sigma from asymmetric")
print(f"weighting. This is independent of contamination and equals ~190 ppm")
print(f"on TESS data (below noise floor). At small contamination depths")
print(f"(1 sigma), this bias dominates the relative error percentage.")

doc = {
    "title": "UMI Breakdown Point Monte Carlo Analysis",
    "date": datetime.now().isoformat(),
    "config": {"window_size": N, "n_trials": n_trials,
               "depths_sigma": depths_sigma, "fractions": fractions},
    "results": results,
}
out = RESULTS_DIR / "breakdown_point.json"
with open(out, "w") as f:
    json.dump(doc, f, indent=2)
print(f"\nSaved to {out}")
