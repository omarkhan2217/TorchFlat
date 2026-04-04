"""Bootstrap 95% confidence intervals for depth recovery accuracy.

Resamples 1000 TESS stars 1000 times, reports median + 95% CI for each method/depth.
"""
import numpy as np
import torch
import wotan
import time
from pathlib import Path
from astropy.io import fits
from scipy.signal import savgol_filter
from torchflat.umi import umi_detrend

device = torch.device("cuda")
DATA_DIR = Path("C:/Users/USER/Desktop/Website/trips-section/Celix/data/fits/sector_6")
N_BOOTSTRAP = 1000
PERIOD = 3.0
DURATION = 3.0 / 24
DEPTHS = [0.0005, 0.001, 0.003, 0.005, 0.01, 0.05]

# Load stars
stars = []
for f in sorted(DATA_DIR.glob("*.fits"))[:1100]:
    try:
        with fits.open(f) as h:
            d = h[1].data
            t = np.array(d["TIME"], dtype=np.float64)
            fl = np.array(d["PDCSAP_FLUX"], dtype=np.float64)
            q = np.array(d["QUALITY"], dtype=np.int32)
            v = ((q & 3455) == 0) & np.isfinite(fl) & np.isfinite(t)
            if v.sum() > 500:
                stars.append((t[v], fl[v].copy()))
    except Exception:
        continue
    if len(stars) >= 1000:
        break
print(f"Loaded {len(stars)} stars")


def rd(det, t):
    finite = np.isfinite(det)
    if finite.sum() < 100:
        return np.nan
    d, tt = det[finite], t[finite]
    ph = ((tt - t[0]) % PERIOD) / PERIOD
    hd = DURATION / PERIOD / 2
    itr = (ph < hd) | (ph > 1 - hd)
    otr = ~itr
    if itr.sum() < 3 or otr.sum() < 50:
        return np.nan
    return float(np.median(d[otr]) - np.median(d[itr]))


# Pre-compute per-star errors for each method/depth
# This avoids re-detrending during bootstrap (just resample the errors)
print("\nPre-computing per-star errors...", flush=True)

methods_config = {
    "UMI": ("umi", 2.0, 5.0),
    "UMI_aggr": ("umi", 10.0, 2.5),
    "welsch": ("wotan", "welsch"),
    "biweight": ("wotan", "biweight"),
    "median": ("wotan", "median"),
    "lowess": ("wotan", "lowess"),
    "savgol": ("savgol",),
}

# Store per-star errors: errors[method][depth] = array of length n_stars
errors = {m: {d: [] for d in DEPTHS} for m in methods_config}

t_start = time.perf_counter()
for depth in DEPTHS:
    pct = f"{depth*100:.2f}%"
    print(f"\n  Depth {pct}:", flush=True)

    # Inject transits into all stars
    injected = []
    for t_v, f_v in stars:
        fi = f_v.copy()
        ph = ((t_v - t_v[0]) % PERIOD) / PERIOD
        hd = DURATION / PERIOD / 2
        itr = (ph < hd) | (ph > 1 - hd)
        fi[itr] -= depth * np.median(f_v)
        injected.append((t_v, fi))

    for method_name, config in methods_config.items():
        t0 = time.perf_counter()
        star_errors = []

        for t_v, fi in injected:
            if config[0] == "umi":
                asym, cval = config[1], config[2]
                L = len(fi)
                ft = torch.tensor(fi, dtype=torch.float32, device=device).unsqueeze(0)
                tt = torch.tensor(t_v, dtype=torch.float64, device=device).unsqueeze(0)
                vt = torch.ones(1, L, dtype=torch.bool, device=device)
                st = torch.zeros(1, L, dtype=torch.int32, device=device)
                det, _ = umi_detrend(ft, tt, vt, st, window_length_days=0.5,
                                     asymmetry=asym, cval=cval)
                det_np = det[0].cpu().numpy()
                torch.cuda.synchronize()
            elif config[0] == "wotan":
                try:
                    det_np, _ = wotan.flatten(t_v, fi, method=config[1],
                                              window_length=0.5, return_trend=True)
                except Exception:
                    star_errors.append(np.nan)
                    continue
            elif config[0] == "savgol":
                cad = np.median(np.diff(t_v))
                W = int(round(0.5 / cad)) | 1
                W = min(W, len(fi) - 1)
                if W < 5:
                    W = 5
                mask = np.ones(len(fi), dtype=bool)
                trend = None
                for _ in range(3):
                    ff = fi.copy()
                    if not mask.all():
                        g = np.where(mask)[0]
                        b = np.where(~mask)[0]
                        if len(g) > 2:
                            ff[b] = np.interp(b, g, fi[g])
                    trend = savgol_filter(ff, W, 2)
                    res = fi - trend
                    mr = np.median(res[mask])
                    mar = np.median(np.abs(res[mask] - mr))
                    if mar < 1e-10:
                        break
                    mask = np.abs(res - mr) < 3.0 * mar / 0.6745
                    if mask.sum() < 100:
                        break
                if trend is not None:
                    det_np = np.where(trend > 0, fi / trend, np.nan)
                else:
                    star_errors.append(np.nan)
                    continue

            r = rd(det_np, t_v)
            if np.isfinite(r) and r > 0:
                star_errors.append(abs(r - depth) / depth * 100)
            else:
                star_errors.append(np.nan)

        errors[method_name][depth] = np.array(star_errors)
        elapsed = time.perf_counter() - t0
        valid = np.sum(np.isfinite(star_errors))
        print(f"    {method_name:>12}: {valid} valid, {elapsed:.0f}s", flush=True)

# Bootstrap
print(f"\n{'='*70}")
print(f"BOOTSTRAP ({N_BOOTSTRAP} resamples)")
print(f"{'='*70}")

rng = np.random.default_rng(42)
n_stars = len(stars)

header = f"{'Method':>12}  {'Depth':>6}  {'Median':>7}  {'CI_lo':>7}  {'CI_hi':>7}  {'CI_width':>8}"
print(f"\n{header}")
print("-" * len(header))

for method_name in methods_config:
    for depth in DEPTHS:
        star_errs = errors[method_name][depth]
        valid_mask = np.isfinite(star_errs)
        valid_errs = star_errs[valid_mask]

        if len(valid_errs) < 50:
            pct = f"{depth*100:.2f}%"
            print(f"{method_name:>12}  {pct:>6}  {'N/A':>7}  {'N/A':>7}  {'N/A':>7}  {'N/A':>8}")
            continue

        boot_medians = []
        for _ in range(N_BOOTSTRAP):
            sample = rng.choice(valid_errs, size=len(valid_errs), replace=True)
            boot_medians.append(np.median(sample))

        boot_medians = np.array(boot_medians)
        med = np.median(valid_errs)
        ci_lo = np.percentile(boot_medians, 2.5)
        ci_hi = np.percentile(boot_medians, 97.5)

        pct = f"{depth*100:.2f}%"
        print(f"{method_name:>12}  {pct:>6}  {med:>6.1f}%  {ci_lo:>6.1f}%  {ci_hi:>6.1f}%  {ci_hi-ci_lo:>7.1f}%")

total = time.perf_counter() - t_start
print(f"\nTotal time: {total/60:.1f} min")
