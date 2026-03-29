"""Test: mean-based Numba biweight kernel (no sort at all)."""
import os
os.environ["TORCHFLAT_NO_KERNEL"] = "1"

import time
import numpy as np
from numba import njit
from pathlib import Path
from astropy.io import fits


@njit
def biweight_mean_init(flux, valid, seg_id, W, n_iter, cval):
    """Biweight detrend using mean+std init instead of median (no sort)."""
    L = len(flux)
    trend = np.full(L, np.nan, dtype=np.float64)
    half_w = W // 2

    for pos in range(half_w, L - half_w):
        center_seg = seg_id[pos]

        buf = np.empty(W, dtype=np.float64)
        n = 0
        for k in range(pos - half_w, pos + half_w + 1):
            if valid[k] and seg_id[k] == center_seg:
                buf[n] = flux[k]
                n += 1

        if n < 50:
            continue

        # Mean (O(n), no sort)
        s = 0.0
        for j in range(n):
            s += buf[j]
        location = s / n

        # Std-based scale (O(n), no sort)
        ss = 0.0
        for j in range(n):
            d = buf[j] - location
            ss += d * d
        std = (ss / n) ** 0.5
        safe_mad = cval * 0.6745 * std
        if safe_mad < 1e-10:
            safe_mad = 1e-10

        for _ in range(n_iter):
            w_sum = 0.0
            wx_sum = 0.0
            for j in range(n):
                u = (buf[j] - location) / safe_mad
                if abs(u) < 1.0:
                    w = (1.0 - u * u) ** 2
                    w_sum += w
                    wx_sum += w * buf[j]
            if w_sum > 1e-10:
                location = wx_sum / w_sum

        trend[pos] = location

    return trend


def main():
    data_dir = Path("C:/Users/USER/Desktop/Website/trips-section/Celix/data/fits/sector_6")
    fits_files = sorted(data_dir.glob("*.fits"))[:5]

    stars = []
    for f in fits_files:
        with fits.open(f) as h:
            d = h[1].data
            t = np.array(d["TIME"], dtype=np.float64)
            fl = np.array(d["PDCSAP_FLUX"], dtype=np.float64)
            q = np.array(d["QUALITY"], dtype=np.int32)
            v = ((q & 3455) == 0) & np.isfinite(fl) & np.isfinite(t)
            dt = np.diff(t)
            mc = np.median(dt[v[1:] & v[:-1]])
            W = int(round(0.5 / mc)) | 1
            seg = np.zeros(len(fl), dtype=np.int32)
            stars.append((fl, v, seg, W))

    # Compile both
    from torchflat.biweight_cpu import _biweight_one_star
    print("Compiling...")
    fl, v, seg, W = stars[0]
    _biweight_one_star(fl, np.arange(len(fl), dtype=np.float64), v, seg, 0.5, 5, 5.0, 50)
    biweight_mean_init(fl, v, seg, W, 3, 5.0)
    print("Done")

    # Benchmark: median-based (current)
    t0 = time.perf_counter()
    for _ in range(10):
        for fl, v, seg, W in stars:
            _biweight_one_star(fl, np.arange(len(fl), dtype=np.float64), v, seg, 0.5, 5, 5.0, 50)
    t_med = (time.perf_counter() - t0) / 10 / len(stars)
    print(f"Median-based (5 iter): {t_med:.4f}s/star ({1/t_med:.1f}/sec)")

    # Benchmark: mean-based (new, 3 iter)
    t0 = time.perf_counter()
    for _ in range(10):
        for fl, v, seg, W in stars:
            biweight_mean_init(fl, v, seg, W, 3, 5.0)
    t_mean3 = (time.perf_counter() - t0) / 10 / len(stars)
    print(f"Mean-based (3 iter):   {t_mean3:.4f}s/star ({1/t_mean3:.1f}/sec)")

    # Mean-based with 5 iter
    t0 = time.perf_counter()
    for _ in range(10):
        for fl, v, seg, W in stars:
            biweight_mean_init(fl, v, seg, W, 5, 5.0)
    t_mean5 = (time.perf_counter() - t0) / 10 / len(stars)
    print(f"Mean-based (5 iter):   {t_mean5:.4f}s/star ({1/t_mean5:.1f}/sec)")

    print(f"\nSpeedup: mean3={t_med/t_mean3:.2f}x  mean5={t_med/t_mean5:.2f}x")

    # Accuracy
    fl, v, seg, W = stars[0]
    _, trend_med = _biweight_one_star(fl, np.arange(len(fl), dtype=np.float64), v, seg, 0.5, 5, 5.0, 50)
    trend_mean = biweight_mean_init(fl, v, seg, W, 5, 5.0)
    both = np.isfinite(trend_med) & np.isfinite(trend_mean)
    if both.sum() > 0:
        rel = np.abs(trend_med[both] - trend_mean[both]) / np.abs(trend_med[both]).clip(1e-10)
        print(f"Accuracy: p99={np.percentile(rel, 99):.2e} max={rel.max():.2e}")

    # Project
    best = min(t_med, t_mean3, t_mean5)
    print(f"\n12-worker projection: {12/best:.1f}/sec = {19618/(12/best)/60:.1f}min/sector")


if __name__ == "__main__":
    main()
