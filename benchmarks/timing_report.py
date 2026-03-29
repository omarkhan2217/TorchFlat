#!/usr/bin/env python
"""Detailed pipeline timing report with complexity analysis."""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits

os.environ["TORCHFLAT_NO_KERNEL"] = "1"

from torchflat._utils import masked_median
from torchflat.batching import assemble_batch, bucket_stars, compute_max_batch, cpu_prescan
from torchflat.umi import umi_detrend as biweight_detrend
from torchflat.clipping import conservative_clip, rolling_clip
from torchflat.gaps import detect_gaps, interpolate_small_gaps
from torchflat.highpass import fft_highpass
from torchflat.normalize import normalize_track_a, normalize_track_b
from torchflat.quality import quality_filter
from torchflat.windows import extract_windows

LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

sync = torch.cuda.synchronize


def main():
    data_dir = Path("C:/Users/USER/Desktop/Website/trips-section/Celix/data/fits/sector_6")
    star_data = []
    for f in sorted(data_dir.glob("*.fits"))[:200]:
        with fits.open(f) as h:
            d = h[1].data
            star_data.append({
                "time": np.array(d["TIME"], dtype=np.float64),
                "pdcsap_flux": np.array(d["PDCSAP_FLUX"], dtype=np.float32),
                "sap_flux": np.array(d["SAP_FLUX"], dtype=np.float32),
                "quality": np.array(d["QUALITY"], dtype=np.int32),
            })

    times_np = [s["time"] for s in star_data]
    fluxes_np = [s["pdcsap_flux"] for s in star_data]
    quals_np = [s["quality"] for s in star_data]

    prescan = cpu_prescan(times_np, fluxes_np, quals_np)
    buckets = bucket_stars(prescan)
    bucket = buckets[0]
    device = torch.device("cuda")

    B = min(50, len(bucket["star_indices"]))
    bi = bucket["star_indices"][:B]
    L = bucket["pad_length"]

    # Warmup
    b = assemble_batch(bi[:5], times_np, fluxes_np, quals_np, L, device)
    quality_filter(b["flux"], b["time"], b["quality"])
    sync()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Assemble batch
    batch = assemble_batch(bi, times_np, fluxes_np, quals_np, L, device)
    flux = batch["flux"]
    time_t = batch["time"]
    qual = batch["quality"]
    vm = batch["valid_mask"]

    steps = []

    # 1. Quality filter: 3 elementwise ops
    sync(); t0 = time.perf_counter()
    vm = quality_filter(flux, time_t, qual) & vm
    sync(); dt = time.perf_counter() - t0
    steps.append(("quality_filter", dt, "O(B*L)", f"[{B},{L}]", "3 elementwise: bitmask AND + 2x isfinite"))

    # 2. Gap interpolation: cummax + cummin + gather
    sync(); t0 = time.perf_counter()
    flux, vm = interpolate_small_gaps(flux, time_t, vm)
    sync(); dt = time.perf_counter() - t0
    steps.append(("interpolate_gaps", dt, "O(B*L)", f"[{B},{L}]", "cummax + cummin + gather + lerp"))

    # 3. Gap detection: diff + median + cumsum
    sync(); t0 = time.perf_counter()
    seg, mc = detect_gaps(time_t, vm)
    sync(); dt = time.perf_counter() - t0
    steps.append(("detect_gaps", dt, "O(B*L)", f"[{B},{L}]", "diff + torch.median + cumsum"))

    # 4. Rolling clip breakdown
    W_clip = 25

    sync(); t0 = time.perf_counter()
    win_c = flux.unfold(1, W_clip, 1)
    vwin_c = vm.unfold(1, W_clip, 1)
    sync(); dt = time.perf_counter() - t0
    steps.append(("clip: unfold", dt, "O(1)", f"[{B},{L-W_clip+1},{W_clip}]", "strided view (no copy)"))

    sync(); t0 = time.perf_counter()
    rolling_med = masked_median(win_c, vwin_c)
    sync(); dt = time.perf_counter() - t0
    N_clip = L - W_clip + 1
    steps.append(("clip: sort_median", dt, f"O(B*{N_clip}*{W_clip}*log{W_clip})", f"[{B},{N_clip},{W_clip}]",
                  f"torch.sort on W={W_clip} per window"))

    sync(); t0 = time.perf_counter()
    pad = W_clip // 2
    rm = F.pad(rolling_med, (pad, pad), mode="replicate")
    if rm.shape[1] > L:
        rm = rm[:, :L]
    res = (flux - rm).abs()
    mad_c = masked_median(res, vm)
    thr = (5.0 * mad_c.clamp(min=1e-10) / 0.6745).unsqueeze(1)
    vm = vm & (res <= thr)
    sync(); dt = time.perf_counter() - t0
    steps.append(("clip: MAD+threshold", dt, "O(B*L)", f"[{B},{L}]", "residuals + masked_median(MAD) + mask"))

    # 5. Biweight breakdown
    W = 361
    N_pos = L - W + 1

    sync(); t0 = time.perf_counter()
    flux_w = flux.float().unfold(1, W, 1).contiguous()
    vm_w = vm.unfold(1, W, 1).contiguous()
    seg_w = seg.unfold(1, W, 1).contiguous()
    center = seg_w[:, :, W // 2: W // 2 + 1]
    wv = vm_w & (seg_w == center)
    sync(); dt = time.perf_counter() - t0
    mem_mb = flux_w.nelement() * 4 / 1024**2
    steps.append(("bw: unfold+contiguous", dt, f"O(B*{N_pos}*{W})", f"[{B},{N_pos},{W}]",
                  f"3 unfold+contiguous copies, {mem_mb:.0f}MB each"))

    sync(); t0 = time.perf_counter()
    location = masked_median(flux_w, wv)
    sync(); dt = time.perf_counter() - t0
    steps.append(("bw: initial_median", dt, f"O(B*{N_pos}*{W}*log{W})", f"[{B},{N_pos},{W}]",
                  f"torch.sort on {B*N_pos} rows of {W} elements"))

    sync(); t0 = time.perf_counter()
    abs_dev = (flux_w - location.unsqueeze(-1)).abs()
    mad_bw = masked_median(abs_dev, wv)
    sync(); dt = time.perf_counter() - t0
    steps.append(("bw: MAD_median", dt, f"O(B*{N_pos}*{W}*log{W})", f"[{B},{N_pos},{W}]",
                  "abs_dev + sort for MAD"))

    sync(); t0 = time.perf_counter()
    safe_mad = (5.0 * mad_bw.clamp(min=1e-10)).unsqueeze(-1)
    for _ in range(5):
        u = (flux_w - location.unsqueeze(-1)) / safe_mad
        weights = ((1.0 - u**2)**2) * (u.abs() < 1.0).float() * wv.float()
        w_sum = weights.sum(dim=-1).clamp(min=1e-10)
        location = (flux_w * weights).sum(dim=-1) / w_sum
    sync(); dt = time.perf_counter() - t0
    steps.append(("bw: 5_iterations", dt, f"O(B*{N_pos}*{W}*5)", f"[{B},{N_pos},{W}]",
                  "weighted mean x5 (elementwise, no sort)"))

    # 6. Normalize
    sync(); t0 = time.perf_counter()
    trend_full = torch.full((B, L), float("nan"), dtype=torch.float32, device=device)
    offset = W // 2
    trend_full[:, offset:offset + N_pos] = location
    det = torch.where((trend_full > 0) & vm & torch.isfinite(trend_full),
                      flux / trend_full, torch.tensor(float("nan"), device=device))
    norm = normalize_track_a(det, vm)
    sync(); dt = time.perf_counter() - t0
    steps.append(("normalize_a", dt, "O(B*L)", f"[{B},{L}]", "median-divide + clamp"))

    # 7. Window extraction
    sync(); t0 = time.perf_counter()
    extract_windows(norm, vm, seg, time_t, window_scales=[(256, 128)])
    sync(); dt = time.perf_counter() - t0
    steps.append(("extract_windows", dt, "O(B*L*scales)", f"[{B},{L}]", "unfold + segment check + CPU transfer"))

    # 8. Track B
    sync(); t0 = time.perf_counter()
    vm_b = quality_filter(batch["flux"], time_t, qual) & batch["valid_mask"]
    flux_b, vm_b = interpolate_small_gaps(batch["flux"], time_t, vm_b)
    seg_b, mc_b = detect_gaps(time_t, vm_b)
    vm_b = conservative_clip(flux_b, vm_b)
    filt = fft_highpass(flux_b, vm_b, seg_b, mc_b)
    normalize_track_b(filt, vm_b)
    sync(); dt = time.perf_counter() - t0
    steps.append(("track_b", dt, "O(B*L*logL)", f"[{B},{L}]", "quality+gaps+clip+FFT+normalize"))

    # Report
    total = sum(s[1] for s in steps)
    print(f"\n{'='*95}")
    print(f"PIPELINE TIMING REPORT - {B} stars x {L} points (real TESS sector 6)")
    print(f"{'='*95}")
    print(f"{'Step':<24} {'Time':>8} {'%':>6} {'Complexity':<28} {'Shape':<20} Description")
    print(f"{'-'*95}")
    for name, dt, cplx, shape, desc in sorted(steps, key=lambda x: -x[1]):
        pct = dt / total * 100
        bar = "#" * int(pct / 2)
        print(f"{name:<24} {dt:>8.4f} {pct:>5.1f}% {cplx:<28} {shape:<20} {desc}")
    print(f"{'-'*95}")
    print(f"{'TOTAL':<24} {total:>8.4f}")
    print(f"Stars/sec: {B / total:.1f}")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")

    # Sort calls account for:
    sort_time = sum(s[1] for s in steps if "sort" in s[0] or "median" in s[0].lower())
    print(f"\nSort-based median calls: {sort_time:.4f}s ({sort_time/total*100:.1f}% of total)")
    iter_time = sum(s[1] for s in steps if "iteration" in s[0])
    print(f"Biweight iterations:     {iter_time:.4f}s ({iter_time/total*100:.1f}% of total)")
    other = total - sort_time - iter_time
    print(f"Everything else:         {other:.4f}s ({other/total*100:.1f}% of total)")

    # Save
    log = {
        "timestamp": datetime.now().isoformat(),
        "config": {"B": B, "L": L, "W_biweight": W, "W_clip": W_clip, "N_pos": N_pos},
        "steps": {s[0]: {"time_s": round(s[1], 5), "pct": round(s[1]/total*100, 1),
                         "complexity": s[2], "shape": s[3], "description": s[4]} for s in steps},
        "total_s": round(total, 4),
        "stars_per_sec": round(B / total, 1),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2),
    }
    log_file = LOGS_DIR / f"timing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nLog: {log_file}")


if __name__ == "__main__":
    main()
