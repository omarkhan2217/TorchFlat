#!/usr/bin/env python
"""Deep profile every sub-step and test optimization ideas."""
import os
os.environ["TORCHFLAT_NO_KERNEL"] = "1"

import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from astropy.io import fits

from torchflat.batching import assemble_batch, bucket_stars, cpu_prescan
from torchflat.quality import quality_filter
from torchflat.gaps import detect_gaps
from torchflat.clipping import rolling_clip
from torchflat._utils import masked_median
from torchflat.normalize import normalize_track_a
from torchflat.windows import extract_windows

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
sync = torch.cuda.synchronize

B = min(50, len(bucket["star_indices"]))
bi = bucket["star_indices"][:B]
L = bucket["pad_length"]

# Warmup
b = assemble_batch(bi[:5], times_np, fluxes_np, quals_np, L, device)
quality_filter(b["flux"], b["time"], b["quality"])
sync()
torch.cuda.empty_cache()

# Full pipeline timing
batch = assemble_batch(bi, times_np, fluxes_np, quals_np, L, device)
flux = batch["flux"]; time_t = batch["time"]; qual = batch["quality"]; vm = batch["valid_mask"]

steps = []

sync(); t0 = time.perf_counter()
vm = quality_filter(flux, time_t, qual) & vm
sync(); steps.append(("quality_filter", time.perf_counter() - t0))

sync(); t0 = time.perf_counter()
seg, mc = detect_gaps(time_t, vm)
sync(); steps.append(("detect_gaps", time.perf_counter() - t0))

W_clip = 25
sync(); t0 = time.perf_counter()
win_c = flux.unfold(1, W_clip, 1)
vwin_c = vm.unfold(1, W_clip, 1)
rolling_med = masked_median(win_c, vwin_c)
sync(); steps.append(("clip_sort_median", time.perf_counter() - t0))

sync(); t0 = time.perf_counter()
pad = W_clip // 2
rm = F.pad(rolling_med, (pad, pad), mode="replicate")[:, :L]
res = (flux - rm).abs()
res_m = res.clone()
res_m[~vm] = float("inf")
mad_c = torch.median(res_m, dim=-1).values
thr = (5.0 * mad_c.clamp(min=1e-10) / 0.6745).unsqueeze(1)
vm = vm & (res <= thr)
sync(); steps.append(("clip_mad_threshold", time.perf_counter() - t0))

W = 361
N_pos = L - W + 1

sync(); t0 = time.perf_counter()
flux_w = flux.float().unfold(1, W, 1).contiguous()
vm_w = vm.unfold(1, W, 1).contiguous()
seg_w = seg.unfold(1, W, 1).contiguous()
center = seg_w[:, :, W // 2: W // 2 + 1]
wv = vm_w & (seg_w == center)
sync(); steps.append(("bw_unfold_contiguous", time.perf_counter() - t0))

sync(); t0 = time.perf_counter()
from torchflat.umi import umi_detrend
det, trend = umi_detrend(flux, time_t, vm, seg)
sync(); steps.append(("bw_full_detrend", time.perf_counter() - t0))

sync(); t0 = time.perf_counter()
norm = normalize_track_a(det, vm)
sync(); steps.append(("normalize", time.perf_counter() - t0))

sync(); t0 = time.perf_counter()
extract_windows(norm, vm, seg, time_t, window_scales=[(256, 128)])
sync(); steps.append(("extract_windows", time.perf_counter() - t0))

total = sum(s[1] for s in steps)
print(f"\nPIPELINE: B={B}, L={L}")
print(f"{'Step':<24} {'Time':>8} {'%':>6}")
print("-" * 42)
for name, dt in sorted(steps, key=lambda x: -x[1]):
    pct = dt / total * 100
    bar = "#" * int(pct / 2)
    print(f"{name:<24} {dt:>8.4f} {pct:>5.1f}%  {bar}")
print("-" * 42)
print(f"{'TOTAL':<24} {total:>8.4f}  {B/total:.1f}/sec")

# =====================================================
# Additional optimization ideas
# =====================================================
print("\n=== OPTIMIZATION IDEAS ===\n")

# 1. Can we avoid 3 contiguous copies?
print("--- Unfold costs ---")
sync(); t0 = time.perf_counter()
_ = flux.float().unfold(1, W, 1).contiguous()
sync()
print(f"  flux unfold+contig:  {time.perf_counter()-t0:.4f}s")

sync(); t0 = time.perf_counter()
_ = vm.unfold(1, W, 1).contiguous()
sync()
print(f"  mask unfold+contig:  {time.perf_counter()-t0:.4f}s")

sync(); t0 = time.perf_counter()
_ = seg.unfold(1, W, 1).contiguous()
sync()
print(f"  seg  unfold+contig:  {time.perf_counter()-t0:.4f}s")

# Non-contiguous mask/seg
sync(); t0 = time.perf_counter()
_ = flux.float().unfold(1, W, 1).contiguous()
vm_nc = vm.unfold(1, W, 1)
seg_nc = seg.unfold(1, W, 1)
c_nc = seg_nc[:, :, W//2:W//2+1]
_ = vm_nc & (seg_nc == c_nc)
sync()
print(f"  1 contig + 2 views:  {time.perf_counter()-t0:.4f}s")

# 2. Clip sort on W=25: can we use kthvalue?
print("\n--- Clip median alternatives ---")
win_c2 = flux.unfold(1, W_clip, 1)
vwin_c2 = vm.unfold(1, W_clip, 1)

sync(); t0 = time.perf_counter()
_ = masked_median(win_c2, vwin_c2)
sync()
print(f"  Sort W=25:    {time.perf_counter()-t0:.4f}s")

# kthvalue on W=25 (all valid)?
n_v_clip = vwin_c2.sum(dim=-1)
all_full_clip = (n_v_clip == W_clip)
pct_full = all_full_clip.float().mean().item() * 100
print(f"  Fully valid windows: {pct_full:.0f}%")

if pct_full > 50:
    working_c = win_c2.clone()
    working_c[~vwin_c2] = float("inf")
    k = W_clip // 2 + 1
    sync(); t0 = time.perf_counter()
    _ = torch.kthvalue(working_c, k, dim=-1).values
    sync()
    print(f"  kthvalue W=25: {time.perf_counter()-t0:.4f}s")

# 3. Float16 sort (half bandwidth)
print("\n--- Sort precision ---")
working_bw = flux_w.clone()
working_bw[~wv] = float("inf")

sync(); t0 = time.perf_counter()
_ = torch.sort(working_bw, dim=-1).values
sync()
print(f"  Sort float32 W=361: {time.perf_counter()-t0:.4f}s")

sync(); t0 = time.perf_counter()
_ = torch.sort(working_bw.half(), dim=-1).values
sync()
print(f"  Sort float16 W=361: {time.perf_counter()-t0:.4f}s")

# 4. Reduce window: use W=181 (every 2nd point)
print("\n--- Reduced window size ---")
sync(); t0 = time.perf_counter()
fw_sub = flux.float().unfold(1, 181, 1).contiguous()
sync()
print(f"  Unfold W=181: {time.perf_counter()-t0:.4f}s")

sync(); t0 = time.perf_counter()
fw_sub2 = flux.float().unfold(1, 361, 1).contiguous()
sync()
print(f"  Unfold W=361: {time.perf_counter()-t0:.4f}s")

ws = fw_sub.clone()
ws[...] = float("inf")  # dummy
sync(); t0 = time.perf_counter()
_ = torch.sort(ws, dim=-1).values
sync()
print(f"  Sort W=181:   {time.perf_counter()-t0:.4f}s")

ws2 = fw_sub2.clone()
ws2[...] = float("inf")
sync(); t0 = time.perf_counter()
_ = torch.sort(ws2, dim=-1).values
sync()
print(f"  Sort W=361:   {time.perf_counter()-t0:.4f}s")

# 5. Can we batch the CPU prescan + assemble?
print("\n--- Batch assembly ---")
sync(); t0 = time.perf_counter()
_ = assemble_batch(bi, times_np, fluxes_np, quals_np, L, device)
sync()
print(f"  Assemble (with CPU interp): {time.perf_counter()-t0:.4f}s")

# 6. What fraction of time is memory allocation?
print("\n--- Memory allocation ---")
sync(); t0 = time.perf_counter()
for _ in range(10):
    _ = torch.empty(B, N_pos, W, dtype=torch.float32, device=device)
sync()
print(f"  10x alloc [{B},{N_pos},{W}]: {(time.perf_counter()-t0)/10:.4f}s")

sync(); t0 = time.perf_counter()
for _ in range(10):
    _ = flux_w.clone()
sync()
print(f"  10x clone [{B},{N_pos},{W}]: {(time.perf_counter()-t0)/10:.4f}s")
