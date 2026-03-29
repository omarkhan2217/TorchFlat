"""Compare biweight vs huber vs welsch: accuracy and GPU speed."""
import os
os.environ["TORCHFLAT_NO_KERNEL"] = "1"

import time
import numpy as np
import torch
import wotan
from pathlib import Path
from astropy.io import fits

device = torch.device("cuda")
data_dir = Path("C:/Users/USER/Desktop/Website/trips-section/Celix/data/fits/sector_6")

# Load 50 real stars
stars = []
for f in sorted(data_dir.glob("*.fits"))[:50]:
    with fits.open(f) as h:
        d = h[1].data
        stars.append({
            "time": np.array(d["TIME"], dtype=np.float64),
            "flux": np.array(d["PDCSAP_FLUX"], dtype=np.float64),
            "quality": np.array(d["QUALITY"], dtype=np.int32),
        })

# ================================================================
# 1. WOTAN: accuracy of each method vs biweight
# ================================================================
print("=== WOTAN: Each method vs biweight (50 real stars) ===\n")

bw_trends = []
for s in stars:
    v = ((s["quality"] & 3455) == 0) & np.isfinite(s["flux"]) & np.isfinite(s["time"])
    t_v, f_v = s["time"][v], s["flux"][v]
    _, trend = wotan.flatten(t_v, f_v, method="biweight", window_length=0.5, return_trend=True)
    bw_trends.append((t_v, f_v, trend))

for method in ["welsch"]:
    errors = []
    t0 = time.perf_counter()
    for i, s in enumerate(stars):
        t_v, f_v, bw_trend = bw_trends[i]
        try:
            _, alt_trend = wotan.flatten(t_v, f_v, method=method,
                                         window_length=0.5, return_trend=True)
        except Exception:
            continue
        both = np.isfinite(bw_trend) & np.isfinite(alt_trend)
        if both.sum() > 0:
            rel = np.abs(bw_trend[both] - alt_trend[both]) / np.abs(bw_trend[both]).clip(1e-10)
            errors.append(rel)
    elapsed = time.perf_counter() - t0
    if errors:
        e = np.concatenate(errors)
        print(f"  {method:>12}: p99={np.percentile(e,99):.2e}  max={e.max():.2e}  "
              f"mean={e.mean():.2e}  ({elapsed:.1f}s)")

# ================================================================
# 2. WOTAN: speed comparison
# ================================================================
print("\n=== WOTAN: Speed (10 stars, single thread) ===\n")

for method in ["biweight", "welsch"]:
    t0 = time.perf_counter()
    for s in stars[:10]:
        v = ((s["quality"] & 3455) == 0) & np.isfinite(s["flux"]) & np.isfinite(s["time"])
        wotan.flatten(s["time"][v], s["flux"][v], method=method,
                      window_length=0.5, return_trend=True)
    elapsed = time.perf_counter() - t0
    rate = 10 / elapsed
    print(f"  {method:>12}: {rate:.1f}/sec/thread  ({rate*12:.0f}/sec 12-worker)  "
          f"sector={19618/rate/12/60:.1f}min")

# ================================================================
# 3. GPU: mean-init biweight vs huber vs welsch
# ================================================================
print("\n=== GPU: Mean-init iterations (50 stars, no sort) ===\n")

from torchflat.batching import assemble_batch, bucket_stars, cpu_prescan
from torchflat.quality import quality_filter
from torchflat.gaps import detect_gaps
from torchflat.clipping import rolling_clip

times_np = [s["time"] for s in stars]
fluxes_np = [s["flux"].astype(np.float32) for s in stars]
quals_np = [s["quality"] for s in stars]
prescan = cpu_prescan(times_np, fluxes_np, quals_np)
buckets = bucket_stars(prescan)
bucket = buckets[0]
B = min(50, len(bucket["star_indices"]))
bi = bucket["star_indices"][:B]
batch = assemble_batch(bi, times_np, fluxes_np, quals_np, bucket["pad_length"], device)
flux = batch["flux"]
time_t = batch["time"]
vm = quality_filter(flux, time_t, batch["quality"]) & batch["valid_mask"]
seg, mc = detect_gaps(time_t, vm)
vm = rolling_clip(flux, vm, seg)
torch.cuda.synchronize()

W = 361
flux_w = flux.float().unfold(1, W, 1).contiguous()
vm_w = vm.unfold(1, W, 1).contiguous()
seg_w = seg.unfold(1, W, 1).contiguous()
center = seg_w[:, :, W // 2: W // 2 + 1]
wv = vm_w & (seg_w == center)
n_valid = wv.float().sum(dim=-1).clamp(min=1)
cval = 5.0

# Mean + std init (shared)
mf = flux_w * wv.float()
loc_init = mf.sum(dim=-1) / n_valid
dsq = ((flux_w - loc_init.unsqueeze(-1)) ** 2) * wv.float()
std = (dsq.sum(dim=-1) / n_valid).sqrt()
mad_init = 0.6745 * std
sm = (cval * mad_init.clamp(min=1e-10)).unsqueeze(-1)
torch.cuda.synchronize()

# Warmup
for _ in range(2):
    u = (flux_w - loc_init.unsqueeze(-1)) / sm
    _ = torch.exp(-0.5 * u ** 2)
torch.cuda.synchronize()

for name in ["biweight", "huber", "welsch"]:
    for n_iter in [3, 5]:
        loc = loc_init.clone()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            u = (flux_w - loc.unsqueeze(-1)) / sm

            if name == "biweight":
                w = ((1.0 - u ** 2) ** 2) * (u.abs() < 1.0).float() * wv.float()
            elif name == "huber":
                w = torch.where(u.abs() <= 1.0, torch.ones_like(u),
                                1.0 / u.abs().clamp(min=1e-10)) * wv.float()
            elif name == "welsch":
                w = torch.exp(-0.5 * u ** 2) * wv.float()

            ws = w.sum(dim=-1).clamp(min=1e-10)
            loc = (flux_w * w).sum(dim=-1) / ws

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        rate = B / dt
        print(f"  {name:>12} ({n_iter} iter): {dt:.4f}s  {rate:.0f} stars/sec  "
              f"sector={19618/rate/60:.1f}min")

# ================================================================
# 4. Full pipeline projection
# ================================================================
print("\n=== PROJECTION: Full pipeline with mean-init (no sort) ===\n")

# The biweight init (mean+std) takes ~0.10s
# The iterations take ~0.09s per iteration
# Unfold+contiguous takes ~0.13s
# Rolling clip takes ~0.13s (sort on W=25)
# Other steps: ~0.15s
# Total WITHOUT sort init: ~0.10 + 0.13 + 0.13 + 0.15 + 3*0.09 = 0.78s for 50 stars

# With sort init (current): add ~0.55s sort = 1.33s total
# Without sort (mean init): 0.78s total

print("  Estimated per-batch (50 stars):")
print("    Sort-based (current): ~1.3s  (~38/sec)")
print("    Mean-init (no sort):  ~0.8s  (~63/sec)")
print("    Sector (mean-init):   ~5.2min")
print("    vs Celix:             ~15x faster")


if __name__ == "__main__":
    pass
