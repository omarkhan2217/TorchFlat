"""Transit injection recovery: compare ALL methods on same data."""
import os
os.environ["TORCHFLAT_NO_KERNEL"] = "1"

import numpy as np
import time
import torch
import wotan
from pathlib import Path
from astropy.io import fits

from torchflat.batching import assemble_batch, bucket_stars, cpu_prescan
from torchflat.quality import quality_filter
from torchflat.gaps import detect_gaps
from torchflat.clipping import rolling_clip
from torchflat.umi import umi_detrend

device = torch.device("cuda")
data_dir = Path("C:/Users/USER/Desktop/Website/trips-section/Celix/data/fits/sector_6")

# Load base stars
base_stars = []
for f in sorted(data_dir.glob("*.fits"))[:10]:
    with fits.open(f) as h:
        d = h[1].data
        base_stars.append({
            "time": np.array(d["TIME"], dtype=np.float64),
            "flux": np.array(d["PDCSAP_FLUX"], dtype=np.float64),
            "quality": np.array(d["QUALITY"], dtype=np.int32),
        })


def recover_depth(det, t, period, duration):
    n = min(len(det), len(t))
    det, t = det[:n], t[:n]
    finite = np.isfinite(det)
    if finite.sum() < 100:
        return np.nan
    d, tt = det[finite], t[finite]
    phase = ((tt - t[0]) % period) / period
    hd = duration / period / 2.0
    in_tr = (phase < hd) | (phase > 1.0 - hd)
    out_tr = ~in_tr
    if in_tr.sum() < 5 or out_tr.sum() < 50:
        return np.nan
    return float(np.median(d[out_tr]) - np.median(d[in_tr]))


period = 3.0
duration = 3.0 / 24.0
depths = [0.001, 0.005, 0.01, 0.05]

print("=== TRANSIT INJECTION RECOVERY: ALL METHODS ===")
print(f"    10 stars, period={period}d, duration={duration*24:.0f}h, clip_sigma=100 (disabled)")
print()
print(f"{'Depth':>7}  {'wotan_bw':>10}  {'TF_sort_bw':>10}  {'GPU_mean_bw':>11}  {'GPU_mean_we':>11}")
print("-" * 60)

for depth in depths:
    results = {
        "wotan_bw": [],
        "tf_sort_bw": [],
        "gpu_mean_bw": [],
        "gpu_mean_we": [],
    }

    for star in base_stars:
        t = star["time"]
        f = star["flux"]
        q = star["quality"]
        v = ((q & 3455) == 0) & np.isfinite(f) & np.isfinite(t)

        # Inject transit
        phase = ((t - t[0]) % period) / period
        hd = duration / period / 2.0
        in_tr = (phase < hd) | (phase > 1.0 - hd)
        f_inj = f.copy()
        f_inj[in_tr] -= depth * np.median(f[v])

        # 1. WOTAN biweight (the reference)
        t_v, f_v = t[v], f_inj[v]
        flat_w, trend_w = wotan.flatten(
            t_v, f_v, method="biweight", window_length=0.5, return_trend=True
        )
        rd = recover_depth(flat_w, t_v, period, duration)
        if np.isfinite(rd) and rd > 0:
            results["wotan_bw"].append(rd)

        # 2. TorchFlat sort-based biweight (current default)
        prescan = cpu_prescan([t], [f_inj.astype(np.float32)], [q])
        bkts = bucket_stars(prescan)
        if not bkts:
            continue
        bkt = bkts[0]
        batch = assemble_batch(
            bkt["star_indices"], [t], [f_inj.astype(np.float32)], [q],
            bkt["pad_length"], device,
        )
        fl = batch["flux"]
        ti = batch["time"]
        vm = quality_filter(fl, ti, batch["quality"]) & batch["valid_mask"]
        seg, mc = detect_gaps(ti, vm)
        vm_c = rolling_clip(fl, vm, seg, sigma=100.0)

        det_tf, trend_tf = umi_detrend(fl, ti, vm_c, seg)
        torch.cuda.synchronize()
        det_np = det_tf[0].cpu().numpy()
        rd = recover_depth(det_np, t, period, duration)
        if np.isfinite(rd) and rd > 0:
            results["tf_sort_bw"].append(rd)

        # 3 & 4. GPU mean-init biweight & welsch
        W = 361
        L = fl.shape[1]
        if L < W:
            continue
        N_pos = L - W + 1
        offset = W // 2

        fw = fl.float().unfold(1, W, 1).contiguous()
        vmw = vm_c.unfold(1, W, 1).contiguous()
        sw = seg.unfold(1, W, 1).contiguous()
        c = sw[:, :, W // 2: W // 2 + 1]
        wv = vmw & (sw == c)
        nv = wv.float().sum(dim=-1).clamp(min=1)

        mf = fw * wv.float()
        loc_init = mf.sum(dim=-1) / nv
        dsq = ((fw - loc_init.unsqueeze(-1)) ** 2) * wv.float()
        std = (dsq.sum(dim=-1) / nv).sqrt()
        sm = (5.0 * (0.6745 * std).clamp(min=1e-10)).unsqueeze(-1)

        for method_name, method_key in [("biweight", "gpu_mean_bw"), ("welsch", "gpu_mean_we")]:
            loc = loc_init.clone()
            for _ in range(5):
                u = (fw - loc.unsqueeze(-1)) / sm
                if method_name == "biweight":
                    w = ((1.0 - u ** 2) ** 2) * (u.abs() < 1.0).float() * wv.float()
                else:
                    w = torch.exp(-0.5 * u ** 2) * wv.float()
                ws = w.sum(dim=-1).clamp(min=1e-10)
                loc = (fw * w).sum(dim=-1) / ws

            trend_full = torch.full((1, L), float("nan"), dtype=torch.float32, device=device)
            trend_full[0, offset:offset + N_pos] = loc[0]
            det_g = torch.where(
                (trend_full > 0) & vm_c & torch.isfinite(trend_full),
                fl / trend_full,
                torch.tensor(float("nan"), device=device),
            )
            torch.cuda.synchronize()
            rd = recover_depth(det_g[0].cpu().numpy(), t, period, duration)
            if np.isfinite(rd) and rd > 0:
                results[method_key].append(rd)

    # Print results
    row = f"{depth:>7.3f}"
    for key in ["wotan_bw", "tf_sort_bw", "gpu_mean_bw", "gpu_mean_we"]:
        recs = results[key]
        if recs:
            mean_r = np.mean(recs)
            err = abs(mean_r - depth) / depth * 100
            row += f"  {err:>8.1f}%"
        else:
            row += f"  {'N/A':>9}"
    print(row)

print()
print("Lower % = better transit preservation")
print("wotan_bw = wotan biweight (reference)")
print("TF_sort_bw = TorchFlat sort-based biweight (current)")
print("GPU_mean_bw = GPU mean-init biweight (no sort)")
print("GPU_mean_we = GPU mean-init welsch (no sort)")
