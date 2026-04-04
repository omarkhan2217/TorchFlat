"""Re-run known planet recovery using umi_detrend wrapper (proper min_segment)."""
import csv, io, json, urllib.request, numpy as np, torch, wotan, time
from pathlib import Path
from astropy.io import fits
from torchflat.umi import umi_detrend

device = torch.device("cuda")
kepler_dir = Path("D:/TorchFlat/data/kepler")
toi_file = Path("C:/Users/USER/Desktop/Website/trips-section/Celix/data/catalogs/toi_catalog.csv")
base = Path("C:/Users/USER/Desktop/Website/trips-section/Celix/data/fits")
RESULTS_DIR = Path(__file__).parent.parent / "results"
TESS_BJD_OFFSET = 2457000.0

# Index files
fits_index = {}
for d in base.iterdir():
    if d.is_dir() and d.name.startswith("sector"):
        for f in d.glob("*.fits"):
            tic = f.stem.split("_")[0].replace("tic", "")
            fits_index[tic] = str(f)
kic_files = {}
for f in kepler_dir.iterdir():
    if f.suffix == ".fits":
        kic = f.stem.split("-")[0].replace("kplr", "").lstrip("0")
        kic_files[kic] = str(f)
print(f"TESS: {len(fits_index)}, Kepler: {len(kic_files)}")

# TESS planets
tess_planets = []; seen = set()
with open(toi_file, "r", encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("TESS Disposition", "") != "KP" and row.get("TFOPWG Disposition", "") != "KP":
            continue
        tic = row.get("TIC ID", "").strip()
        if tic not in fits_index or tic in seen:
            continue
        try:
            d_ppm = float(row["Depth (ppm)"]); p = float(row["Period (days)"])
            dur = float(row["Duration (hours)"]); epoch = float(row["Epoch (BJD)"])
        except:
            continue
        seen.add(tic)
        tess_planets.append({"id": tic, "name": row.get("Planet Name", "") or row.get("TOI", ""),
            "depth_ppm": d_ppm, "period": p, "duration": dur, "epoch": epoch,
            "fits": fits_index[tic], "mission": "tess", "qual_col": "QUALITY", "bitmask": 3455})

# Kepler planets
kepler_planets = []
url = ("https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
       "?table=cumulative&where=koi_disposition=%27CONFIRMED%27"
       "&select=kepid,kepler_name,koi_period,koi_depth,koi_duration,koi_time0bk&format=csv")
print("Downloading Kepler catalog...")
with urllib.request.urlopen(url, timeout=30) as resp:
    text = resp.read().decode("utf-8")
for row in csv.DictReader(io.StringIO(text)):
    kic = str(row.get("kepid", "")).strip()
    if kic not in kic_files:
        continue
    try:
        d_ppm = float(row["koi_depth"]); p = float(row["koi_period"])
        dur = float(row["koi_duration"]); epoch = float(row["koi_time0bk"])
    except:
        continue
    kepler_planets.append({"id": kic, "name": row.get("kepler_name", ""),
        "depth_ppm": d_ppm, "period": p, "duration": dur, "epoch": epoch,
        "fits": kic_files[kic], "mission": "kepler", "qual_col": "SAP_QUALITY", "bitmask": 0})

all_planets = tess_planets + kepler_planets
print(f"Total: {len(tess_planets)} TESS + {len(kepler_planets)} Kepler = {len(all_planets)}")


def recover(det, t_v, pl):
    finite = np.isfinite(det)
    if finite.sum() < 50:
        return np.nan
    epoch = pl["epoch"] - TESS_BJD_OFFSET if pl["mission"] == "tess" else pl["epoch"]
    phase = ((t_v[finite] - epoch) % pl["period"]) / pl["period"]
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    hd = (pl["duration"] / 24.0) / pl["period"] / 2.0
    in_tr = np.abs(phase) < hd
    out_tr = ~in_tr
    if in_tr.sum() < 2 or out_tr.sum() < 20:
        return np.nan
    return float(np.median(det[finite][out_tr]) - np.median(det[finite][in_tr]))


results = []
t0 = time.perf_counter()
for i, pl in enumerate(all_planets):
    try:
        with fits.open(pl["fits"]) as h:
            dd = h[1].data
            t = np.array(dd["TIME"], dtype=np.float64)
            fl = np.array(dd["PDCSAP_FLUX"], dtype=np.float64)
            q = np.array(dd[pl["qual_col"]], dtype=np.int32)
    except Exception:
        continue

    if pl["bitmask"] > 0:
        v = ((q & pl["bitmask"]) == 0) & np.isfinite(fl) & np.isfinite(t)
    else:
        v = (q == 0) & np.isfinite(fl) & np.isfinite(t)
    if v.sum() < 100:
        continue
    t_v, f_v = t[v], fl[v]
    pub = pl["depth_ppm"] / 1e6

    r = {"name": pl["name"], "mission": pl["mission"], "depth_ppm": pl["depth_ppm"]}

    # UMI (wrapper with proper min_segment)
    L = len(f_v)
    ft = torch.tensor(f_v, dtype=torch.float32, device=device).unsqueeze(0)
    tt = torch.tensor(t_v, dtype=torch.float64, device=device).unsqueeze(0)
    vt = torch.ones(1, L, dtype=torch.bool, device=device)
    st = torch.zeros(1, L, dtype=torch.int32, device=device)
    det, _ = umi_detrend(ft, tt, vt, st, window_length_days=0.5, asymmetry=2.0)
    det_np = det[0].cpu().numpy()
    torch.cuda.synchronize()
    rec = recover(det_np, t_v, pl)
    if np.isfinite(rec) and rec > 0 and pub > 0:
        r["UMI_err"] = round(abs(rec - pub) / pub * 100, 1)
    else:
        r["UMI_err"] = None

    # Wotan methods
    for method in ["biweight", "welsch"]:
        try:
            flat, _ = wotan.flatten(t_v, f_v, method=method, window_length=0.5, return_trend=True)
            rec = recover(flat, t_v, pl)
            if np.isfinite(rec) and rec > 0 and pub > 0:
                r[f"{method}_err"] = round(abs(rec - pub) / pub * 100, 1)
            else:
                r[f"{method}_err"] = None
        except Exception:
            r[f"{method}_err"] = None

    # Savgol (skip -- too slow and bad anyway)
    r["savgol_err"] = None

    results.append(r)
    if (i + 1) % 100 == 0:
        elapsed = time.perf_counter() - t0
        print(f"  {i+1}/{len(all_planets)} processed, {len(results)} valid, {elapsed:.0f}s", flush=True)

# Summary
methods = ["UMI", "biweight", "welsch"]
valid = [r for r in results if all(r.get(f"{m}_err") is not None for m in methods)]
print(f"\n{len(valid)} planets with all 3 methods")

wins = {m: 0 for m in methods}
tess_wins = {m: 0 for m in methods}
kep_wins = {m: 0 for m in methods}
for r in valid:
    errs = {m: r[f"{m}_err"] for m in methods}
    w = min(errs, key=errs.get)
    wins[w] += 1
    if r["mission"] == "tess":
        tess_wins[w] += 1
    else:
        kep_wins[w] += 1

for m in methods:
    errs = [r[f"{m}_err"] for r in valid]
    print(f"  {m}: median={np.median(errs):.1f}%, wins={wins[m]}")

n_tess = sum(1 for r in valid if r["mission"] == "tess")
n_kep = sum(1 for r in valid if r["mission"] == "kepler")
print(f"\nTESS ({n_tess}): {tess_wins}")
print(f"Kepler ({n_kep}): {kep_wins}")
print(f"Overall: {wins}")

doc = {"title": "Known Planet Recovery (umi_detrend wrapper, a=2.0)",
       "n_tess": n_tess, "n_kepler": n_kep, "n_valid": len(valid),
       "wins": wins, "per_planet": results}
with open(RESULTS_DIR / "known_planet_recovery_all.json", "w") as f:
    json.dump(doc, f, indent=2)
print("Saved results/known_planet_recovery_all.json")
