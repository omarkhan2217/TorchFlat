# TorchFlat Quickstart

## Installation

```bash
git clone https://github.com/omarkhan2217/TorchFlat.git
cd TorchFlat
pip install -e .
```

Requires: PyTorch >= 2.1.0, NumPy, SciPy. Works with NVIDIA CUDA and AMD ROCm.

## CLI Usage

### Detrend a sector

```bash
torchflat umi_detrend --input /path/to/fits/sector_6/ --output ./results/
```

### Benchmark speed

```bash
torchflat benchmark --input /path/to/fits/sector_6/ --n-stars 500
```

### Skip Track B (faster, transit search only)

```bash
torchflat umi_detrend --input /path/to/fits/sector_6/ --skip-track-b
```

## Python Usage

### Detrend a single star

```python
import numpy as np
import torch
from astropy.io import fits
from torchflat import umi_detrend
from torchflat.gaps import detect_gaps

# Load FITS
with fits.open("tic12345678_s0006.fits") as h:
    d = h[1].data
    time = torch.tensor(d["TIME"], dtype=torch.float64).unsqueeze(0).cuda()
    flux = torch.tensor(d["PDCSAP_FLUX"], dtype=torch.float32).unsqueeze(0).cuda()
    quality = d["QUALITY"]

# Quality mask
valid = torch.ones_like(flux, dtype=torch.bool)
valid[0] = torch.tensor((quality & 3455) == 0) & torch.isfinite(flux[0])

# Gap detection
seg_id, cadence = detect_gaps(time, valid)

# UMI detrend
detrended, trend = umi_detrend(flux, time, valid, seg_id)

# Result: detrended[0] is the normalized light curve
```

### Process an entire sector

```python
import torchflat
from astropy.io import fits
from pathlib import Path

# Load all FITS files
star_data = []
for f in sorted(Path("sector_6/").glob("*.fits")):
    with fits.open(f) as h:
        d = h[1].data
        star_data.append({
            "time": d["TIME"].astype("float64"),
            "pdcsap_flux": d["PDCSAP_FLUX"].astype("float32"),
            "sap_flux": d["SAP_FLUX"].astype("float32"),
            "quality": d["QUALITY"].astype("int32"),
        })

# Process all stars on GPU
results, skipped = torchflat.preprocess_sector(star_data, device="cuda")

# Access results
for i, r in enumerate(results):
    if r:
        trend = r["trend"]           # stellar variability
        windows = r["windows_2048"]  # transit search windows
```

### Custom asymmetry parameter

```python
# Standard biweight (symmetric, same as wotan)
detrended, trend = umi_detrend(flux, time, valid, seg, asymmetry=1.0)

# Aggressive transit preservation (for shallow planet searches)
detrended, trend = umi_detrend(flux, time, valid, seg, asymmetry=2.0)

# Default (balanced, validated on train/test split)
detrended, trend = umi_detrend(flux, time, valid, seg, asymmetry=1.5)
```

## UMI Kernel

The fused HIP/CUDA kernel compiles automatically on first use. To check:

```python
from torchflat._kernel_loader import _get_umi_kernel
kern = _get_umi_kernel()
print("Kernel loaded" if kern else "Using fallback")
```

If the kernel doesn't load:
- Ensure `TORCHFLAT_NO_KERNEL` is not set to `1`
- ROCm SDK or CUDA toolkit must be installed for compilation
- The fallback (torch.sort) produces identical results, just slower
