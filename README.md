# TorchFlat

**GPU-native photometric preprocessing pipeline for exoplanet transit searches.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

TorchFlat replaces the standard CPU preprocessing workflow (quality filtering, gap handling, sigma clipping, detrending, normalization, and windowing) with a GPU-accelerated pipeline. It uses **UMI** (Unified Median Iterative), a novel asymmetric robust location estimator implemented as a fused HIP/CUDA kernel, to detrend light curves faster and more accurately than existing methods.

## Performance

Benchmarked on AMD Radeon RX 9060 XT (16 GB VRAM) with real TESS sector 6 data (19,618 stars):

| Method | Rate | Full Sector | Speedup |
|--------|------|-------------|---------|
| Celix wotan 12-worker | 4.2 stars/sec | ~78 min | baseline |
| TorchFlat v0.5.0 (hybrid) | 59.3 stars/sec | ~5.5 min | 14.2x |
| **TorchFlat v0.8.0 + UMI kernel** | **87.5 stars/sec** | **~3.7 min** | **20.9x** |

### Transit Depth Recovery Accuracy

Injection-recovery test on 500 real TESS stars, median per-star error (lower = better):

| Depth | wotan biweight | TorchFlat UMI | Winner |
|-------|---------------|---------------|--------|
| 0.1% (super-Earths) | 19.0% | **15.8%** | TorchFlat |
| 0.3% (sub-Neptunes) | 10.5% | **5.7%** | TorchFlat |
| 0.5% (Neptunes) | 4.0% | **2.6%** | TorchFlat |
| 1.0% (hot Jupiters) | **0.5%** | 1.2% | wotan |
| 5.0% (deep transits) | **0.0%** | 0.2% | both perfect |

TorchFlat is more accurate at the transit depths where most detectable planets live (0.1-0.5%).

Validated on 3 TESS sectors (6, 7, 12), 42 confirmed planets, and 1000-star train/test split. Results in `results/`.

## The UMI Algorithm

UMI (Unified Median Iterative) is a three-phase robust location estimator:

1. **Quickselect median** -- exact median via O(n) selection algorithm, computed per-thread on GPU
2. **Quickselect MAD** -- exact median absolute deviation (scale estimate), same kernel call
3. **Asymmetric bisquare iterations** -- weighted location refinement where downward deviations (transit dips) are penalized 1.5x more than upward ones

The asymmetric weight function exploits the fact that **transits are always below the continuum**. Standard biweight treats dips and spikes equally. UMI penalizes dips more aggressively, so the trend stays above the transit and transit depth is preserved.

All three phases run in a single fused GPU kernel call -- median, MAD, and 5 iterations happen per-thread in registers with zero global memory traffic between steps.

When the GPU kernel is not available (no ROCm/CUDA toolkit), UMI falls back to a pure-PyTorch path using histogram-based approximate median + Welsch scale cleaning.

## Installation

```bash
git clone https://github.com/omarkhan2217/TorchFlat.git
cd TorchFlat
pip install -e .
```

**Requirements:** PyTorch >= 2.1.0, NumPy >= 1.24.0, Numba >= 0.57.0, SciPy >= 1.10.0

Works with both **NVIDIA CUDA** and **AMD ROCm** (via PyTorch's unified CUDA API). The UMI kernel compiles automatically on first use via JIT (requires ROCm SDK or CUDA toolkit).

## Quick Start

### Process a TESS sector

```python
import numpy as np
import torchflat

star_data = [
    {
        "time": np.load("star_001_time.npy"),
        "pdcsap_flux": np.load("star_001_pdcsap.npy"),
        "sap_flux": np.load("star_001_sap.npy"),
        "quality": np.load("star_001_quality.npy"),
    }
    # ... for each star in the sector
]

results, skipped = torchflat.preprocess_sector(
    star_data,
    device="cuda",
)

for i, result in enumerate(results):
    if not result:
        continue
    windows = result["windows_2048"]
    trend = result["trend"]
```

### Standalone UMI detrending

```python
import torch
from torchflat import umi_detrend

# flux, time, valid_mask, segment_id are [B, L] tensors on GPU
detrended, trend = umi_detrend(
    flux, time, valid_mask, segment_id,
    window_length_days=0.5,
    asymmetry=1.5,       # dip penalty (1.0 = standard biweight)
)
```

## Architecture

TorchFlat implements two processing tracks:

- **Track A (Transit Search):** Quality filter > gap interpolation > sigma clipping > UMI detrending > normalization > multi-scale window extraction
- **Track B (Anomaly Detection):** Quality filter > gap interpolation > conservative clipping > FFT highpass filter > MAD normalization > fixed-length padding

### UMI kernel

The fused HIP/CUDA kernel (`torchflat/csrc/umi_kernel.cu`) runs one thread per sliding-window position. Each thread:
1. Gathers valid points into a 512-element local buffer
2. Runs quickselect to find the exact median (O(n))
3. Computes absolute deviations and runs quickselect again for exact MAD
4. Re-gathers original values and runs 5 asymmetric bisquare iterations
5. Writes the final location estimate

The kernel compiles via JIT on first import and is cached for subsequent runs.

## Validation

All validation results are saved as JSON in `results/`:

| Validation | Result | File |
|-----------|--------|------|
| Asymmetry train/test split | optimal=1.5, generalizes across held-out stars | `asymmetry_validation.json` |
| Known planet recovery | TorchFlat wins 24/41 (59%) confirmed planets | `known_planet_recovery.json` |
| Multi-sector consistency | UMI wins 10/15 depth-sector combos across sectors 6,7,12 | `multisector_validation.json` |

135/135 unit tests passing.

## Benchmarks

```bash
# Full sector speed benchmark
python benchmarks/bench_real_tess.py --data-dir /path/to/fits/sector_6 --n-stars 19618

# Asymmetry parameter validation
python benchmarks/validate_asymmetry.py

# Known planet recovery
python benchmarks/validate_known_planets.py

# Multi-sector validation
python benchmarks/validate_multisector.py
```

**Note:** Set `$env:TORCHFLAT_NO_KERNEL = "0"` (PowerShell) or `export TORCHFLAT_NO_KERNEL=0` (bash) to enable the UMI kernel.

## API Reference

### Main Entry Points

- **`torchflat.preprocess_sector(star_data, ...)`** -- Full pipeline (Track A + Track B).
- **`torchflat.preprocess_track_a(times, fluxes, qualities, ...)`** -- Track A only.
- **`torchflat.preprocess_track_b(times, sap_fluxes, qualities, ...)`** -- Track B only.
- **`torchflat.umi_detrend(flux, time, valid_mask, segment_id, ...)`** -- Standalone UMI kernel.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device` | `"cuda"` | Torch device |
| `window_length_days` | `0.5` | Sliding window width (days) |
| `asymmetry` | `1.5` | Dip penalty factor (1.0 = standard biweight) |
| `n_iter` | `5` | Number of bisquare iterations |
| `cval` | `5.0` | Rejection threshold in MAD units |
| `skip_track_b` | `False` | Skip Track B (FFT highpass) |
| `window_scales` | 4 scales | `[(256,128), (512,256), (2048,512), (8192,2048)]` |
| `dtype` | `float32` | Computation precision |

## Development

```bash
git clone https://github.com/omarkhan2217/TorchFlat.git
cd TorchFlat
pip install -e ".[dev]"
pytest tests/ -v
```

## Citation

If you use TorchFlat in your research, please cite:

```bibtex
@software{torchflat,
  author = {Khan, Omar},
  title = {TorchFlat: GPU-Accelerated Photometric Preprocessing with UMI Detrending},
  year = {2026},
  url = {https://github.com/omarkhan2217/TorchFlat}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
