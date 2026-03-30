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
| **TorchFlat v0.9.1 + UMI kernel** | **154 stars/sec** | **~2.1 min** | **37x** |

### Transit Depth Recovery Accuracy

Injection-recovery test on 200 real TESS stars, median per-star error (lower = better):

| Depth | wotan biweight | TorchFlat UMI | Winner |
|-------|---------------|---------------|--------|
| 0.1% (super-Earths) | 19.8% | **14.7%** | TorchFlat |
| 0.3% (sub-Neptunes) | 8.4% | **3.7%** | TorchFlat |
| 0.5% (Neptunes) | 1.5% | **1.5%** | tie |
| 1.0% (hot Jupiters) | **0.4%** | 0.7% | wotan |
| 5.0% (deep transits) | **0.0%** | 0.1% | both perfect |

TorchFlat is more accurate at the transit depths where most detectable planets live (0.1-0.5%).

Validated on 3 TESS sectors (6, 7, 12), Kepler, K2, 42 confirmed planets, and 10,000-star parameter validation. Results in `results/`.

## The UMI Algorithm

UMI (Unified Median Iterative) is a three-phase robust location estimator:

1. **Quickselect median** -- exact median via O(n) selection algorithm, computed per-thread on GPU
2. **Upper-RMS scale** -- RMS of points above the median only. Transit dips never contaminate the scale estimate, giving a tighter and more accurate noise measurement than standard MAD
3. **Asymmetric bisquare iterations** -- weighted location refinement where downward deviations (transit dips) are penalized 2x more than upward ones

The asymmetric weight function exploits the fact that **transits are always below the continuum**. Standard biweight treats dips and spikes equally. UMI penalizes dips more aggressively, so the trend stays above the transit and transit depth is preserved.

All three phases run in a single fused GPU kernel call -- median, upper-RMS, and 5 iterations happen per-thread with zero global memory traffic between steps.

When the GPU kernel is not available (no ROCm/CUDA toolkit), UMI falls back to a pure-PyTorch path using torch.sort for median + upper-RMS scale.

## Installation

```bash
pip install torchflat
```

Or from source:
```bash
git clone https://github.com/omarkhan2217/TorchFlat.git
cd TorchFlat
pip install -e .
```

**Requirements:** PyTorch >= 2.1.0, NumPy >= 1.24.0, SciPy >= 1.10.0

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
    asymmetry=2.0,       # 2.0=best accuracy, 1.0=variable stars, 1.5=mixed
)
```

## Architecture

TorchFlat implements two processing tracks:

- **Track A (Transit Search):** Quality filter > gap interpolation > sigma clipping > UMI detrending > normalization > multi-scale window extraction
- **Track B (Anomaly Detection):** Quality filter > gap interpolation > conservative clipping > FFT highpass filter > MAD normalization > fixed-length padding

### UMI kernel

The direct HIP/CUDA kernel (`torchflat/csrc/umi_kernel.cu`) runs one thread per (star, window position) pair. Each thread reads directly from the raw `[B, L]` flux array, no unfold or tensor copies needed:

1. Reads W values from raw flux, checks segment validity inline
2. Quickselect for exact median (O(n))
3. Upper-RMS scale from above-median points (no sort needed)
4. 5 asymmetric bisquare iterations
5. Writes the final location estimate

VRAM usage: 319 MB for a 50-star batch. The kernel compiles via JIT on first import and is cached for subsequent runs.

## Validation

All validation results are saved as JSON in `results/`:

| Validation | Result | File |
|-----------|--------|------|
| Asymmetry train/test split | optimal=2.0, validated on 2000+10,000 stars | `asymmetry_validation_2k.json` |
| Known planet recovery | TorchFlat wins 24/41 (59%) confirmed planets | `known_planet_recovery.json` |
| Multi-sector consistency | UMI wins 9/15 across sectors 6, 7, 12 (2000 stars each) | `multisector_validation_2k.json` |
| Multi-mission | Kepler: 10.5% vs wotan 36.6% at 0.1%. K2: 4.5% vs 46.6% at 0.5% | `multi_mission.json` |
| Method comparison | UMI #1 at 0.1% vs 8 methods (biweight, welsch, lowess, etc.) | `method_comparison.md` |

135/135 unit tests passing.

## CLI

```bash
# Detrend a TESS sector
torchflat umi_detrend --input /path/to/fits/

# Detrend a single star
torchflat umi_detrend --input star.fits --output-format fits

# Kepler data
torchflat umi_detrend --input /path/to/kepler/ --mission kepler

# Plot a star
torchflat plot --fits star.fits --save output.png

# Speed benchmark
torchflat benchmark --input /path/to/fits/ --n-stars 500
```

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
| `asymmetry` | `2.0` | Dip penalty: 2.0 (quiet stars), 1.5 (mixed), 1.0 (variable stars) |
| `n_iter` | `5` | Number of bisquare iterations |
| `cval` | `5.0` | Rejection threshold in MAD units |
| `skip_track_b` | `False` | Skip Track B (FFT highpass) |
| `window_scales` | 4 scales | `[(256,128), (512,256), (2048,512), (8192,2048)]` |
| `dtype` | `float32` | Computation precision |

## Limitations

- **CUDA untested.** The HIP kernel is validated on AMD ROCm (RX 9060 XT). The CUDA compilation path exists but has not been tested on NVIDIA GPUs. The torch.sort fallback works on any device.
- **Fallback is slower.** Without the compiled HIP/CUDA kernel, UMI uses torch.sort (6x slower). Install the ROCm or CUDA toolkit to enable the fused kernel.
- **Asymmetry bias.** The default asymmetry=2.0 introduces a -451 ppm bias on flat stars. This is below TESS noise (~1000 ppm) but may matter for population-level radius studies. Use `--bias-correct` to remove it, or `--asymmetry 1.0` for zero bias.
- **Variable star bias.** On stars with >1% variability, asymmetry=2.0 causes -7240 ppm bias. Use `--asymmetry 1.0` for variable stars.
- **8-hour transits.** Both UMI and wotan fail on transits longer than ~5 hours with the default 0.5-day window. Use `--window-length 1.5` for long-duration transits.

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
