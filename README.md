# TorchFlat

**GPU-native photometric preprocessing pipeline for exoplanet transit searches.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

TorchFlat replaces the standard CPU preprocessing workflow &mdash; quality filtering, gap handling, sigma clipping, biweight detrending, normalization, and windowing &mdash; with a single batched GPU pass via pure PyTorch tensor operations. It processes entire TESS sectors (15,000+ stars) in minutes instead of hours on consumer GPUs.

## Installation

```bash
pip install torchflat
```

**Requirements:** PyTorch >= 2.1.0, NumPy >= 1.24.0

Works with both **NVIDIA CUDA** and **AMD ROCm** (via PyTorch's unified CUDA API).

## Quick Start

```python
import numpy as np
import torchflat

# Prepare your TESS sector data as a list of dicts
star_data = [
    {
        "time": np.load("star_001_time.npy"),
        "pdcsap_flux": np.load("star_001_pdcsap.npy"),
        "sap_flux": np.load("star_001_sap.npy"),
        "quality": np.load("star_001_quality.npy"),
    }
    # ... for each star in the sector
]

# Process entire sector on GPU
results, skipped = torchflat.preprocess_sector(
    star_data,
    device="cuda",           # Works for both NVIDIA and AMD GPUs
    vram_budget_gb=12.0,     # Adjust for your GPU
)

# Access results
for i, result in enumerate(results):
    if not result:
        continue
    windows = result["windows_2048"]          # [N_windows, 2048] transit search windows
    trend = result["trend"]                   # Estimated stellar trend
    anomaly_curve = result["track_b_curve"]   # Anomaly detection curve
```

## API Reference

### Main Entry Points

**`torchflat.preprocess_sector(star_data, ...)`** &mdash; Full sector preprocessing (Track A + Track B).

**`torchflat.preprocess_track_a(times, fluxes, qualities, ...)`** &mdash; PDCSAP flux to transit search windows via biweight detrending.

**`torchflat.preprocess_track_b(times, sap_fluxes, qualities, ...)`** &mdash; SAP flux to anomaly detection curves via FFT highpass filtering.

**`torchflat.biweight_detrend(flux, time, valid_mask, segment_id, ...)`** &mdash; Standalone biweight detrending kernel.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device` | `"cuda"` | Torch device (`"cuda"` for GPU, `"cpu"` for CPU) |
| `vram_budget_gb` | Auto-detect | Available VRAM budget in GB |
| `max_batch` | Auto | Override batch size directly |
| `window_scales` | 4 scales | `[(256,128), (512,256), (2048,512), (8192,2048)]` |
| `window_length_days` | `0.5` | Biweight sliding window width (days) |
| `dtype` | `float32` | Computation precision (`torch.float32` or `torch.float64`) |

## How It Works

TorchFlat implements two processing tracks:

- **Track A (Transit Search):** Quality filter &rarr; gap interpolation &rarr; sigma clipping &rarr; biweight detrending &rarr; normalization &rarr; multi-scale window extraction
- **Track B (Anomaly Detection):** Quality filter &rarr; gap interpolation &rarr; conservative clipping &rarr; FFT highpass filter &rarr; MAD normalization &rarr; fixed-length padding

Stars are batched by post-filter length into buckets, minimizing padding waste. Batch sizes are computed dynamically from available VRAM. The biweight kernel uses a segment-aware masked median to prevent trend estimation from crossing data gaps.

See [TORCHFLAT-DESIGN.md](TORCHFLAT-DESIGN.md) for the full design document.

## Benchmarks

Run benchmarks with:

```bash
python benchmarks/bench_vs_wotan.py
python benchmarks/bench_vs_wotan.py --bench biweight
python benchmarks/bench_vs_wotan.py --bench scaling
```

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
  title = {TorchFlat: GPU-accelerated photometric preprocessing for large-scale exoplanet transit searches},
  year = {2026},
  url = {https://github.com/omarkhan2217/TorchFlat}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
