# TorchFlat

**GPU-native photometric preprocessing pipeline for exoplanet transit searches.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

TorchFlat replaces the standard CPU preprocessing workflow (quality filtering, gap handling, sigma clipping, biweight detrending, normalization, and windowing) with a hybrid GPU+CPU pipeline. It uses batched PyTorch tensor operations on GPU and a Numba JIT biweight kernel on CPU, running both simultaneously for maximum throughput.

## Performance

Benchmarked on AMD Radeon RX 9060 XT (16 GB VRAM) + 12 CPU threads with real TESS sector 6 data (19,618 stars):

| Method | Rate | Full Sector |
|--------|------|-------------|
| wotan single-thread | ~1 star/sec | ~5.4 hours |
| Celix wotan 12-worker | 4.2 stars/sec | ~78 min |
| TorchFlat GPU-only | 35.8 stars/sec | ~9.1 min |
| **TorchFlat Hybrid (GPU+CPU)** | **59.3 stars/sec** | **~5.5 min** |

14.2x faster than Celix 12-worker CPU pipeline. TorchFlat does more work than wotan alone: the timing includes quality filtering, gap interpolation, sigma clipping, biweight detrending, normalization, window extraction (Track A), and FFT highpass anomaly detection (Track B). Track B can optionally be disabled with `skip_track_b=True`.

Precision: p99 relative error < 2.1e-5 vs GPU path, < 6.8e-6 vs wotan reference. 135/135 tests passing.

## Installation

```bash
pip install torchflat
```

**Requirements:** PyTorch >= 2.1.0, NumPy >= 1.24.0, Numba >= 0.57.0, SciPy >= 1.10.0

Works with both **NVIDIA CUDA** and **AMD ROCm** (via PyTorch's unified CUDA API).

## Quick Start

### Hybrid mode (recommended, fastest)

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

# Process entire sector using GPU + CPU simultaneously
results, skipped = torchflat.preprocess_sector_hybrid(
    star_data,
    gpu_device="cuda",       # GPU for batched tensor ops
    cpu_fraction=0.65,       # 65% of stars on CPU (Numba), 35% on GPU
    cpu_workers=12,          # Number of CPU worker processes
)

# Access results
for i, result in enumerate(results):
    if not result:
        continue
    windows = result["windows_2048"]          # [N_windows, 2048] transit search windows
    trend = result["trend"]                   # Estimated stellar trend
    anomaly_curve = result.get("track_b_curve")  # Anomaly detection curve (if Track B enabled)
```

### GPU-only mode

```python
results, skipped = torchflat.preprocess_sector(
    star_data,
    device="cuda",
    vram_budget_gb=12.0,
)
```

### Skip Track B for transit-search-only

```python
results, skipped = torchflat.preprocess_sector(
    star_data,
    device="cuda",
    skip_track_b=True,
)
```

### Standalone biweight kernel

```python
import torch
from torchflat import biweight_detrend

# flux, time, valid_mask, segment_id are [B, L] tensors on GPU
detrended, trend = biweight_detrend(
    flux, time, valid_mask, segment_id,
    window_length_days=0.5,
    dtype=torch.float32,
)
```

## Architecture

TorchFlat implements two processing tracks:

- **Track A (Transit Search):** Quality filter > gap interpolation > sigma clipping > biweight detrending > normalization > multi-scale window extraction
- **Track B (Anomaly Detection):** Quality filter > gap interpolation > conservative clipping > FFT highpass filter > MAD normalization > fixed-length padding

### Hybrid CPU+GPU design

In hybrid mode, stars are split between GPU and CPU:

- **GPU path** (`torchflat/biweight.py`): Batched tensor operations using `torch.sort` for masked median. Processes multiple stars simultaneously in VRAM.
- **CPU path** (`torchflat/biweight_cpu.py`): Numba JIT kernel using `numpy.median` (quickselect, O(n) vs GPU's O(n log n) sort). Runs 12+ worker processes in parallel.

Both paths produce identical results (p99 error < 2.1e-5). The optimal split (default `cpu_fraction=0.65`) was determined by benchmarking on real TESS data.

### Batching strategy

Stars are batched by post-filter length into buckets, minimizing padding waste. Batch sizes are computed dynamically from available VRAM (capped at 90 stars to prevent VRAM spill to shared memory). The biweight kernel uses a segment-aware masked median to prevent trend estimation from crossing data gaps.

## API Reference

### Main Entry Points

- **`torchflat.preprocess_sector_hybrid(star_data, ...)`** - Hybrid CPU+GPU processing (fastest).
- **`torchflat.preprocess_sector(star_data, ...)`** - GPU-only or CPU-only processing.
- **`torchflat.preprocess_track_a(times, fluxes, qualities, ...)`** - Track A only (biweight detrending).
- **`torchflat.preprocess_track_b(times, sap_fluxes, qualities, ...)`** - Track B only (FFT highpass).
- **`torchflat.biweight_detrend(flux, time, valid_mask, segment_id, ...)`** - Standalone GPU biweight kernel.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device` / `gpu_device` | `"cuda"` | Torch device (`"cuda"` for GPU, `"cpu"` for CPU) |
| `cpu_fraction` | `0.65` | Fraction of stars for CPU workers (hybrid mode) |
| `cpu_workers` | `12` | Number of CPU worker processes (hybrid mode) |
| `vram_budget_gb` | Auto-detect | Available VRAM budget in GB |
| `max_batch` | Auto (cap 90) | Override batch size directly |
| `window_scales` | 4 scales | `[(256,128), (512,256), (2048,512), (8192,2048)]` |
| `window_length_days` | `0.5` | Biweight sliding window width (days) |
| `dtype` | `float32` | Computation precision (`torch.float32` or `torch.float64`) |
| `skip_track_b` | `False` | Skip Track B (FFT highpass) |

## Benchmarks

```bash
# Full benchmark suite
python benchmarks/bench_vs_wotan.py

# Individual benchmarks
python benchmarks/bench_vs_wotan.py --bench biweight
python benchmarks/bench_vs_wotan.py --bench scaling
python benchmarks/bench_vs_wotan.py --bench precision

# Real TESS data benchmark
python benchmarks/bench_real_tess.py --data-dir /path/to/fits/sector_6 --n-stars 1000

# GPU profiler
python benchmarks/profile_pipeline.py
```

## Development

```bash
git clone https://github.com/omarkhan2217/TorchFlat.git
cd TorchFlat
pip install -e ".[dev]"
pytest tests/ -v
```

## Roadmap

- **v0.5.0** (current): Hybrid CPU+GPU pipeline, Numba JIT CPU kernel, 14.2x faster than Celix
- **v0.6.0**: Custom HIP kernel for O(n) masked quickselect on GPU (targeting 100+ stars/sec)
- **v1.0.0**: Celix integration, full validation on multi-sector data

## Citation

If you use TorchFlat in your research, please cite:

```bibtex
@software{torchflat,
  author = {Khan, Omar},
  title = {TorchFlat},
  year = {2026},
  url = {https://github.com/omarkhan2217/TorchFlat}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
