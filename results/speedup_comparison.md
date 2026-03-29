# Speedup Comparison (Honest Numbers)

Benchmarked on AMD Radeon RX 9060 XT + 12-core CPU, real TESS sector 6 data.

## Full pipeline comparison (apples-to-apples)

Both pipelines do the same work: quality filter, gap handling, sigma
clipping, biweight-family detrending, normalization, multi-scale window
extraction (4 scales), and anomaly detection (Track B).

| Method | Rate | Sector (19,618 stars) | Speedup |
|--------|------|----------------------|---------|
| wotan full pipeline (12 workers) | 4.2/sec | 78 min | baseline |
| TorchFlat v0.5.0 (hybrid GPU+CPU) | 59.3/sec | 5.5 min | 14.2x |
| **TorchFlat v0.8.0 (GPU + UMI kernel)** | **95.9/sec** | **3.4 min** | **22.8x** |

## Detrend-only comparison

Isolates the detrending algorithm (wotan biweight vs TorchFlat UMI),
no pipeline overhead.

| Method | Rate | Sector | Speedup |
|--------|------|--------|---------|
| wotan biweight (1 thread) | 4.9/sec | 66 min | baseline |
| wotan biweight (12 workers) | 22.4/sec | 15 min | 4.5x |
| **TorchFlat UMI (GPU kernel)** | **115.7/sec** | **2.8 min** | **5.2x vs 12-worker** |

Note: wotan 12-worker achieves only 4.5x scaling (not 12x) due to
Numba JIT recompilation overhead in spawned processes on Windows.

## Why the full pipeline speedup (22.8x) is larger than detrend-only (5.2x)

The wotan full pipeline (4.2/sec) is much slower than wotan detrend-only
(22.4/sec) because the pipeline adds per-star Python overhead: gap
detection with interpolation, sigma clipping with rolling median,
normalization, and sliding window extraction at 4 scales - all in
serial Python loops.

TorchFlat does all of these steps as batched GPU tensor operations,
eliminating the per-star Python loop overhead entirely. The GPU
processes 50 stars simultaneously per batch.

## How to report in a paper

"TorchFlat processes a full TESS sector (19,618 stars) in 3.4 minutes
on a single AMD Radeon RX 9060 XT GPU - 22.8x faster than a 12-worker
CPU pipeline performing the same preprocessing steps (quality filtering,
gap handling, sigma clipping, detrending, normalization, multi-scale
window extraction, and FFT anomaly detection). For the detrending step
in isolation, TorchFlat's UMI kernel achieves 115.7 stars/sec - 5.2x
faster than 12-worker parallel wotan biweight (22.4 stars/sec)."
