# Speedup Comparison (Final Numbers)

Benchmarked on AMD Radeon RX 9060 XT + 12-core CPU, 500 real TESS sector 6 stars.

## Full pipeline comparison (apples-to-apples)

Both pipelines do the same work: quality filter, gap handling, sigma
clipping, detrending, normalization, multi-scale window extraction
(4 scales), and anomaly detection (Track B).

| Method | Rate | Sector (19,618 stars) | Speedup |
|--------|------|----------------------|---------|
| wotan single-thread | 4.9/sec | 66 min | baseline |
| wotan full pipeline (12 workers) | 4.2/sec | 78 min | 0.9x |
| TorchFlat v0.5.0 (hybrid GPU+CPU) | 59.3/sec | 5.5 min | 14.2x |
| **TorchFlat v0.8.0 (direct kernel + upper-RMS)** | **154/sec** | **2.1 min** | **37x** |

## Detrend-only comparison

| Method | Rate | Sector | Speedup |
|--------|------|--------|---------|
| wotan biweight (1 thread) | 4.9/sec | 66 min | baseline |
| wotan biweight (12 workers) | 22.4/sec | 15 min | 4.5x |
| **TorchFlat UMI (GPU kernel)** | **248/sec** | **1.3 min** | **11x vs 12-worker** |

## Per-star speed vs all methods

| Method | Time | vs UMI |
|--------|------|--------|
| **UMI (GPU)** | **3.8ms** | baseline |
| mean | 9ms | 2x slower |
| median | 57ms | 15x slower |
| trim_mean | 167ms | 44x slower |
| biweight | 310ms | 82x slower |
| welsch | 473ms | 124x slower |
| lowess | 521ms | 137x slower |
| huber | ~3000ms | ~789x slower |

## Multi-mission speed

| Mission | UMI rate | wotan rate | Speedup |
|---------|----------|------------|---------|
| TESS | 154/sec | 4.2/sec | 37x |
| Kepler | 564/sec | 69.9/sec | 8.1x |
| K2 | 2112/sec | 87.5/sec | 24x |

## VRAM usage

319 MB for a 50-star batch. Runs on any GPU with 1+ GB VRAM.
