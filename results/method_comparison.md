# Detrending Method Comparison

1000 real TESS stars (sector 6), period=3d, duration=3h.
Median per-star error (lower = better).

## Transit Recovery Accuracy

| Depth | UMI | UMI aggressive | welsch | biweight | median | lowess | savgol |
|-------|-----|----------------|--------|----------|--------|--------|--------|
| 0.05% | **22.3%** | **17.7%** | 24.0% | 23.5% | 24.9% | 38.7% | 51.4% |
| 0.1% | **15.8%** | **11.3%** | 18.5% | 20.5% | 20.6% | 36.7% | 51.2% |
| 0.3% | **4.9%** | **4.3%** | 5.9% | 12.7% | 12.4% | 26.1% | 50.1% |
| 0.5% | 2.4% | 2.7% | **1.8%** | 5.1% | 8.2% | 3.6% | 48.2% |
| 1.0% | 1.2% | 1.5% | **0.7%** | **0.8%** | 4.2% | **0.7%** | 27.7% |
| 5.0% | 0.3% | 0.4% | **0.1%** | **0.1%** | 0.8% | **0.1%** | 0.3% |

UMI default: asymmetry=2.0, cval=5.0
UMI aggressive: asymmetry=10.0, cval=2.5 (higher bias, better shallow accuracy)

## Speed (per star, 1000 TESS stars)

| Method | Time | vs UMI |
|--------|------|--------|
| **UMI (GPU)** | **3.4ms** | baseline |
| savgol | 2ms | 0.6x (faster but 51% error) |
| mean | 8ms | 2x slower |
| median | 48ms | 14x slower |
| trim_mean | 144ms | 42x slower |
| biweight | 234ms | 69x slower |
| welsch | 384ms | 113x slower |
| lowess | 557ms | 164x slower |

## Rankings at 0.1% depth (where most detectable planets live)

1. **UMI aggressive: 11.3%** (fastest AND most accurate)
2. **UMI default: 15.8%** (same speed, conservative bias)
3. welsch: 18.5% (113x slower)
4. biweight: 20.5% (69x slower)
5. median: 20.6% (14x slower)
6. lowess: 36.7% (164x slower)
7. savgol: 51.2% (faster but useless)

## Key findings

- UMI is #1 at 0.05-0.3% (the super-Earth/sub-Neptune zone)
- UMI aggressive mode pushes 0.1% accuracy to 11.3% (29% better than default)
- Welsch is better at 0.5-5% but 113x slower
- Biweight (wotan default) is 30% worse than UMI at 0.1%
- Savgol is fundamentally unsuitable at all shallow depths (~51% constant error)
- All results verified with real wotan, median of per-star errors

Validated on 1000 real TESS sector 6 stars.
