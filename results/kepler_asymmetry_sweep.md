# Kepler Asymmetry Sweep

1000 Kepler stars, injection recovery at 0.1% and 0.5% depth.
Period=10d, duration=5h.

## Results

| a | 0.1% error | 0.5% error |
|---|-----------|-----------|
| 2.0 | 10.7% | **1.1%** |
| 2.2 | 10.0% | 1.2% |
| 2.4 | 9.2% | 1.3% |
| 2.6 | 8.6% | 1.3% |
| 2.8 | 8.4% | 1.4% |
| **3.0** | **8.2%** | 1.4% |
| 3.2 | 8.1% | 1.4% |
| 3.4 | 8.0% | 1.5% |
| 4.0 | 8.5% | 1.7% |
| 5.0 | 8.5% | 1.8% |
| 7.0 | 8.2% | 1.8% |
| 10.0 | 7.2% | 1.7% |

## Recommendations by mission

| Mission | Noise level | Optimal a | 0.1% error | Bias |
|---------|-----------|-----------|-----------|------|
| TESS | ~1000 ppm | 2.0 | 12.7% | -451 ppm |
| Kepler | ~100 ppm | 3.0 | 8.2% | -247 ppm |
| K2 | ~300 ppm | 2.0-2.5 | (not tested) | (not tested) |
| PLATO | ~50 ppm | 3.0+ | (future) | (future) |

Higher precision data benefits from higher asymmetry because transit
dips are more clearly separated from noise, allowing more aggressive
downweighting without over-rejecting legitimate noise points.
