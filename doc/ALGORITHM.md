# UMI Algorithm: Unified Median Iterative Detrending

## Overview

UMI is a robust location estimator designed for photometric time-series
detrending. It removes stellar variability from light curves while
preserving transit signals. It runs as a fused GPU kernel (HIP/CUDA)
with a pure-PyTorch fallback.

## The Problem

Exoplanet transit detection requires removing slow stellar variability
(the "trend") from light curves. The trend estimator must:

1. Track slow variations (timescale > 6 hours)
2. Reject fast outliers (flares, cosmic rays)
3. **Not absorb transit dips** (timescale 1-8 hours, depth 0.01-5%)

Standard biweight detrending (wotan) treats upward and downward
outliers equally. But transit dips are always downward. UMI exploits
this asymmetry.

## Algorithm

For each sliding window position (W ~ 361 points, 0.5 days):

### Phase 1: Exact Median (location init)

Quickselect (O(n)) finds the exact sample median. This is a rank
statistic - immune to outlier contamination regardless of depth.

### Phase 2: Upper-RMS Scale

Compute RMS of only the points ABOVE the median. Transit dips (below
median) never contaminate the scale estimate. This gives a tighter
and more accurate noise measurement than standard MAD, which uses
all deviations including transit-contaminated ones.

    upper_rms = sqrt(mean((x - median)^2 for x > median))
    scale = upper_rms * 0.6745   (convert to MAD-equivalent)

No extra sort needed - just a sum-of-squares loop over the buffer.

### Phase 3: Asymmetric Bisquare Iterations

Refine the location using the UMI weight function:

```
u = (flux - location) / (cval * scale)

if u < 0:  u_eff = u * asymmetry    (penalize downward dips)
else:      u_eff = u                 (standard for upward)

if |u_eff| < 1:  w = (1 - u_eff^2)^2
else:            w = 0               (rejected)

location = sum(w * flux) / sum(w)
```

Repeat for `n_iter` iterations (default 5).

**Why asymmetric:** A transit dip at `u = -0.3` gets:
- Standard biweight: weight 0.83 (nearly full - dip absorbed into trend)
- UMI (asymmetry=1.5): weight 0.64 (reduced - dip preserved)

The trend stays above the transit. Transit depth is preserved.

### Direct GPU Kernel

All three phases run in a single kernel call. One GPU thread per
(star, window position) pair. Reads directly from raw [B, L] flux
arrays, no unfold or tensor copies needed.

1. Read W values from raw flux array, check segment validity inline
2. Quickselect for median (in local buffer, O(n))
3. Upper-RMS of above-median points (sum-of-squares, no sort)
4. 5 asymmetric bisquare iterations (in local buffer, no memory traffic)
5. Write final location (one write)

No unfold, no [B, N_pos, W] tensor allocation. VRAM usage: 319 MB
for a full 50-star batch.

## Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `asymmetry` | 1.5 | Higher = more transit preservation, slight upward bias. 1.0 = standard biweight. |
| `cval` | 5.0 | Rejection threshold in MAD units. Higher = less rejection. |
| `n_iter` | 5 | Bisquare iterations. 5 is sufficient for convergence. |
| `window_length_days` | 0.5 | Window width. Shorter = tracks faster variability but absorbs more transit. |

## Comparison to Wotan Biweight

| Property | wotan biweight | UMI |
|----------|---------------|-----|
| Weight function | Symmetric (1-u^2)^2 | Asymmetric: 1.5x penalty for dips |
| Scale estimate | MAD (both sides) | Upper-RMS (above-median only) |
| Median | numpy quickselect (CPU) | GPU quickselect (direct kernel) |
| Speed | 4.2 stars/sec (12 workers) | 154 stars/sec (single GPU) |
| Accuracy 0.1% | 19.8% median error | 14.7% (26% better) |
| Accuracy 0.3% | 8.4% median error | 3.7% (56% better) |
| Bias on flat stars | 0 ppm | -208 ppm (below noise floor) |

## Theoretical Properties

- **Bounded influence**: outliers beyond the rejection threshold have zero weight
- **Redescending**: influence function returns to zero for large deviations
- **Asymmetric**: tighter rejection for downward deviations (transit-side)
- **Breakdown point**: 40%+ for contamination > 3 sigma (matches biweight)
- **Asymptotic efficiency**: ~95% at the Gaussian model
- **Bias**: -190 ppm on flat stars (constant, independent of contamination)

Full theoretical analysis: `results/theoretical_analysis.md`
