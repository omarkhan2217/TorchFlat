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
| `asymmetry` | 2.0 | Higher = more transit preservation, more upward bias. 1.0 = standard biweight. |
| `cval` | 5.0 | Rejection threshold in MAD units. Higher = less rejection. |
| `n_iter` | 5 | Bisquare iterations. 5 is sufficient for convergence. |
| `window_length_days` | 0.5 | Window width. Shorter = tracks faster variability but absorbs more transit. |

### Why asymmetry = 2.0?

The optimal asymmetry depends on the ratio of transit depth to photometric
noise. A transit dip of depth `d` in noise `sigma` produces a standardized
residual of `u = -d / sigma` at the dip center. The asymmetry parameter
controls how aggressively that residual is down-weighted.

**Physical reasoning:** For the weight function to reject a transit dip,
the effective residual `u_eff = u * asymmetry` must approach the rejection
threshold `|u_eff| >= 1` (in cval-scaled MAD units). Higher asymmetry
pushes dips closer to rejection, preserving their depth.

**TESS noise floor (~1000 ppm):** A 0.1% (1000 ppm) transit produces
`|u| ~ 1.0 sigma`. At asymmetry=1.0 (standard biweight), this gets
weight ~0.83 -- nearly full weight, so the trend absorbs the dip. At
asymmetry=2.0, the effective residual doubles (`|u_eff| ~ 2.0`), giving
weight ~0.41 -- the dip is mostly excluded.

**Kepler noise floor (~100 ppm):** A 0.1% transit is ~10 sigma, already
well-rejected at any asymmetry. But 0.01% (100 ppm) transits need
asymmetry=3.0 to gain the same rejection benefit.

**Empirical validation:** Grid search over [1.0, 1.5, 2.0, 2.5, 3.0]
on 2,000 TESS stars (train), validated on 10,000 stars (test):

| asymmetry | 0.1% error | 0.3% error | Bias (ppm) |
|-----------|-----------|-----------|------------|
| 1.0 | 18.2% | 10.1% | -2 |
| 1.5 | 16.0% | 7.2% | -209 |
| **2.0** | **14.4%** | **5.0%** | **-451** |
| 2.5 | 14.0% | 4.6% | -687 |
| 3.0 | 14.2% | 4.7% | -896 |

asymmetry=2.0 is the inflection point: accuracy improves sharply from
1.0 to 2.0 but plateaus beyond. The bias doubles with each step above
2.0 while accuracy barely improves. The -451 ppm bias is below the TESS
single-cadence noise floor.

**Recommendation:** Use 2.0 for TESS surveys (default), 3.0 for Kepler
high-precision work, 1.0 for variable star surveys where bias matters.

### Variable star bias warning

On stars with >1% intrinsic variability, asymmetry=2.0 causes -7240 ppm
systematic bias. This occurs because the upper-RMS scale inflates on
variable stars, and the asymmetric weight consistently pulls the trend
above the true mean.

**Use `asymmetry=1.0` for variable star surveys.** This reverts UMI to
standard symmetric biweight behavior with near-zero bias (-2 ppm).

## Why not Gaussian Processes?

Gaussian Process (GP) regression (e.g., celerite, george) is increasingly
popular for stellar variability modeling. UMI does not compete with GPs --
they serve different roles in the pipeline:

| | UMI | GP regression |
|---|-----|---------------|
| **Use case** | Survey-scale preprocessing | Per-target detailed modeling |
| **Speed** | 154 stars/sec (GPU) | ~1-10 min per star (CPU) |
| **Full sector** | 2.1 minutes | ~30-100 hours |
| **Requires** | Window length only | Kernel choice, hyperparameter optimization |
| **Transit masking** | Built-in (asymmetric weight) | Requires iterative masking or joint model |

GPs model the covariance structure of stellar variability and can
produce physically motivated noise models. UMI is a robust location
estimator that makes no assumptions about the variability structure.

**Recommended workflow:** Use UMI for initial survey-scale detrending
and transit candidate identification, then apply GP modeling on
individual targets of interest for precise parameter estimation.

## Comparison to Wotan Biweight

| Property | wotan biweight | UMI |
|----------|---------------|-----|
| Weight function | Symmetric (1-u^2)^2 | Asymmetric: 2x penalty for dips |
| Scale estimate | MAD (both sides) | Upper-RMS (above-median only) |
| Median | numpy quickselect (CPU) | GPU quickselect (direct kernel) |
| Speed | 4.2 stars/sec (12 workers) | 154 stars/sec (single GPU) |
| Accuracy 0.1% (TESS) | 20.5% median error | 15.8% (23% better) |
| Accuracy 0.3% (TESS) | 12.7% median error | 4.9% (61% better) |
| Known planets | - | Wins 425/802 (53%) across TESS + Kepler |
| Bias on flat stars | 0 ppm | -451 ppm (below TESS noise floor) |

## Theoretical Properties

- **Bounded influence**: outliers beyond the rejection threshold have zero weight
- **Redescending**: influence function returns to zero for large deviations
- **Asymmetric**: tighter rejection for downward deviations (transit-side)
- **Breakdown point**: 40%+ for contamination > 3 sigma (matches biweight)
- **Asymptotic efficiency**: ~95% at the Gaussian model
- **Bias**: -451 ppm on TESS, -122 ppm on Kepler (scales with data noise)

Full theoretical analysis: `results/theoretical_analysis.md`
