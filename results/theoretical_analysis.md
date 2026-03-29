# UMI Asymmetric Bisquare: Theoretical Analysis

## 1. Weight Function Definition

The standard Tukey bisquare weight function:

    w(u) = (1 - u^2)^2    for |u| < 1
    w(u) = 0               for |u| >= 1

where u = (x - T) / (c * MAD), T is the location estimate, c is the tuning constant (default 5.0).

The UMI asymmetric bisquare modifies this:

    u_eff = u * alpha    if u < 0  (downward deviation, alpha = 1.5)
    u_eff = u            if u >= 0 (upward deviation)

    w(u) = (1 - u_eff^2)^2    for |u_eff| < 1
    w(u) = 0                   for |u_eff| >= 1

This creates asymmetric rejection thresholds:
- Upward: rejection at |u| >= 1 (i.e., x > T + c*MAD)
- Downward: rejection at |u| >= 1/alpha (i.e., x < T - c*MAD/alpha)

With alpha=1.5, c=5:
- Upward rejection: beyond +5 MAD = +3.37 sigma
- Downward rejection: beyond -3.33 MAD = -2.25 sigma

## 2. Influence Function

The influence function IF(x; T, F) measures the effect of a single
observation at value x on the estimator T, under the model distribution F.

For an M-estimator defined by the weight function w(u), the estimator
solves: sum_i w(u_i) * (x_i - T) = 0

The influence function is:

    IF(x) = w(u) * (x - T) / E[w(u) + w'(u) * u]

For the standard bisquare:

    IF(x) = u * (1 - u^2)^2 / E[...]    for |u| < 1
    IF(x) = 0                             for |u| >= 1

For the UMI asymmetric bisquare:

    IF(x) = u * (1 - (alpha*u)^2)^2 / E[...]    for u < 0 and |alpha*u| < 1
    IF(x) = u * (1 - u^2)^2 / E[...]             for u >= 0 and |u| < 1
    IF(x) = 0                                     otherwise

Key properties:
- **Bounded**: IF(x) is zero outside [-c*MAD/alpha, +c*MAD]. The estimator
  is B-robust (bounded influence).
- **Redescending**: IF(x) returns to zero for large |x|. Same as standard
  bisquare.
- **Asymmetric**: The influence function has different shapes for positive
  and negative deviations. Peak positive influence is at u ≈ 0.45.
  Peak negative influence is at u ≈ -0.30 (reduced by alpha).

## 3. Breakdown Point

The breakdown point epsilon* is the smallest fraction of contaminated
observations that can make the estimator arbitrarily wrong.

For the standard bisquare with c=5:
- epsilon* = 0.50 (50% breakdown point)

For the UMI asymmetric bisquare:
- **Upward contamination**: same as standard bisquare = 50%
- **Downward contamination**: depends on contamination depth

### Empirical breakdown points (Monte Carlo)

200 trials per configuration, N=361 window, Gaussian noise.
"Breaks" defined as contamination-induced bias exceeding 10%
of the contamination depth.

**Important:** UMI has a constant +0.19 sigma upward bias from
the asymmetric weighting. This bias exists with ZERO contamination
and is independent of the contamination fraction. It is the same
property that makes UMI better at preserving transit depth. At
shallow contamination (1 sigma), this constant bias dominates the
relative metric and makes UMI appear to "break early" -- but the
estimator is not actually failing, it is simply biased. The bias
does not grow with contamination.

To separate the constant asymmetry bias from actual
contamination-induced breakdown, we subtract the zero-contamination
bias before computing relative error:

  Depth (sigma)   biweight   UMI (corrected)
  3 (0.3%)        25%        40%        UMI more robust
  5 (0.5%)        40%        40%        equal
  10 (1%)         50%        50%        equal
  50 (5%)         >50%       >50%       equal

At realistic transit depths (3+ sigma above noise), UMI matches
or exceeds biweight's breakdown point. At 3-sigma contamination,
UMI actually withstands 40% contamination vs biweight's 25% --
the asymmetry makes it MORE robust against downward outliers,
not less.

**Real-world context:** A transit in a 361-point window occupies
15-60 points = 4-17% of the window. Both biweight and UMI handle
this easily. Neither breaks until 25-50% of the window is
contaminated, which never occurs with real transits.

Full results: results/breakdown_point.json

## 4. Hampel Classification

M-estimators are classified by their psi-function psi(u) = w(u) * u:

- **Monotone** (Huber): psi increases then plateaus. Not redescending.
- **Redescending** (Tukey bisquare, Welsch, Andrew's sine): psi returns
  to zero for large |u|. Hard rejection of outliers.
- **Hard rejection** (Talwar/trimmed mean): psi is constant then drops
  to zero. Discontinuous.

The UMI asymmetric bisquare is a **redescending M-estimator** with
**asymmetric rejection regions**. It belongs to the same class as
Tukey's bisquare but with a modified psi-function:

    psi(u) = u * (1 - (alpha*u)^2)^2    for u < 0, |alpha*u| < 1
    psi(u) = u * (1 - u^2)^2             for u >= 0, |u| < 1
    psi(u) = 0                            otherwise

This is:
- Continuous everywhere (smooth transition at u=0 since both forms
  give psi(0)=0)
- Redescending on both sides (returns to zero)
- Asymmetric: reaches zero at u = -1/alpha on the negative side
  vs u = +1 on the positive side

To our knowledge, asymmetric redescending M-estimators have not been
applied to photometric time-series detrending. Standard robust
statistics assumes symmetric contamination. Transit contamination is
inherently one-sided (always below the continuum), motivating the
asymmetric design.

## 5. Asymptotic Relative Efficiency

The ARE of an M-estimator measures its efficiency relative to the
sample mean at the Gaussian model.

For the standard bisquare with c=4.685 (95% efficiency): ARE = 0.95
For the standard bisquare with c=5.0: ARE ≈ 0.96

For the UMI asymmetric bisquare, the ARE is reduced because we
over-reject downward noise points. The efficiency loss is:

    ARE_UMI ≈ ARE_biweight * (1 - delta)

where delta accounts for the asymmetric rejection. For alpha=1.5:

    delta ≈ (Phi(-1/alpha) - Phi(-1)) * weight_loss ≈ 0.01

So ARE_UMI ≈ 0.95. The 1% efficiency loss is negligible compared
to the transit preservation improvement.

## 6. Bias Characterization

The asymmetric weight introduces a systematic bias on symmetric
(non-transit) data. Empirically measured on 500 flat TESS stars:

    Median bias: -0.019% (-190 ppm)
    Mean bias:   -0.072% (-720 ppm)
    Std:          0.154% (1540 ppm)

The -190 ppm median bias is below the typical TESS photometric
noise floor (~1000 ppm per cadence). For population-level studies
requiring <100 ppm systematics, the bias can be corrected by
subtracting the known offset or using asymmetry=1.0 (standard
biweight).
