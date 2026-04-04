"""Tests for torchflat.umi - UMI (Unified Median Iterative) detrending."""

import math

import numpy as np
import torch

from torchflat.umi import umi_detrend

CADENCE = 2.0 / 1440.0  # 2-min cadence in days


def _make_lc(n_points: int = 5000, noise_std: float = 0.0, seed: int = 42):
    """Helper: uniform time, constant flux=1.0 + noise, all valid, single segment."""
    rng = np.random.default_rng(seed)
    time = torch.arange(n_points, dtype=torch.float64).unsqueeze(0) * CADENCE
    flux = torch.ones(1, n_points, dtype=torch.float64)
    if noise_std > 0:
        flux = flux + torch.tensor(rng.normal(0, noise_std, (1, n_points)), dtype=torch.float64)
    valid = torch.ones(1, n_points, dtype=torch.bool)
    seg_id = torch.zeros(1, n_points, dtype=torch.int32)
    return time, flux, valid, seg_id


class TestBiweightSynthetic:
    """Core correctness tests - no external dependencies."""

    def test_constant_flux(self):
        time, flux, valid, seg_id = _make_lc(5000, noise_std=0.0)
        detrended, trend = umi_detrend(flux, time, valid, seg_id, dtype=torch.float64)
        # Interior trend should be exactly 1.0
        finite_trend = trend[torch.isfinite(trend)]
        assert finite_trend.numel() > 0
        assert torch.allclose(finite_trend, torch.ones_like(finite_trend), atol=1e-10)
        # Detrended should be 1.0
        finite_det = detrended[torch.isfinite(detrended)]
        assert torch.allclose(finite_det, torch.ones_like(finite_det), atol=1e-10)

    def test_sinusoidal_trend(self):
        n = 10000
        time, flux, valid, seg_id = _make_lc(n, noise_std=0.0)
        # Inject slow sinusoidal trend (2-day period)
        trend_signal = 0.01 * torch.sin(2.0 * math.pi * time / 2.0)
        flux = flux + trend_signal
        detrended, trend = umi_detrend(flux, time, valid, seg_id, dtype=torch.float64)

        # Where trend is finite, it should track the injected sinusoid
        finite_mask = torch.isfinite(trend[0])
        recovered_trend = trend[0, finite_mask]
        expected_trend = (1.0 + trend_signal[0, finite_mask])
        # RMSE should be small (biweight is a local estimator with finite window,
        # so it won't perfectly track the sinusoid - some smoothing error is expected)
        rmse = ((recovered_trend - expected_trend) ** 2).mean().sqrt().item()
        # UMI's asymmetric weight introduces a small bias on symmetric
        # variability (~0.02% for default asymmetry=1.5)
        assert rmse < 2e-3

    def test_outlier_rejection(self):
        rng = np.random.default_rng(42)
        n = 5000
        time, flux, valid, seg_id = _make_lc(n, noise_std=0.001)
        # Inject 5% outliers at 10-sigma
        n_outliers = n // 20
        outlier_idx = rng.choice(n, n_outliers, replace=False)
        flux[0, outlier_idx] += 0.01 * rng.choice([-1, 1], n_outliers)

        detrended, trend = umi_detrend(flux, time, valid, seg_id, dtype=torch.float64)

        # Trend should still be close to 1.0 (outliers rejected by biweight)
        finite_trend = trend[torch.isfinite(trend)]
        assert (finite_trend - 1.0).abs().max().item() < 0.002

    def test_transit_preservation(self):
        n = 10000
        time, flux, valid, seg_id = _make_lc(n, noise_std=0.0001)
        # Inject 1% box transit at center (2 hours = ~60 cadences)
        transit_start = n // 2 - 30
        transit_end = n // 2 + 30
        flux[0, transit_start:transit_end] -= 0.01

        detrended, trend = umi_detrend(flux, time, valid, seg_id, dtype=torch.float64)

        # Transit should not be absorbed into trend
        # Detrended values at transit should show the dip
        transit_det = detrended[0, transit_start:transit_end]
        finite_transit = transit_det[torch.isfinite(transit_det)]
        if finite_transit.numel() > 0:
            # Transit dip should be preserved (detrended ~0.99, not ~1.0)
            assert finite_transit.mean().item() < 0.995

    def test_edge_nan_behavior(self):
        time, flux, valid, seg_id = _make_lc(5000, noise_std=0.0)
        detrended, trend = umi_detrend(flux, time, valid, seg_id, dtype=torch.float64)

        # Compute expected W
        W = round(0.5 / CADENCE)
        W = W | 1  # ensure odd
        half_w = W // 2

        # First half_w and last half_w positions should be NaN
        assert torch.isnan(trend[0, :half_w]).all()
        assert torch.isnan(trend[0, -half_w:]).all()
        # Middle should be finite
        assert torch.isfinite(trend[0, half_w + 10 : -half_w - 10]).all()

    def test_segment_boundary_isolation(self):
        n = 4000
        time = torch.arange(n, dtype=torch.float64).unsqueeze(0) * CADENCE
        # Two segments: seg 0 has flux ~1.0, seg 1 has flux ~2.0
        flux = torch.ones(1, n, dtype=torch.float64)
        flux[0, n // 2:] = 2.0
        valid = torch.ones(1, n, dtype=torch.bool)
        seg_id = torch.zeros(1, n, dtype=torch.int32)
        seg_id[0, n // 2:] = 1

        detrended, trend = umi_detrend(flux, time, valid, seg_id, dtype=torch.float64)

        # Trend in segment 0 (interior) should be ~1.0
        W = round(0.5 / CADENCE) | 1
        half_w = W // 2
        seg0_interior = trend[0, half_w + 10 : n // 2 - half_w - 10]
        seg0_finite = seg0_interior[torch.isfinite(seg0_interior)]
        if seg0_finite.numel() > 0:
            assert (seg0_finite - 1.0).abs().max().item() < 0.001

        # Trend in segment 1 (interior) should be ~2.0
        seg1_interior = trend[0, n // 2 + half_w + 10 : -half_w - 10]
        seg1_finite = seg1_interior[torch.isfinite(seg1_interior)]
        if seg1_finite.numel() > 0:
            assert (seg1_finite - 2.0).abs().max().item() < 0.001

    def test_zero_mad_constant(self):
        # Perfectly constant flux: MAD=0, clamp prevents div-by-zero
        time, flux, valid, seg_id = _make_lc(2000, noise_std=0.0)
        detrended, trend = umi_detrend(flux, time, valid, seg_id, dtype=torch.float64)
        finite_trend = trend[torch.isfinite(trend)]
        assert finite_trend.numel() > 0
        assert torch.allclose(finite_trend, torch.ones_like(finite_trend), atol=1e-10)

    def test_short_segment_nan(self):
        # Single segment shorter than window -> all NaN trend
        n = 100
        time = torch.arange(n, dtype=torch.float64).unsqueeze(0) * CADENCE
        flux = torch.ones(1, n, dtype=torch.float64)
        valid = torch.ones(1, n, dtype=torch.bool)
        seg_id = torch.zeros(1, n, dtype=torch.int32)

        # Window for 0.5 days at 2-min cadence = ~360 samples > 100
        detrended, trend = umi_detrend(flux, time, valid, seg_id, dtype=torch.float64)
        assert torch.isnan(trend).all()
        assert torch.isnan(detrended).all()

    def test_dtype_float64(self):
        time, flux, valid, seg_id = _make_lc(5000, noise_std=0.001)
        detrended, trend = umi_detrend(flux, time, valid, seg_id, dtype=torch.float64)
        # Should produce valid finite output
        finite_trend = trend[torch.isfinite(trend)]
        assert finite_trend.numel() > 0
        assert (finite_trend - 1.0).abs().max().item() < 0.01

    def test_dtype_float32(self):
        time, flux, valid, seg_id = _make_lc(5000, noise_std=0.001)
        detrended, trend = umi_detrend(
            flux.float(), time, valid, seg_id, dtype=torch.float32,
        )
        finite_trend = trend[torch.isfinite(trend)]
        assert finite_trend.numel() > 0
        assert (finite_trend - 1.0).abs().max().item() < 0.01

    def test_batch_dimension(self):
        B = 3
        n = 5000
        time = torch.arange(n, dtype=torch.float64).unsqueeze(0).expand(B, -1).contiguous() * CADENCE
        flux = torch.ones(B, n, dtype=torch.float64)
        flux[1] = 2.0  # star 1 has different flux level
        flux[2] = 0.5  # star 2 has different flux level
        valid = torch.ones(B, n, dtype=torch.bool)
        seg_id = torch.zeros(B, n, dtype=torch.int32)

        detrended, trend = umi_detrend(flux, time, valid, seg_id, dtype=torch.float64)

        for b, expected_level in enumerate([1.0, 2.0, 0.5]):
            finite_trend = trend[b][torch.isfinite(trend[b])]
            assert finite_trend.numel() > 0
            assert (finite_trend - expected_level).abs().max().item() < 0.001

    def test_float32_vs_float64_divergence(self):
        """Quantify precision difference between float32 and float64."""
        rng = np.random.default_rng(123)
        n = 5000
        time = torch.arange(n, dtype=torch.float64).unsqueeze(0) * CADENCE
        flux = torch.tensor(
            1.0 + rng.normal(0, 0.001, (1, n)), dtype=torch.float64,
        )
        valid = torch.ones(1, n, dtype=torch.bool)
        seg_id = torch.zeros(1, n, dtype=torch.int32)

        _, trend_f64 = umi_detrend(flux, time, valid, seg_id, dtype=torch.float64)
        _, trend_f32 = umi_detrend(flux.float(), time, valid, seg_id, dtype=torch.float32)

        # Compare where both are finite
        both_finite = torch.isfinite(trend_f64[0]) & torch.isfinite(trend_f32[0])
        if both_finite.sum().item() > 0:
            t64 = trend_f64[0, both_finite].double()
            t32 = trend_f32[0, both_finite].double()
            rel_err = ((t32 - t64) / t64.clamp(min=1e-10)).abs()
            max_err = rel_err.max().item()
            mean_err = rel_err.mean().item()
            p99_err = rel_err.quantile(0.99).item()
            # float32 error should be small relative to TESS noise floor (~500 ppm)
            assert p99_err < 1e-4, f"99th percentile relative error {p99_err:.2e} exceeds 1e-4"
            # Print for informational purposes (visible in pytest -v -s)
            print(f"\nfloat32 vs float64: max={max_err:.2e}, mean={mean_err:.2e}, p99={p99_err:.2e}")
