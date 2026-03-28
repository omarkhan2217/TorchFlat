"""Tests for torchflat.highpass."""

import math

import numpy as np
import pytest
import torch

from torchflat.highpass import _tukey_window, fft_highpass

CADENCE = 2.0 / 1440.0  # 2-min cadence in days


class TestTukeyWindow:

    def test_alpha_zero_rectangular(self):
        w = _tukey_window(100, 0.0, torch.device("cpu"))
        assert torch.allclose(w, torch.ones(100, dtype=torch.float64))

    def test_alpha_one_hann(self):
        n = 100
        w = _tukey_window(n, 1.0, torch.device("cpu"))
        # Hann window: 0.5 * (1 - cos(2*pi*k/(n-1)))
        idx = torch.arange(n, dtype=torch.float64)
        expected = 0.5 * (1.0 - torch.cos(2.0 * math.pi * idx / (n - 1)))
        assert torch.allclose(w, expected)

    def test_shape(self):
        w = _tukey_window(1000, 0.1, torch.device("cpu"))
        assert w.shape == (1000,)

    def test_taper_edges(self):
        n = 1000
        alpha = 0.1
        w = _tukey_window(n, alpha, torch.device("cpu"))
        taper_len = int(alpha * n / 2)  # 50
        # First point should be near 0
        assert w[0].item() < 0.01
        # Last taper point should be near 1
        assert w[taper_len].item() == pytest.approx(1.0, abs=0.01)
        # Middle should be exactly 1
        assert w[n // 2].item() == pytest.approx(1.0)
        # Symmetric
        assert torch.allclose(w[:taper_len], w[-taper_len:].flip(0))

    def test_single_point(self):
        w = _tukey_window(1, 0.1, torch.device("cpu"))
        assert w.shape == (1,)
        assert w[0].item() == 1.0


class TestFFTHighpass:

    def _make_sinusoid(self, period_days: float, n_points: int = 5000):
        """Create a single-star sinusoidal signal."""
        time = torch.arange(n_points, dtype=torch.float64) * CADENCE
        flux = torch.sin(2.0 * math.pi * time / period_days)
        flux = flux.unsqueeze(0)  # [1, L]
        time = time.unsqueeze(0)
        valid = torch.ones(1, n_points, dtype=torch.bool)
        seg_id = torch.zeros(1, n_points, dtype=torch.int32)
        med_cad = torch.tensor([CADENCE], dtype=torch.float64)
        return flux, time, valid, seg_id, med_cad

    def test_removes_low_frequency(self):
        # 10-day sinusoid should be removed by 5-day cutoff
        # Use 20000 points (~28 days) so the signal completes multiple cycles
        flux, time, valid, seg_id, med_cad = self._make_sinusoid(10.0, n_points=20000)
        filtered = fft_highpass(flux, valid, seg_id, med_cad, cutoff_days=5.0)
        input_power = (flux ** 2).sum().item()
        output_power = (filtered ** 2).sum().item()
        # Taper edges leak some power, but bulk should be removed
        assert output_power / input_power < 0.05

    def test_preserves_high_frequency(self):
        # 1-day sinusoid should be preserved by 5-day cutoff
        flux, time, valid, seg_id, med_cad = self._make_sinusoid(1.0)
        filtered = fft_highpass(flux, valid, seg_id, med_cad, cutoff_days=5.0)
        input_power = (flux ** 2).sum().item()
        output_power = (filtered ** 2).sum().item()
        assert output_power / input_power > 0.8  # >80% preserved

    def test_single_segment(self):
        flux, time, valid, seg_id, med_cad = self._make_sinusoid(1.0, n_points=2000)
        filtered = fft_highpass(flux, valid, seg_id, med_cad, cutoff_days=5.0)
        assert filtered.shape == flux.shape
        # Not all zeros
        assert filtered.abs().sum().item() > 0

    def test_multiple_segments(self):
        # Two segments with different signals
        n = 2000
        flux, time, valid, seg_id, med_cad = self._make_sinusoid(1.0, n_points=n)
        # Split into two segments
        seg_id[0, n // 2 :] = 1
        filtered = fft_highpass(flux, valid, seg_id, med_cad, cutoff_days=5.0)
        assert filtered.shape == flux.shape
        # Both segments should have non-zero output
        assert filtered[0, : n // 2].abs().sum().item() > 0
        assert filtered[0, n // 2 :].abs().sum().item() > 0

    def test_segment_independence(self):
        # Inject low-freq signal in segment 0, high-freq in segment 1
        # Use longer segments so the filter has enough data to work with
        n = 20000
        time = torch.arange(n, dtype=torch.float64).unsqueeze(0) * CADENCE
        flux_seg0 = torch.sin(2.0 * math.pi * time[0, : n // 2] / 10.0)  # 10-day, should be removed
        flux_seg1 = torch.sin(2.0 * math.pi * time[0, n // 2 :] / 1.0)   # 1-day, should be kept
        flux = torch.cat([flux_seg0, flux_seg1]).unsqueeze(0)
        valid = torch.ones(1, n, dtype=torch.bool)
        seg_id = torch.zeros(1, n, dtype=torch.int32)
        seg_id[0, n // 2 :] = 1
        med_cad = torch.tensor([CADENCE], dtype=torch.float64)

        filtered = fft_highpass(flux, valid, seg_id, med_cad, cutoff_days=5.0)
        # Segment 0 (low-freq) should be mostly removed
        power_seg0 = (filtered[0, : n // 2] ** 2).sum().item()
        input_power_seg0 = (flux[0, : n // 2] ** 2).sum().item()
        assert power_seg0 / max(input_power_seg0, 1e-10) < 0.1

        # Segment 1 (high-freq) should be mostly preserved
        power_seg1 = (filtered[0, n // 2 :] ** 2).sum().item()
        input_power_seg1 = (flux[0, n // 2 :] ** 2).sum().item()
        assert power_seg1 / max(input_power_seg1, 1e-10) > 0.7

    def test_batch_dimension(self):
        B = 3
        n = 20000  # ~28 days, enough for 10-day sinusoid to complete cycles
        time = torch.arange(n, dtype=torch.float64).unsqueeze(0).expand(B, -1).contiguous() * CADENCE
        # Different periods per star
        flux_list = []
        for i, period in enumerate([1.0, 5.0, 10.0]):
            flux_list.append(torch.sin(2.0 * math.pi * time[i] / period))
        flux = torch.stack(flux_list)
        valid = torch.ones(B, n, dtype=torch.bool)
        seg_id = torch.zeros(B, n, dtype=torch.int32)
        med_cad = torch.full((B,), CADENCE, dtype=torch.float64)

        filtered = fft_highpass(flux, valid, seg_id, med_cad, cutoff_days=5.0)
        assert filtered.shape == (B, n)

        # Star 0 (1-day period): preserved
        assert (filtered[0] ** 2).sum().item() / (flux[0] ** 2).sum().item() > 0.7
        # Star 2 (10-day period): removed
        assert (filtered[2] ** 2).sum().item() / (flux[2] ** 2).sum().item() < 0.05

    def test_power_spectrum_shape(self):
        # White noise input: after highpass, power below cutoff should be ~0
        rng = np.random.default_rng(42)
        n = 4096
        flux = torch.tensor(rng.normal(0, 1, (1, n)), dtype=torch.float64)
        valid = torch.ones(1, n, dtype=torch.bool)
        seg_id = torch.zeros(1, n, dtype=torch.int32)
        med_cad = torch.tensor([CADENCE], dtype=torch.float64)

        filtered = fft_highpass(flux, valid, seg_id, med_cad, cutoff_days=5.0)

        # Check power spectrum of filtered output
        F = torch.fft.rfft(filtered[0])
        power = (F.abs() ** 2)
        freqs = torch.fft.rfftfreq(n, d=CADENCE)
        cutoff_freq = 1.0 / 5.0

        low_freq_power = power[freqs < cutoff_freq * 0.5].sum().item()
        high_freq_power = power[freqs >= cutoff_freq * 2].sum().item()

        # Low-freq power should be tiny compared to high-freq
        assert low_freq_power / max(high_freq_power, 1e-10) < 0.01

    def test_all_invalid_no_crash(self):
        flux = torch.ones(1, 100, dtype=torch.float64)
        valid = torch.zeros(1, 100, dtype=torch.bool)
        seg_id = torch.zeros(1, 100, dtype=torch.int32)
        med_cad = torch.tensor([CADENCE], dtype=torch.float64)
        result = fft_highpass(flux, valid, seg_id, med_cad)
        # All invalid -> output should be zeros
        assert (result == 0).all()
