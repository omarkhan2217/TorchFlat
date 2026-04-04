"""Tests for torchflat.normalize."""


import numpy as np
import pytest
import torch

from torchflat._utils import masked_median
from torchflat.normalize import normalize_track_a, normalize_track_b


class TestNormalizeTrackA:

    def test_constant_flux(self):
        flux = torch.full((1, 100), 1.0)
        valid = torch.ones(1, 100, dtype=torch.bool)
        result = normalize_track_a(flux, valid)
        # (1.0 / 1.0) - 1.0 = 0.0
        assert torch.allclose(result, torch.zeros_like(result))

    def test_transit_dip(self):
        flux = torch.full((1, 1000), 1.0)
        valid = torch.ones(1, 1000, dtype=torch.bool)
        flux[0, 500:520] = 0.99  # 1% dip
        result = normalize_track_a(flux, valid)
        # Dip should appear as ~ -0.01
        dip_vals = result[0, 500:520]
        assert (dip_vals < -0.005).all()
        assert (dip_vals > -0.02).all()

    def test_clamped(self):
        flux = torch.full((1, 100), 1.0)
        flux[0, 0] = 100.0  # extreme outlier
        valid = torch.ones(1, 100, dtype=torch.bool)
        result = normalize_track_a(flux, valid)
        assert result.min().item() >= -0.5
        assert result.max().item() <= 0.5

    def test_median_near_zero(self):
        rng = np.random.default_rng(42)
        flux = torch.tensor(1.0 + rng.normal(0, 0.001, (1, 5000)), dtype=torch.float32)
        valid = torch.ones(1, 5000, dtype=torch.bool)
        result = normalize_track_a(flux, valid)
        med = masked_median(result, valid)
        assert abs(med.item()) < 0.001

    def test_all_invalid(self):
        flux = torch.ones(1, 10)
        valid = torch.zeros(1, 10, dtype=torch.bool)
        result = normalize_track_a(flux, valid)
        # NaN median -> NaN division -> invalid positions set to 0.0
        assert (result == 0.0).all()

    def test_batch_independence(self):
        flux = torch.tensor([[1.0] * 50, [2.0] * 50, [0.5] * 50])
        valid = torch.ones(3, 50, dtype=torch.bool)
        result = normalize_track_a(flux, valid)
        # All constant -> all 0.0
        assert torch.allclose(result, torch.zeros_like(result))


class TestNormalizeTrackB:

    def test_unit_scale(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (1, 10000)).astype(np.float64)
        flux = torch.tensor(data)
        valid = torch.ones(1, 10000, dtype=torch.bool)
        result = normalize_track_b(flux, valid)
        # MAD of N(0,1) ≈ 0.6745, scale = 0.6745 * 1.4826 ≈ 1.0
        # So output should have MAD ≈ 0.6745 (i.e. roughly unit-scale)
        abs_dev = result[0, valid[0]].abs()
        result_mad = abs_dev.median().item()
        assert result_mad == pytest.approx(0.6745, abs=0.05)

    def test_constant_flux(self):
        flux = torch.full((1, 100), 5.0, dtype=torch.float64)
        valid = torch.ones(1, 100, dtype=torch.bool)
        result = normalize_track_b(flux, valid)
        # center=5, MAD=0 -> scale clamped to 1e-8*1.4826 -> (5-5)/scale = 0
        assert torch.allclose(result, torch.zeros_like(result))

    def test_all_invalid(self):
        flux = torch.ones(1, 10)
        valid = torch.zeros(1, 10, dtype=torch.bool)
        result = normalize_track_b(flux, valid)
        assert (result == 0.0).all()

    def test_batch_independence(self):
        rng = np.random.default_rng(99)
        flux = torch.tensor(
            np.stack([
                rng.normal(10, 1, 500),
                rng.normal(100, 5, 500),
                rng.normal(0, 0.1, 500),
            ]),
            dtype=torch.float64,
        )
        valid = torch.ones(3, 500, dtype=torch.bool)
        result = normalize_track_b(flux, valid)
        # Each star should be centered near 0
        for i in range(3):
            center = result[i, valid[i]].median().item()
            assert abs(center) < 0.1
