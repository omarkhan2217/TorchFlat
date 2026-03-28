"""Tests for torchflat.clipping."""

import numpy as np
import pytest
import torch

from torchflat._utils import masked_median
from torchflat.clipping import conservative_clip, rolling_clip


class TestRollingClip:

    def test_no_outliers_preserved(self):
        rng = np.random.default_rng(42)
        flux = torch.tensor(1.0 + rng.normal(0, 0.001, (1, 2000)), dtype=torch.float32)
        valid = torch.ones(1, 2000, dtype=torch.bool)
        seg_id = torch.zeros(1, 2000, dtype=torch.int32)
        result = rolling_clip(flux, valid, seg_id, sigma=5.0)
        # Almost all points should survive 5-sigma clipping on clean data
        assert result.sum().item() >= 1990

    def test_obvious_outliers_clipped(self):
        rng = np.random.default_rng(42)
        flux = torch.tensor(1.0 + rng.normal(0, 0.001, (1, 2000)), dtype=torch.float32)
        valid = torch.ones(1, 2000, dtype=torch.bool)
        seg_id = torch.zeros(1, 2000, dtype=torch.int32)
        # Inject 5 extreme outliers at 100-sigma
        outlier_pos = [100, 400, 800, 1200, 1600]
        for p in outlier_pos:
            flux[0, p] += 0.1  # ~100x the noise std
        result = rolling_clip(flux, valid, seg_id, sigma=5.0)
        for p in outlier_pos:
            assert result[0, p].item() is False

    def test_rolling_median_shape(self):
        flux = torch.ones(3, 500)
        valid = torch.ones(3, 500, dtype=torch.bool)
        seg_id = torch.zeros(3, 500, dtype=torch.int32)
        result = rolling_clip(flux, valid, seg_id)
        assert result.shape == (3, 500)

    def test_edge_padding(self):
        # Constant flux -> rolling median should be constant everywhere including edges
        flux = torch.full((1, 100), 5.0)
        valid = torch.ones(1, 100, dtype=torch.bool)
        seg_id = torch.zeros(1, 100, dtype=torch.int32)
        result = rolling_clip(flux, valid, seg_id)
        # No clipping on constant data
        assert result.all()

    def test_mad_calculation(self):
        # N(0,1) data: MAD ≈ 0.6745
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (1, 50000))
        flux = torch.tensor(data, dtype=torch.float64)
        valid = torch.ones(1, 50000, dtype=torch.bool)
        abs_dev = flux.abs()
        mad = masked_median(abs_dev, valid)
        assert mad.item() == pytest.approx(0.6745, abs=0.02)

    def test_batch_independence(self):
        rng = np.random.default_rng(42)
        B = 3
        flux_list = []
        for i in range(B):
            noise_std = 0.001 * (i + 1)
            flux_list.append(1.0 + rng.normal(0, noise_std, 1000))
        flux = torch.tensor(np.stack(flux_list), dtype=torch.float32)
        valid = torch.ones(B, 1000, dtype=torch.bool)
        seg_id = torch.zeros(B, 1000, dtype=torch.int32)

        # Inject outlier only in star 1
        flux[1, 500] += 0.5

        result = rolling_clip(flux, valid, seg_id, sigma=5.0)
        # Star 1's outlier should be clipped
        assert result[1, 500].item() is False
        # Star 0 and 2 should be mostly intact
        assert result[0].sum().item() >= 990
        assert result[2].sum().item() >= 990

    def test_all_invalid_no_crash(self):
        flux = torch.ones(1, 50)
        valid = torch.zeros(1, 50, dtype=torch.bool)
        seg_id = torch.zeros(1, 50, dtype=torch.int32)
        result = rolling_clip(flux, valid, seg_id)
        assert not result.any()


class TestConservativeClip:

    def test_conservative_wider(self):
        rng = np.random.default_rng(42)
        flux = torch.tensor(1.0 + rng.normal(0, 0.001, (1, 2000)), dtype=torch.float32)
        valid = torch.ones(1, 2000, dtype=torch.bool)
        seg_id = torch.zeros(1, 2000, dtype=torch.int32)

        # Inject moderate outliers (6-sigma relative to noise)
        flux[0, 100] += 0.006
        flux[0, 200] += 0.006

        result_rolling = rolling_clip(flux, valid, seg_id, sigma=5.0)
        result_conserv = conservative_clip(flux, valid, sigma=10.0)

        # Conservative should clip equal or fewer points
        assert result_conserv.sum().item() >= result_rolling.sum().item()

    def test_conservative_large_flare(self):
        rng = np.random.default_rng(42)
        flux = torch.tensor(1.0 + rng.normal(0, 0.001, (1, 2000)), dtype=torch.float32)
        valid = torch.ones(1, 2000, dtype=torch.bool)
        flux[0, 999] += 0.5  # massive flare (~500 sigma)
        result = conservative_clip(flux, valid, sigma=10.0)
        assert result[0, 999].item() is False

    def test_clean_data_preserved(self):
        rng = np.random.default_rng(42)
        flux = torch.tensor(1.0 + rng.normal(0, 0.001, (1, 5000)), dtype=torch.float32)
        valid = torch.ones(1, 5000, dtype=torch.bool)
        result = conservative_clip(flux, valid, sigma=10.0)
        # 10-sigma on clean data: essentially nothing clipped
        assert result.sum().item() >= 4990
