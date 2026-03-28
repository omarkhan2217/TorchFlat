"""Tests for torchflat.quality."""

import torch

from torchflat.quality import quality_filter


class TestQualityFilter:

    def test_clean_data(self):
        B, L = 2, 100
        flux = torch.ones(B, L)
        time = torch.arange(L, dtype=torch.float64).unsqueeze(0).expand(B, -1)
        quality = torch.zeros(B, L, dtype=torch.int32)
        mask = quality_filter(flux, time, quality)
        assert mask.all()

    def test_bitmask_rejection(self):
        flux = torch.ones(1, 10)
        time = torch.arange(10, dtype=torch.float64).unsqueeze(0)
        quality = torch.zeros(1, 10, dtype=torch.int32)
        quality[0, 3] = 1  # attitude tweak bit
        quality[0, 7] = 2  # safe mode bit
        mask = quality_filter(flux, time, quality)
        assert mask[0, 3].item() is False
        assert mask[0, 7].item() is False
        assert mask[0, 0].item() is True
        assert mask.sum().item() == 8

    def test_nan_flux_rejected(self):
        flux = torch.ones(1, 5)
        flux[0, 2] = float("nan")
        time = torch.arange(5, dtype=torch.float64).unsqueeze(0)
        quality = torch.zeros(1, 5, dtype=torch.int32)
        mask = quality_filter(flux, time, quality)
        assert mask[0, 2].item() is False
        assert mask.sum().item() == 4

    def test_inf_flux_rejected(self):
        flux = torch.ones(1, 5)
        flux[0, 1] = float("inf")
        time = torch.arange(5, dtype=torch.float64).unsqueeze(0)
        quality = torch.zeros(1, 5, dtype=torch.int32)
        mask = quality_filter(flux, time, quality)
        assert mask[0, 1].item() is False

    def test_nan_time_rejected(self):
        flux = torch.ones(1, 5)
        time = torch.arange(5, dtype=torch.float64).unsqueeze(0).clone()
        time[0, 4] = float("nan")
        quality = torch.zeros(1, 5, dtype=torch.int32)
        mask = quality_filter(flux, time, quality)
        assert mask[0, 4].item() is False

    def test_batch_dimension(self):
        B = 5
        flux = torch.ones(B, 10)
        time = torch.arange(10, dtype=torch.float64).unsqueeze(0).expand(B, -1)
        quality = torch.zeros(B, 10, dtype=torch.int32)
        # Each star has a different bad point
        for i in range(B):
            quality[i, i] = 1
        mask = quality_filter(flux, time, quality)
        assert mask.shape == (B, 10)
        for i in range(B):
            assert mask[i, i].item() is False
            assert mask[i].sum().item() == 9

    def test_custom_bitmask(self):
        flux = torch.ones(1, 5)
        time = torch.arange(5, dtype=torch.float64).unsqueeze(0)
        quality = torch.full((1, 5), 255, dtype=torch.int32)  # all bits set
        mask = quality_filter(flux, time, quality, bitmask=0)  # accept all
        assert mask.all()
