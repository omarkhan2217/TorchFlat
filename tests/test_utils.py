"""Tests for torchflat._utils: masked_median and pad_to_length."""

import math

import numpy as np
import pytest
import torch

from torchflat._utils import (
    GAP_THRESHOLD,
    MIN_POINTS,
    MIN_SEGMENT_LENGTH,
    QUALITY_BITMASK,
    masked_median,
    pad_to_length,
)


# ===================================================================
# masked_median tests
# ===================================================================


class TestMaskedMedianBasic:
    """Core correctness tests for masked_median."""

    def test_odd_valid_count(self):
        x = torch.tensor([[3.0, 1.0, 4.0, 1.0, 5.0]])
        mask = torch.ones_like(x, dtype=torch.bool)
        result = masked_median(x, mask)
        assert result.shape == (1,)
        assert result.item() == pytest.approx(3.0)

    def test_even_valid_count(self):
        x = torch.tensor([[3.0, 1.0, 4.0, 2.0]])
        mask = torch.ones_like(x, dtype=torch.bool)
        result = masked_median(x, mask)
        # sorted: [1, 2, 3, 4] -> median = (2+3)/2 = 2.5
        assert result.item() == pytest.approx(2.5)

    def test_with_mask(self):
        x = torch.tensor([[3.0, 1.0, 999.0, 4.0, 2.0]])
        mask = torch.tensor([[True, True, False, True, True]])
        result = masked_median(x, mask)
        # valid: [3, 1, 4, 2] sorted: [1, 2, 3, 4] -> 2.5
        assert result.item() == pytest.approx(2.5)

    def test_all_invalid(self):
        x = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.zeros_like(x, dtype=torch.bool)
        result = masked_median(x, mask)
        assert math.isnan(result.item())

    def test_single_valid(self):
        x = torch.tensor([[5.0, 0.0, 0.0]])
        mask = torch.tensor([[True, False, False]])
        result = masked_median(x, mask)
        assert result.item() == pytest.approx(5.0)

    def test_two_valid_averages(self):
        x = torch.tensor([[10.0, 20.0, 0.0]])
        mask = torch.tensor([[True, True, False]])
        result = masked_median(x, mask)
        assert result.item() == pytest.approx(15.0)

    def test_empty_last_dim(self):
        x = torch.zeros(2, 0)
        mask = torch.zeros(2, 0, dtype=torch.bool)
        result = masked_median(x, mask)
        assert result.shape == (2,)
        assert math.isnan(result[0].item())
        assert math.isnan(result[1].item())


class TestMaskedMedianNumpyEquivalence:
    """Fuzz tests comparing masked_median to numpy.median."""

    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
    def test_matches_numpy_random(self, dtype):
        rng = np.random.default_rng(12345)
        n_tests = 1000
        N = 25  # window size

        for _ in range(n_tests):
            arr = rng.standard_normal(N).astype(
                np.float64 if dtype == torch.float64 else np.float32
            )
            # Random mask with at least 1 valid
            mask_np = rng.random(N) > 0.3
            if not mask_np.any():
                mask_np[0] = True

            expected = float(np.median(arr[mask_np]))

            x = torch.tensor(arr, dtype=dtype).unsqueeze(0)
            mask = torch.tensor(mask_np).unsqueeze(0)
            result = masked_median(x, mask).item()

            if dtype == torch.float64:
                assert result == pytest.approx(expected, abs=0), (
                    f"float64 mismatch: got {result}, expected {expected}"
                )
            else:
                assert result == pytest.approx(expected, abs=1e-6), (
                    f"float32 mismatch: got {result}, expected {expected}"
                )


class TestMaskedMedianBatched:
    """Verify masked_median works correctly on batched tensors."""

    def test_batched_2d(self):
        rng = np.random.default_rng(42)
        B, N = 8, 20

        arr = rng.standard_normal((B, N)).astype(np.float64)
        mask_np = rng.random((B, N)) > 0.2
        # Ensure at least 1 valid per row
        for i in range(B):
            if not mask_np[i].any():
                mask_np[i, 0] = True

        x = torch.tensor(arr)
        mask = torch.tensor(mask_np)
        result = masked_median(x, mask)

        assert result.shape == (B,)
        for i in range(B):
            expected = float(np.median(arr[i][mask_np[i]]))
            assert result[i].item() == pytest.approx(expected, abs=0)

    def test_batched_3d(self):
        """The biweight use case: [B, N_pos, W]."""
        rng = np.random.default_rng(99)
        B, N_pos, W = 3, 10, 15

        arr = rng.standard_normal((B, N_pos, W)).astype(np.float64)
        mask_np = rng.random((B, N_pos, W)) > 0.2
        # Ensure at least 1 valid per window
        for i in range(B):
            for j in range(N_pos):
                if not mask_np[i, j].any():
                    mask_np[i, j, 0] = True

        x = torch.tensor(arr)
        mask = torch.tensor(mask_np)
        result = masked_median(x, mask)

        assert result.shape == (B, N_pos)
        for i in range(B):
            for j in range(N_pos):
                expected = float(np.median(arr[i, j][mask_np[i, j]]))
                assert result[i, j].item() == pytest.approx(expected, abs=0)

    def test_mixed_valid_counts_per_row(self):
        """Different numbers of valid elements per row in same batch."""
        x = torch.tensor([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 20.0, 30.0, 40.0, 50.0],
        ])
        mask = torch.tensor([
            [True, True, True, True, True],    # 5 valid -> median=3
            [True, False, False, False, True],  # 2 valid: [10, 50] -> median=30
        ])
        result = masked_median(x, mask)
        assert result[0].item() == pytest.approx(3.0)
        assert result[1].item() == pytest.approx(30.0)


# ===================================================================
# pad_to_length tests
# ===================================================================


class TestPadToLength:

    def test_basic_padding(self):
        tensors = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])]
        result = pad_to_length(tensors, target_len=5, pad_value=0.0)
        expected = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0], [4.0, 5.0, 0.0, 0.0, 0.0]])
        assert torch.equal(result, expected)

    def test_no_padding_needed(self):
        tensors = [torch.tensor([1.0, 2.0, 3.0])]
        result = pad_to_length(tensors, target_len=3)
        expected = torch.tensor([[1.0, 2.0, 3.0]])
        assert torch.equal(result, expected)

    def test_custom_pad_value(self):
        tensors = [torch.tensor([1.0, 2.0])]
        result = pad_to_length(tensors, target_len=4, pad_value=float("nan"))
        assert result[0, 0].item() == 1.0
        assert result[0, 1].item() == 2.0
        assert math.isnan(result[0, 2].item())
        assert math.isnan(result[0, 3].item())

    def test_preserves_dtype(self):
        tensors = [torch.tensor([1.0, 2.0], dtype=torch.float32)]
        result = pad_to_length(tensors, target_len=5)
        assert result.dtype == torch.float32

        tensors64 = [torch.tensor([1.0, 2.0], dtype=torch.float64)]
        result64 = pad_to_length(tensors64, target_len=5)
        assert result64.dtype == torch.float64

    def test_output_shape(self):
        tensors = [torch.randn(10), torch.randn(8), torch.randn(12)]
        result = pad_to_length(tensors, target_len=15)
        assert result.shape == (3, 15)


# ===================================================================
# Constants tests
# ===================================================================


class TestConstants:

    def test_quality_bitmask_value(self):
        assert QUALITY_BITMASK == 3455

    def test_min_points(self):
        assert MIN_POINTS == 100

    def test_gap_threshold(self):
        assert GAP_THRESHOLD == 5.0

    def test_min_segment_length(self):
        assert MIN_SEGMENT_LENGTH == 50
