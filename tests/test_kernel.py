"""Tests for the custom HIP/CUDA quickselect kernel.

All tests are skipped if no GPU is available or if the kernel fails to compile.
"""

import math
import os

import numpy as np
import pytest
import torch

from torchflat._utils import masked_median

# Skip entire module if no GPU
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No GPU available"
)


@pytest.fixture(autouse=True)
def _check_kernel():
    """Skip tests if the kernel couldn't compile."""
    from torchflat._kernel_loader import _get_umi_kernel as _get_kernel

    if _get_kernel() is None:
        pytest.skip("Kernel not compiled (no toolkit or compilation failed)")


class TestKernelBasic:
    """Same tests as TestMaskedMedianBasic but on GPU."""

    def test_odd_valid_count(self):
        x = torch.tensor([[3.0, 1.0, 4.0, 1.0, 5.0]], device="cuda")
        mask = torch.ones_like(x, dtype=torch.bool)
        result = masked_median(x, mask)
        assert result.item() == pytest.approx(3.0)

    def test_even_valid_count(self):
        x = torch.tensor([[3.0, 1.0, 4.0, 2.0]], device="cuda")
        mask = torch.ones_like(x, dtype=torch.bool)
        result = masked_median(x, mask)
        assert result.item() == pytest.approx(2.5)

    def test_with_mask(self):
        x = torch.tensor([[3.0, 1.0, 999.0, 4.0, 2.0]], device="cuda")
        mask = torch.tensor([[True, True, False, True, True]], device="cuda")
        result = masked_median(x, mask)
        assert result.item() == pytest.approx(2.5)

    def test_all_invalid(self):
        x = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
        mask = torch.zeros_like(x, dtype=torch.bool, device="cuda")
        result = masked_median(x, mask)
        assert math.isnan(result.item())

    def test_single_valid(self):
        x = torch.tensor([[5.0, 0.0, 0.0]], device="cuda")
        mask = torch.tensor([[True, False, False]], device="cuda")
        result = masked_median(x, mask)
        assert result.item() == pytest.approx(5.0)

    def test_two_valid_averages(self):
        x = torch.tensor([[10.0, 20.0, 0.0]], device="cuda")
        mask = torch.tensor([[True, True, False]], device="cuda")
        result = masked_median(x, mask)
        assert result.item() == pytest.approx(15.0)


class TestKernelNumpyEquivalence:
    """Fuzz test: kernel output must match numpy.median on valid subset."""

    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
    @pytest.mark.parametrize("W", [25, 361, 512])
    def test_matches_numpy(self, dtype, W):
        rng = np.random.default_rng(12345)
        n_tests = 500

        for _ in range(n_tests):
            arr = rng.standard_normal(W).astype(
                np.float64 if dtype == torch.float64 else np.float32
            )
            mask_np = rng.random(W) > 0.3
            if not mask_np.any():
                mask_np[0] = True

            expected = float(np.median(arr[mask_np]))

            x = torch.tensor(arr, dtype=dtype, device="cuda").unsqueeze(0)
            mask = torch.tensor(mask_np, device="cuda").unsqueeze(0)
            result = masked_median(x, mask).item()

            if dtype == torch.float64:
                assert result == pytest.approx(expected, abs=1e-10), (
                    f"float64 mismatch: got {result}, expected {expected}"
                )
            else:
                assert result == pytest.approx(expected, abs=1e-6), (
                    f"float32 mismatch: got {result}, expected {expected}"
                )


class TestKernelBatched:
    """Test kernel on batched 3D tensors (the biweight use case)."""

    def test_batched_3d(self):
        rng = np.random.default_rng(99)
        B, N_pos, W = 3, 10, 25

        arr = rng.standard_normal((B, N_pos, W)).astype(np.float64)
        mask_np = rng.random((B, N_pos, W)) > 0.2
        for i in range(B):
            for j in range(N_pos):
                if not mask_np[i, j].any():
                    mask_np[i, j, 0] = True

        x = torch.tensor(arr, device="cuda")
        mask = torch.tensor(mask_np, device="cuda")
        result = masked_median(x, mask)

        assert result.shape == (B, N_pos)
        for i in range(B):
            for j in range(N_pos):
                expected = float(np.median(arr[i, j][mask_np[i, j]]))
                assert result[i, j].item() == pytest.approx(expected, abs=1e-10)

    def test_biweight_shape(self):
        """Test with realistic biweight tensor shape (smaller batch)."""
        B, N_pos, W = 5, 1000, 361
        x = torch.randn(B, N_pos, W, device="cuda", dtype=torch.float32)
        mask = torch.ones_like(x, dtype=torch.bool)
        # Mask out some positions (30% partial like real data)
        mask[:, :300, 300:] = False

        result = masked_median(x, mask)
        assert result.shape == (B, N_pos)
        assert torch.isfinite(result).all()


class TestKernelMatchesSort:
    """Verify kernel produces identical results to the torch.sort fallback."""

    def test_matches_sort_float64(self):
        rng = np.random.default_rng(42)
        B, N_pos, W = 5, 100, 361

        x = torch.tensor(
            rng.standard_normal((B, N_pos, W)),
            dtype=torch.float64, device="cuda",
        )
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, :30, 300:] = False  # 30% partial

        # Kernel result
        kernel_result = masked_median(x, mask)

        # Sort result (force fallback)
        os.environ["TORCHFLAT_NO_KERNEL"] = "1"
        import torchflat._kernel_loader as kl
        old_attempted = kl._umi_kernel_load_attempted
        old_module = kl._umi_kernel_module
        kl._umi_kernel_load_attempted = False
        kl._umi_kernel_module = None

        sort_result = masked_median(x, mask)

        # Restore
        kl._umi_kernel_load_attempted = old_attempted
        kl._umi_kernel_module = old_module
        os.environ.pop("TORCHFLAT_NO_KERNEL", None)

        both = torch.isfinite(kernel_result) & torch.isfinite(sort_result)
        diff = (kernel_result[both] - sort_result[both]).abs()
        assert diff.max().item() < 1e-10, f"Max diff: {diff.max().item()}"


class TestKernelFallback:
    """Test that fallback to torch.sort works correctly."""

    def test_cpu_uses_sort(self):
        """CPU tensors should never touch the kernel."""
        x = torch.tensor([[3.0, 1.0, 4.0, 1.0, 5.0]])
        mask = torch.ones_like(x, dtype=torch.bool)
        result = masked_median(x, mask)
        assert result.item() == pytest.approx(3.0)

    def test_large_w_uses_sort(self):
        """W > 512 should fall back to sort."""
        x = torch.randn(1, 1000, device="cuda")
        mask = torch.ones_like(x, dtype=torch.bool)
        result = masked_median(x, mask)
        assert torch.isfinite(result).all()
