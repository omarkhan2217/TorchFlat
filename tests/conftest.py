"""Shared test fixtures for TorchFlat test suite."""

import numpy as np
import pytest
import torch

from torchflat._utils import QUALITY_BITMASK


# ---------------------------------------------------------------------------
# Device fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    """Return GPU device if available (works for both CUDA and ROCm), else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Synthetic light curve factories
# ---------------------------------------------------------------------------

TESS_CADENCE_DAYS = 2.0 / 1440.0  # 2-minute cadence in days
BJD_OFFSET = 2457000.0             # Arbitrary BJD start


@pytest.fixture
def synthetic_flat_lc():
    """Factory fixture: returns a function that creates flat light curves.

    Usage in tests:
        time, flux, quality = synthetic_flat_lc(n_points=18000, noise_std=0.001)
    """

    def _make(n_points: int = 18000, noise_std: float = 0.001, seed: int = 42):
        rng = np.random.default_rng(seed)
        time = BJD_OFFSET + np.arange(n_points, dtype=np.float64) * TESS_CADENCE_DAYS
        flux = (1.0 + rng.normal(0, noise_std, n_points)).astype(np.float32)
        quality = np.zeros(n_points, dtype=np.int32)
        return time, flux, quality

    return _make


@pytest.fixture
def synthetic_transit_lc(synthetic_flat_lc):
    """Factory fixture: flat LC with injected box transits.

    Usage in tests:
        time, flux, quality = synthetic_transit_lc(
            depth=0.01, period=3.0, duration=2/24, n_points=18000,
        )
    """

    def _make(
        depth: float = 0.01,
        period: float = 3.0,
        duration: float = 2.0 / 24.0,
        n_points: int = 18000,
        noise_std: float = 0.001,
        seed: int = 42,
    ):
        time, flux, quality = synthetic_flat_lc(n_points, noise_std, seed)

        # Phase-fold and inject box transits
        phase = ((time - time[0]) % period) / period
        in_transit = phase < (duration / period)
        flux[in_transit] -= depth

        return time, flux, quality

    return _make


@pytest.fixture
def synthetic_gapped_lc(synthetic_flat_lc):
    """Factory fixture: flat LC with gaps inserted.

    Small gaps (<=4 cadences): flux set to NaN at those positions.
    Large gaps (>4 cadences): points removed, time shifted forward.

    Usage in tests:
        time, flux, quality = synthetic_gapped_lc(
            gap_sizes=[2, 4, 10], n_points=18000,
        )
    """

    def _make(
        gap_sizes: list[int] | None = None,
        n_points: int = 18000,
        noise_std: float = 0.001,
        seed: int = 42,
    ):
        if gap_sizes is None:
            gap_sizes = [2, 4, 10]  # two small, one large

        time, flux, quality = synthetic_flat_lc(n_points, noise_std, seed)

        # Space gaps evenly across the light curve
        n_gaps = len(gap_sizes)
        spacing = n_points // (n_gaps + 1)

        # Process gaps from end to start so indices don't shift
        remove_indices = []
        for i, gap_size in reversed(list(enumerate(gap_sizes))):
            center = spacing * (i + 1)
            start = max(0, center - gap_size // 2)
            end = min(n_points, start + gap_size)

            if gap_size <= 4:
                # Small gap: set to NaN (points remain, will be interpolated)
                flux[start:end] = np.nan
                quality[start:end] = 1  # Set a quality bit
            else:
                # Large gap: remove points and shift time forward
                remove_indices.extend(range(start, end))

        if remove_indices:
            keep_mask = np.ones(n_points, dtype=bool)
            keep_mask[remove_indices] = False
            time = time[keep_mask]
            flux = flux[keep_mask]
            quality = quality[keep_mask]

            # Re-insert the time gap: shift all points after each large gap
            # This is already handled naturally since we removed points from
            # a uniformly-spaced time array - the diff will show the gap.

        return time, flux, quality

    return _make
