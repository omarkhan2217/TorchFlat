"""Tests for output determinism and reproducibility."""

import numpy as np
import pytest

from torchflat.pipeline import preprocess_track_a

CADENCE = 2.0 / 1440.0


def _make_stars(n_stars: int = 3, n_points: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    times, fluxes, qualities = [], [], []
    for _ in range(n_stars):
        t = np.arange(n_points, dtype=np.float64) * CADENCE
        f = (1.0 + rng.normal(0, 0.001, n_points)).astype(np.float32)
        q = np.zeros(n_points, dtype=np.int32)
        times.append(t)
        fluxes.append(f)
        qualities.append(q)
    return times, fluxes, qualities


class TestDeterminism:

    def test_same_input_same_output(self):
        times, fluxes, qualities = _make_stars(3, 5000)
        kwargs = dict(device="cpu", max_batch=10, window_scales=[(256, 128)])

        results1, _ = preprocess_track_a(times, fluxes, qualities, **kwargs)
        results2, _ = preprocess_track_a(times, fluxes, qualities, **kwargs)

        for r1, r2 in zip(results1, results2):
            if r1 is None:
                assert r2 is None
                continue
            # Trends should be bit-identical
            np.testing.assert_array_equal(r1["trend"], r2["trend"])
            np.testing.assert_array_equal(r1["detrended"], r2["detrended"])
            if "windows_256" in r1:
                np.testing.assert_array_equal(r1["windows_256"], r2["windows_256"])

    def test_batch_size_invariance(self):
        times, fluxes, qualities = _make_stars(3, 5000)
        kwargs = dict(device="cpu", window_scales=[(256, 128)])

        results_b1, _ = preprocess_track_a(times, fluxes, qualities, max_batch=1, **kwargs)
        results_b10, _ = preprocess_track_a(times, fluxes, qualities, max_batch=10, **kwargs)

        for r1, r10 in zip(results_b1, results_b10):
            if r1 is None:
                assert r10 is None
                continue
            # Same star should produce same trend regardless of batch size
            np.testing.assert_allclose(
                r1["trend"], r10["trend"],
                rtol=1e-5, atol=1e-7, equal_nan=True,
            )
