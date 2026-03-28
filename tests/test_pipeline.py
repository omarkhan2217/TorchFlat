"""Tests for torchflat.pipeline — Tier 2 pipeline accuracy."""

import numpy as np
import pytest
import torch

from torchflat.pipeline import preprocess_track_a, preprocess_track_b, preprocess_sector

CADENCE = 2.0 / 1440.0


def _make_stars(n_stars: int = 5, n_points: int = 5000, seed: int = 42):
    """Create a list of clean synthetic stars."""
    rng = np.random.default_rng(seed)
    times, fluxes, qualities = [], [], []
    for i in range(n_stars):
        t = np.arange(n_points, dtype=np.float64) * CADENCE
        f = (1.0 + rng.normal(0, 0.001, n_points)).astype(np.float32)
        q = np.zeros(n_points, dtype=np.int32)
        times.append(t)
        fluxes.append(f)
        qualities.append(q)
    return times, fluxes, qualities


class TestTrackA:

    def test_output_format(self):
        times, fluxes, qualities = _make_stars(3, 5000)
        results, skipped = preprocess_track_a(
            times, fluxes, qualities,
            device="cpu", max_batch=10,
            window_scales=[(256, 128)],
        )
        assert len(results) == 3
        for r in results:
            if r is not None:
                assert "trend" in r
                assert "detrended" in r
                assert "windows_256" in r
                assert "window_times_256" in r

    def test_window_counts(self):
        times, fluxes, qualities = _make_stars(2, 5000)
        results, _ = preprocess_track_a(
            times, fluxes, qualities,
            device="cpu", max_batch=10,
            window_scales=[(256, 128)],
        )
        for r in results:
            if r is not None and "windows_256" in r:
                w = r["windows_256"]
                if w.shape[0] > 0:
                    assert w.shape[1] == 256

    def test_single_scale(self):
        times, fluxes, qualities = _make_stars(2, 5000)
        results, _ = preprocess_track_a(
            times, fluxes, qualities,
            device="cpu", max_batch=10,
            window_scales=[(512, 256)],
        )
        for r in results:
            if r is not None:
                assert "windows_512" in r
                assert "windows_256" not in r

    def test_empty_input(self):
        results, skipped = preprocess_track_a([], [], [], device="cpu")
        assert results == []
        assert skipped == []

    def test_skipped_stars_result_none(self):
        times, fluxes, qualities = _make_stars(3, 5000)
        # Make star 1 degenerate (too few points)
        times[1] = np.arange(50, dtype=np.float64) * CADENCE
        fluxes[1] = np.ones(50, dtype=np.float32)
        qualities[1] = np.zeros(50, dtype=np.int32)

        results, skipped = preprocess_track_a(
            times, fluxes, qualities,
            device="cpu", max_batch=10,
            window_scales=[(256, 128)],
        )
        assert results[1] is None
        assert any(s["index"] == 1 for s in skipped)


class TestTrackB:

    def test_output_format(self):
        times, fluxes, qualities = _make_stars(3, 5000)
        results, skipped = preprocess_track_b(
            times, fluxes, qualities,
            device="cpu", max_batch=10,
        )
        assert len(results) == 3
        for r in results:
            if r is not None:
                curve, mask = r
                assert curve.shape == (20000,)
                assert mask.shape == (20000,)

    def test_curve_length(self):
        times, fluxes, qualities = _make_stars(2, 8000)
        results, _ = preprocess_track_b(
            times, fluxes, qualities,
            device="cpu", max_batch=10, output_length=20000,
        )
        for r in results:
            if r is not None:
                assert r[0].shape == (20000,)

    def test_custom_output_length(self):
        times, fluxes, qualities = _make_stars(1, 3000)
        results, _ = preprocess_track_b(
            times, fluxes, qualities,
            device="cpu", max_batch=10, output_length=10000,
        )
        if results[0] is not None:
            assert results[0][0].shape == (10000,)


class TestPreprocessSector:

    def test_combined(self):
        n_stars = 3
        rng = np.random.default_rng(42)
        star_data = []
        for i in range(n_stars):
            n = 5000
            t = np.arange(n, dtype=np.float64) * CADENCE
            star_data.append({
                "time": t,
                "pdcsap_flux": (1.0 + rng.normal(0, 0.001, n)).astype(np.float32),
                "sap_flux": (1.0 + rng.normal(0, 0.001, n)).astype(np.float32),
                "quality": np.zeros(n, dtype=np.int32),
            })

        results, skipped = preprocess_sector(
            star_data, device="cpu", max_batch=10,
            window_scales=[(256, 128)],
        )
        assert len(results) == n_stars
        # Each result should have Track A and Track B outputs
        for r in results:
            if r:
                assert "trend" in r or "track_b_curve" in r

    def test_skipped_stars_reported(self):
        star_data = [
            {
                "time": np.arange(50, dtype=np.float64) * CADENCE,
                "pdcsap_flux": np.ones(50, dtype=np.float32),
                "sap_flux": np.ones(50, dtype=np.float32),
                "quality": np.zeros(50, dtype=np.int32),
            }
        ]
        results, skipped = preprocess_sector(
            star_data, device="cpu", max_batch=10,
            window_scales=[(256, 128)],
        )
        assert len(skipped) > 0
        assert skipped[0]["index"] == 0
