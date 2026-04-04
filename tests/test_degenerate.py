"""Tests for degenerate/edge case handling."""

import logging

import numpy as np

from torchflat.pipeline import preprocess_track_a

CADENCE = 2.0 / 1440.0


class TestDegenerateCases:

    def test_all_invalid_star_nan(self):
        t = np.arange(1000, dtype=np.float64) * CADENCE
        f = np.full(1000, np.nan, dtype=np.float32)  # all NaN
        q = np.zeros(1000, dtype=np.int32)
        results, skipped = preprocess_track_a(
            [t], [f], [q], device="cpu", max_batch=1, window_scales=[(256, 128)],
        )
        assert results[0] is None
        assert len(skipped) == 1

    def test_too_few_points(self):
        t = np.arange(50, dtype=np.float64) * CADENCE
        f = np.ones(50, dtype=np.float32)
        q = np.zeros(50, dtype=np.int32)
        results, skipped = preprocess_track_a(
            [t], [f], [q], device="cpu", max_batch=1, window_scales=[(256, 128)],
        )
        assert results[0] is None
        assert skipped[0]["reason"] == "too_few_valid_points"

    def test_zero_mad_no_crash(self):
        # Perfectly constant flux
        n = 5000
        t = np.arange(n, dtype=np.float64) * CADENCE
        f = np.ones(n, dtype=np.float32)
        q = np.zeros(n, dtype=np.int32)
        results, skipped = preprocess_track_a(
            [t], [f], [q], device="cpu", max_batch=1, window_scales=[(256, 128)],
        )
        # Should not crash; may produce a result or be skipped
        assert len(results) == 1

    def test_all_degenerate_batch(self):
        # All stars degenerate -> no GPU processing needed
        stars_t = [np.arange(50, dtype=np.float64) * CADENCE for _ in range(5)]
        stars_f = [np.ones(50, dtype=np.float32) for _ in range(5)]
        stars_q = [np.zeros(50, dtype=np.int32) for _ in range(5)]
        results, skipped = preprocess_track_a(
            stars_t, stars_f, stars_q,
            device="cpu", max_batch=1, window_scales=[(256, 128)],
        )
        assert all(r is None for r in results)
        assert len(skipped) == 5

    def test_skipped_stars_logged(self, caplog):
        t = np.arange(50, dtype=np.float64) * CADENCE
        f = np.ones(50, dtype=np.float32)
        q = np.zeros(50, dtype=np.int32)
        with caplog.at_level(logging.WARNING, logger="torchflat"):
            preprocess_track_a(
                [t], [f], [q], device="cpu", max_batch=1, window_scales=[(256, 128)],
            )
        assert any("skipped" in record.message.lower() for record in caplog.records)

    def test_mixed_degenerate_and_valid(self):
        # Star 0: valid, Star 1: degenerate, Star 2: valid
        rng = np.random.default_rng(42)
        n = 5000
        times = [
            np.arange(n, dtype=np.float64) * CADENCE,
            np.arange(30, dtype=np.float64) * CADENCE,
            np.arange(n, dtype=np.float64) * CADENCE,
        ]
        fluxes = [
            (1.0 + rng.normal(0, 0.001, n)).astype(np.float32),
            np.ones(30, dtype=np.float32),
            (1.0 + rng.normal(0, 0.001, n)).astype(np.float32),
        ]
        qualities = [np.zeros(n, dtype=np.int32), np.zeros(30, dtype=np.int32), np.zeros(n, dtype=np.int32)]

        results, skipped = preprocess_track_a(
            times, fluxes, qualities,
            device="cpu", max_batch=10, window_scales=[(256, 128)],
        )
        assert results[0] is not None
        assert results[1] is None
        assert results[2] is not None
        assert len(skipped) == 1
        assert skipped[0]["index"] == 1
