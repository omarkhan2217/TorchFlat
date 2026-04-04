"""Tests for torchflat.batching."""

import numpy as np
import torch

from torchflat.batching import (
    assemble_batch,
    bucket_stars,
    compute_max_batch,
    cpu_prescan,
    estimate_peak_vram,
)

CADENCE = 2.0 / 1440.0


def _make_star(n_points: int = 18000, seed: int = 0):
    """Create a clean synthetic star as numpy arrays."""
    rng = np.random.default_rng(seed)
    time = np.arange(n_points, dtype=np.float64) * CADENCE
    flux = (1.0 + rng.normal(0, 0.001, n_points)).astype(np.float32)
    quality = np.zeros(n_points, dtype=np.int32)
    return time, flux, quality


class TestCpuPrescan:

    def test_clean_star(self):
        t, f, q = _make_star(18000)
        results = cpu_prescan([t], [f], [q])
        assert len(results) == 1
        r = results[0]
        assert r["index"] == 0
        assert r["n_valid"] == 18000
        assert r["degenerate"] is False
        assert r["post_filter_length"] >= r["n_valid"]

    def test_degenerate_too_few_points(self):
        t, f, q = _make_star(50)
        results = cpu_prescan([t], [f], [q])
        assert results[0]["degenerate"] is True
        assert results[0]["degenerate_reason"] == "too_few_valid_points"

    def test_degenerate_segment_too_short(self):
        # Star with many small segments, none long enough for biweight window
        n = 5000
        t = np.arange(n, dtype=np.float64) * CADENCE
        # Insert large gaps every 100 points -> segments of ~100, window=360
        for i in range(100, n, 100):
            t[i:] += 20 * CADENCE
        f = np.ones(n, dtype=np.float32)
        q = np.zeros(n, dtype=np.int32)
        results = cpu_prescan([t], [f], [q], window_samples=360)
        assert results[0]["degenerate"] is True
        assert results[0]["degenerate_reason"] == "segment_too_short"

    def test_multiple_stars(self):
        stars = [_make_star(n, seed=i) for i, n in enumerate([18000, 50, 15000])]
        ts, fs, qs = zip(*stars)
        results = cpu_prescan(list(ts), list(fs), list(qs))
        assert len(results) == 3
        assert results[0]["degenerate"] is False
        assert results[1]["degenerate"] is True
        assert results[2]["degenerate"] is False


class TestBucketStars:

    def test_bucketing_correctness(self):
        prescan = [
            {"index": 0, "post_filter_length": 14500, "degenerate": False},
            {"index": 1, "post_filter_length": 15500, "degenerate": False},
            {"index": 2, "post_filter_length": 14800, "degenerate": False},
            {"index": 3, "post_filter_length": 50, "degenerate": True, "degenerate_reason": "x"},
        ]
        buckets = bucket_stars(prescan, bucket_width=1000)
        # Star 3 is degenerate -> excluded
        total_stars = sum(len(b["star_indices"]) for b in buckets)
        assert total_stars == 3

    def test_bucket_width(self):
        prescan = [
            {"index": i, "post_filter_length": 14000 + i * 200, "degenerate": False}
            for i in range(10)
        ]
        buckets = bucket_stars(prescan, bucket_width=1000)
        for b in buckets:
            # All stars in bucket should fit within pad_length
            for idx in b["star_indices"]:
                pfl = prescan[idx]["post_filter_length"]
                assert pfl <= b["pad_length"]


class TestAssembleBatch:

    def test_padding_no_data_leakage(self):
        t1, f1, q1 = _make_star(100, seed=1)
        t2, f2, q2 = _make_star(80, seed=2)
        batch = assemble_batch([0, 1], [t1, t2], [f1, f2], [q1, q2], 120, torch.device("cpu"))
        # Padded positions should be zero
        assert (batch["flux"][0, 100:] == 0).all()
        assert (batch["flux"][1, 80:] == 0).all()

    def test_padding_mask(self):
        t, f, q = _make_star(100)
        batch = assemble_batch([0], [t], [f], [q], 150, torch.device("cpu"))
        assert batch["valid_mask"][0, :100].all()
        assert not batch["valid_mask"][0, 100:].any()

    def test_lengths(self):
        stars = [_make_star(n) for n in [100, 200, 150]]
        ts, fs, qs = zip(*stars)
        batch = assemble_batch([0, 1, 2], list(ts), list(fs), list(qs), 250, torch.device("cpu"))
        assert batch["lengths"].tolist() == [100, 200, 150]


class TestVramEstimation:

    def test_monotonic(self):
        v1 = estimate_peak_vram(15000, 360)
        v2 = estimate_peak_vram(20000, 360)
        assert v2 > v1

    def test_reasonable_range(self):
        v = estimate_peak_vram(20000, 360)
        # Should be between 100MB and 500MB per star
        assert 100 * 1024**2 < v < 500 * 1024**2


class TestComputeMaxBatch:

    def test_override(self):
        assert compute_max_batch(20000, max_batch_override=10) == 10

    def test_budget(self):
        mb_12 = compute_max_batch(20000, vram_budget_gb=12.0)
        mb_6 = compute_max_batch(20000, vram_budget_gb=6.0)
        assert mb_12 > mb_6
        assert mb_6 >= 1

    def test_cpu_fallback(self):
        assert compute_max_batch(20000, device=torch.device("cpu")) == 1
