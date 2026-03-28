"""Tests for torchflat.gaps."""

import torch
import pytest

from torchflat.gaps import detect_gaps, interpolate_small_gaps

CADENCE = 2.0 / 1440.0  # 2-min cadence in days


def _uniform_time(n: int, cadence: float = CADENCE) -> torch.Tensor:
    """Create a uniform time array [1, n]."""
    return (torch.arange(n, dtype=torch.float64) * cadence).unsqueeze(0)


class TestDetectGaps:

    def test_no_gaps(self):
        L = 100
        time = _uniform_time(L)
        valid = torch.ones(1, L, dtype=torch.bool)
        seg_id, med_cad = detect_gaps(time, valid)
        assert seg_id.shape == (1, L)
        assert (seg_id == 0).all()
        assert med_cad.item() == pytest.approx(CADENCE, rel=1e-6)

    def test_single_large_gap(self):
        # 50 points, 10-cadence gap, 50 points
        t1 = torch.arange(50, dtype=torch.float64) * CADENCE
        t2 = torch.arange(50, dtype=torch.float64) * CADENCE + 60 * CADENCE
        time = torch.cat([t1, t2]).unsqueeze(0)
        valid = torch.ones(1, 100, dtype=torch.bool)
        seg_id, _ = detect_gaps(time, valid)
        assert seg_id[0, 0].item() == 0
        assert seg_id[0, 49].item() == 0
        assert seg_id[0, 50].item() == 1
        assert seg_id[0, 99].item() == 1

    def test_multiple_gaps(self):
        # 3 segments separated by large gaps
        t1 = torch.arange(30, dtype=torch.float64) * CADENCE
        t2 = torch.arange(30, dtype=torch.float64) * CADENCE + 40 * CADENCE
        t3 = torch.arange(30, dtype=torch.float64) * CADENCE + 80 * CADENCE
        time = torch.cat([t1, t2, t3]).unsqueeze(0)
        valid = torch.ones(1, 90, dtype=torch.bool)
        seg_id, _ = detect_gaps(time, valid)
        assert seg_id[0, 0].item() == 0
        assert seg_id[0, 29].item() == 0
        assert seg_id[0, 30].item() == 1
        assert seg_id[0, 59].item() == 1
        assert seg_id[0, 60].item() == 2
        assert seg_id[0, 89].item() == 2

    def test_small_gap_not_segment_boundary(self):
        # 3-cadence gap (ratio ~3) should NOT create a new segment (threshold=5)
        t = torch.arange(100, dtype=torch.float64) * CADENCE
        t[50:] += 3 * CADENCE  # shift by 3 extra cadences
        time = t.unsqueeze(0)
        valid = torch.ones(1, 100, dtype=torch.bool)
        seg_id, _ = detect_gaps(time, valid)
        # gap ratio ~4 < 5, so still one segment
        assert (seg_id == 0).all()

    def test_segment_ids_monotonic(self):
        # Random gaps
        t = torch.arange(200, dtype=torch.float64) * CADENCE
        t[50:] += 20 * CADENCE
        t[120:] += 30 * CADENCE
        time = t.unsqueeze(0)
        valid = torch.ones(1, 200, dtype=torch.bool)
        seg_id, _ = detect_gaps(time, valid)
        diffs = seg_id[0, 1:] - seg_id[0, :-1]
        assert (diffs >= 0).all()

    def test_median_cadence_calculation(self):
        time = _uniform_time(1000)
        valid = torch.ones(1, 1000, dtype=torch.bool)
        _, med_cad = detect_gaps(time, valid)
        assert med_cad.item() == pytest.approx(CADENCE, rel=1e-6)

    def test_batch_consistency(self):
        B = 3
        L = 100
        times = []
        for i in range(B):
            t = torch.arange(L, dtype=torch.float64) * CADENCE
            if i >= 1:
                t[50:] += 20 * CADENCE  # large gap for stars 1,2
            if i >= 2:
                t[80:] += 20 * CADENCE  # extra gap for star 2
            times.append(t)
        time = torch.stack(times)
        valid = torch.ones(B, L, dtype=torch.bool)
        seg_id, _ = detect_gaps(time, valid)

        assert seg_id[0].max().item() == 0  # star 0: no gaps
        assert seg_id[1].max().item() == 1  # star 1: 1 gap -> 2 segments
        assert seg_id[2].max().item() == 2  # star 2: 2 gaps -> 3 segments


class TestInterpolateSmallGaps:

    def test_no_gaps_unchanged(self):
        flux = torch.ones(1, 50)
        time = _uniform_time(50)
        valid = torch.ones(1, 50, dtype=torch.bool)
        flux_out, valid_out = interpolate_small_gaps(flux, time, valid)
        assert torch.equal(flux_out, flux)
        assert valid_out.all()

    def test_small_gap_interpolated(self):
        # 2-cadence gap between values 10.0 and 20.0
        flux = torch.ones(1, 10)
        flux[0, :] = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        valid = torch.ones(1, 10, dtype=torch.bool)
        valid[0, 3] = False
        valid[0, 4] = False
        time = _uniform_time(10)

        flux_out, valid_out = interpolate_small_gaps(flux, time, valid)
        # Should interpolate linearly between 3.0 (idx 2) and 6.0 (idx 5)
        assert valid_out[0, 3].item() is True
        assert valid_out[0, 4].item() is True
        assert flux_out[0, 3].item() == pytest.approx(4.0)
        assert flux_out[0, 4].item() == pytest.approx(5.0)

    def test_4_cadence_gap_interpolated(self):
        flux = torch.zeros(1, 20)
        flux[0, 5] = 10.0
        flux[0, 10] = 20.0
        valid = torch.ones(1, 20, dtype=torch.bool)
        valid[0, 6:10] = False  # 4-point gap
        time = _uniform_time(20)

        flux_out, valid_out = interpolate_small_gaps(flux, time, valid, max_gap=4)
        assert valid_out[0, 6:10].all()
        # Linear interp from 10.0 to 20.0 over 4 gaps (fractions 1/5, 2/5, 3/5, 4/5)
        assert flux_out[0, 6].item() == pytest.approx(12.0)
        assert flux_out[0, 7].item() == pytest.approx(14.0)
        assert flux_out[0, 8].item() == pytest.approx(16.0)
        assert flux_out[0, 9].item() == pytest.approx(18.0)

    def test_5_cadence_gap_not_interpolated(self):
        flux = torch.ones(1, 20)
        valid = torch.ones(1, 20, dtype=torch.bool)
        valid[0, 5:10] = False  # 5-point gap
        time = _uniform_time(20)

        flux_out, valid_out = interpolate_small_gaps(flux, time, valid, max_gap=4)
        # Gap too large -> not interpolated
        assert not valid_out[0, 5:10].any()

    def test_gap_at_start_not_interpolated(self):
        flux = torch.ones(1, 10)
        valid = torch.ones(1, 10, dtype=torch.bool)
        valid[0, 0:2] = False  # gap at start, no left boundary
        time = _uniform_time(10)

        flux_out, valid_out = interpolate_small_gaps(flux, time, valid)
        assert not valid_out[0, 0:2].any()

    def test_gap_at_end_not_interpolated(self):
        flux = torch.ones(1, 10)
        valid = torch.ones(1, 10, dtype=torch.bool)
        valid[0, 8:10] = False  # gap at end, no right boundary
        time = _uniform_time(10)

        flux_out, valid_out = interpolate_small_gaps(flux, time, valid)
        assert not valid_out[0, 8:10].any()

    def test_does_not_modify_input(self):
        flux = torch.ones(1, 10)
        valid = torch.ones(1, 10, dtype=torch.bool)
        valid[0, 3:5] = False
        time = _uniform_time(10)
        flux_orig = flux.clone()
        valid_orig = valid.clone()

        interpolate_small_gaps(flux, time, valid)
        assert torch.equal(flux, flux_orig)
        assert torch.equal(valid, valid_orig)
