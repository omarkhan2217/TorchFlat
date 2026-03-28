"""Tests for torchflat.windows."""

import torch
import pytest

from torchflat.windows import DEFAULT_WINDOW_SCALES, extract_windows


def _make_simple_data(B: int, L: int):
    """Create simple test data: arange flux, single segment, all valid."""
    flux = torch.arange(L, dtype=torch.float32).unsqueeze(0).expand(B, -1).contiguous()
    valid = torch.ones(B, L, dtype=torch.bool)
    seg_id = torch.zeros(B, L, dtype=torch.int32)
    time = torch.arange(L, dtype=torch.float64).unsqueeze(0).expand(B, -1).contiguous()
    return flux, valid, seg_id, time


class TestExtractWindows:

    def test_output_shapes(self):
        B, L = 2, 1024
        flux, valid, seg_id, time = _make_simple_data(B, L)
        result = extract_windows(flux, valid, seg_id, time,
                                 window_scales=[(256, 128)])
        assert 256 in result
        w = result[256]
        assert w["windows"].shape[1] == 256
        assert w["window_times"].shape[1] == 2
        assert w["star_indices"].shape[0] == w["windows"].shape[0]
        # Expected windows: (L - win_size) // stride + 1 = (1024-256)//128 + 1 = 7 per star
        expected_per_star = (L - 256) // 128 + 1
        assert w["windows"].shape[0] == B * expected_per_star

    def test_cross_segment_excluded(self):
        B, L = 1, 600
        flux, valid, seg_id, time = _make_simple_data(B, L)
        # Two segments: 0..299 = seg 0, 300..599 = seg 1
        seg_id[0, 300:] = 1
        result = extract_windows(flux, valid, seg_id, time,
                                 window_scales=[(256, 128)])
        w = result[256]
        # Windows that span position 300 should be excluded
        # Valid windows: those entirely within seg 0 or entirely within seg 1
        for i in range(w["windows"].shape[0]):
            start_time = w["window_times"][i, 0].item()
            end_time = w["window_times"][i, 1].item()
            # Window shouldn't cross the boundary at position 300
            assert not (start_time < 300 and end_time >= 300)

    def test_single_scale(self):
        B, L = 1, 1024
        flux, valid, seg_id, time = _make_simple_data(B, L)
        result = extract_windows(flux, valid, seg_id, time,
                                 window_scales=[(512, 256)])
        assert len(result) == 1
        assert 512 in result

    def test_all_scales(self):
        B, L = 1, 16384  # large enough for all default scales
        flux, valid, seg_id, time = _make_simple_data(B, L)
        result = extract_windows(flux, valid, seg_id, time)
        for win_size, _ in DEFAULT_WINDOW_SCALES:
            assert win_size in result

    def test_window_content(self):
        B, L = 1, 512
        flux, valid, seg_id, time = _make_simple_data(B, L)
        result = extract_windows(flux, valid, seg_id, time,
                                 window_scales=[(256, 256)])
        w = result[256]
        # First window: flux[0:256], second window: flux[256:512]
        assert w["windows"].shape[0] == 2
        assert w["windows"][0, 0].item() == pytest.approx(0.0)
        assert w["windows"][0, -1].item() == pytest.approx(255.0)
        assert w["windows"][1, 0].item() == pytest.approx(256.0)
        assert w["windows"][1, -1].item() == pytest.approx(511.0)

    def test_star_indices(self):
        B, L = 3, 512
        flux, valid, seg_id, time = _make_simple_data(B, L)
        result = extract_windows(flux, valid, seg_id, time,
                                 window_scales=[(256, 256)])
        w = result[256]
        # 2 windows per star, 3 stars = 6 windows
        assert w["star_indices"].shape[0] == 6
        assert (w["star_indices"][:2] == 0).all()
        assert (w["star_indices"][2:4] == 1).all()
        assert (w["star_indices"][4:6] == 2).all()

    def test_window_timestamps(self):
        B, L = 1, 512
        flux, valid, seg_id, time = _make_simple_data(B, L)
        result = extract_windows(flux, valid, seg_id, time,
                                 window_scales=[(256, 256)])
        w = result[256]
        # First window: time[0] .. time[255]
        assert w["window_times"][0, 0].item() == pytest.approx(0.0)
        assert w["window_times"][0, 1].item() == pytest.approx(255.0)

    def test_short_star_no_windows(self):
        B, L = 1, 100  # shorter than 256
        flux, valid, seg_id, time = _make_simple_data(B, L)
        result = extract_windows(flux, valid, seg_id, time,
                                 window_scales=[(256, 128)])
        w = result[256]
        assert w["windows"].shape[0] == 0
        assert w["star_indices"].shape[0] == 0

    def test_invalid_points_exclude_window(self):
        B, L = 1, 512
        flux, valid, seg_id, time = _make_simple_data(B, L)
        # Invalidate one point in the first window
        valid[0, 100] = False
        result = extract_windows(flux, valid, seg_id, time,
                                 window_scales=[(256, 256)])
        w = result[256]
        # First window (0:256) has an invalid point -> excluded
        # Second window (256:512) is fully valid
        assert w["windows"].shape[0] == 1
        assert w["windows"][0, 0].item() == pytest.approx(256.0)

    def test_results_on_cpu(self):
        B, L = 1, 512
        flux, valid, seg_id, time = _make_simple_data(B, L)
        result = extract_windows(flux, valid, seg_id, time,
                                 window_scales=[(256, 128)])
        w = result[256]
        assert w["windows"].device.type == "cpu"
        assert w["window_times"].device.type == "cpu"
        assert w["star_indices"].device.type == "cpu"
