"""Fast CPU biweight detrending kernel using Numba JIT.

Uses numpy.median (quickselect, O(n)) instead of torch.sort (O(n log n)).
Processes one star at a time but is designed to run across multiple CPU
workers in parallel via ProcessPoolExecutor.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _biweight_one_star(
    flux: np.ndarray,
    time: np.ndarray,
    valid: np.ndarray,
    segment_id: np.ndarray,
    window_length_days: float,
    n_iter: int,
    cval: float,
    min_segment_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Biweight detrend a single star. Called from Numba-compatible code.

    Args:
        flux: 1D float64 array of length L.
        time: 1D float64 array of length L.
        valid: 1D bool array of length L.
        segment_id: 1D int32 array of length L.
        window_length_days: Window width in days.
        n_iter: Number of biweight iterations.
        cval: Biweight c parameter.
        min_segment_points: Minimum valid points for a valid trend.

    Returns:
        (detrended, trend) both 1D float64 arrays of length L.
    """
    L = len(flux)
    trend = np.full(L, np.nan, dtype=np.float64)
    detrended = np.full(L, np.nan, dtype=np.float64)

    if L < 3:
        return detrended, trend

    # Compute median cadence from valid consecutive pairs
    dt_values = np.empty(L - 1, dtype=np.float64)
    n_dt = 0
    for i in range(L - 1):
        if valid[i] and valid[i + 1]:
            dt_values[n_dt] = time[i + 1] - time[i]
            n_dt += 1

    if n_dt == 0:
        return detrended, trend

    median_cadence = np.median(dt_values[:n_dt])
    if median_cadence <= 0:
        return detrended, trend

    W = int(round(window_length_days / median_cadence))
    if W < 3:
        W = 3
    if W % 2 == 0:
        W += 1  # ensure odd

    if L < W:
        return detrended, trend

    half_w = W // 2

    # Process each window position
    # Pre-allocate scratch arrays
    window_data = np.empty(W, dtype=np.float64)

    for pos in range(half_w, L - half_w):
        center_seg = segment_id[pos]

        # Gather valid points in this window that share the center's segment
        n_valid = 0
        for k in range(pos - half_w, pos + half_w + 1):
            if valid[k] and segment_id[k] == center_seg:
                window_data[n_valid] = flux[k]
                n_valid += 1

        if n_valid < min_segment_points:
            continue

        # Initial location: median (quickselect, O(n))
        location = np.median(window_data[:n_valid])

        # Compute MAD once
        abs_dev = np.empty(n_valid, dtype=np.float64)
        for j in range(n_valid):
            abs_dev[j] = abs(window_data[j] - location)
        mad = np.median(abs_dev)
        if mad < 1e-10:
            mad = 1e-10
        safe_mad = cval * mad

        # Biweight iterations (fixed MAD, matching GPU kernel)
        for _it in range(n_iter):
            w_sum = 0.0
            wx_sum = 0.0
            for j in range(n_valid):
                u = (window_data[j] - location) / safe_mad
                if abs(u) < 1.0:
                    w = (1.0 - u * u) ** 2
                    w_sum += w
                    wx_sum += w * window_data[j]
            if w_sum > 1e-10:
                location = wx_sum / w_sum

        trend[pos] = location

    # Detrend
    for i in range(L):
        if not np.isnan(trend[i]) and trend[i] > 0 and valid[i]:
            detrended[i] = flux[i] / trend[i]

    return detrended, trend


def biweight_detrend_cpu_single(
    flux: np.ndarray,
    time: np.ndarray,
    valid: np.ndarray,
    segment_id: np.ndarray,
    window_length_days: float = 0.5,
    n_iter: int = 5,
    cval: float = 5.0,
    min_segment_points: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Biweight detrend a single star on CPU using Numba.

    This is the entry point for CPU workers. Wraps the Numba JIT kernel
    with type conversion.

    Returns:
        (detrended, trend) as numpy float64 arrays.
    """
    return _biweight_one_star(
        flux.astype(np.float64),
        time.astype(np.float64),
        valid.astype(np.bool_),
        segment_id.astype(np.int32),
        window_length_days,
        n_iter,
        cval,
        min_segment_points,
    )
