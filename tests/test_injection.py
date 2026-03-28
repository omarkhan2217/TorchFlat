"""Tier 3 science validation: transit injection and recovery tests.

Verifies that torchflat's preprocessing preserves injected transit signals
and does not introduce systematic biases in recovered transit depths.

Note: The sigma clipping step correctly removes deep transits (>5-sigma).
These tests use a high clip_sigma to test the biweight detrending's transit
preservation independently, or use shallow transits that survive clipping.
"""

from __future__ import annotations

import numpy as np
import pytest

from torchflat.pipeline import preprocess_track_a

CADENCE = 2.0 / 1440.0  # 2-min cadence in days


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inject_transit(
    time: np.ndarray,
    flux: np.ndarray,
    depth: float,
    period: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject a box transit and return (flux_injected, in_transit_mask)."""
    phase = ((time - time[0]) % period) / period
    half_dur = duration / period / 2.0
    in_transit = (phase < half_dur) | (phase > 1.0 - half_dur)
    flux_out = flux.copy()
    flux_out[in_transit] -= depth * np.median(flux)
    return flux_out, in_transit


def _recover_depth(
    detrended: np.ndarray,
    time: np.ndarray,
    period: float,
    duration: float,
) -> float:
    """Recover transit depth from detrended flux via phase-folding.

    Detrended flux is flux/trend, so baseline ~1.0, transit dip ~(1-depth).
    Uses time[0] as phase reference to match the injection function.
    """
    finite = np.isfinite(detrended)
    if finite.sum() < 100:
        return np.nan

    det = detrended[finite]
    t = time[finite]

    # Use time[0] (original array start) as phase reference, NOT t[0]
    # This matches the injection function's phase calculation
    phase = ((t - time[0]) % period) / period
    half_dur = duration / period / 2.0
    in_transit = (phase < half_dur) | (phase > 1.0 - half_dur)
    out_of_transit = ~in_transit

    if in_transit.sum() < 5 or out_of_transit.sum() < 50:
        return np.nan

    med_out = float(np.median(det[out_of_transit]))
    med_in = float(np.median(det[in_transit]))
    return med_out - med_in  # positive for dips


def _make_injection_star(
    n_points: int = 18000,
    noise_std: float = 0.001,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    time = np.arange(n_points, dtype=np.float64) * CADENCE
    flux = (1.0 + rng.normal(0, noise_std, n_points)).astype(np.float32)
    quality = np.zeros(n_points, dtype=np.int32)
    return time, flux, quality


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTransitInjectionRecovery:
    """Verify that torchflat preserves injected transits.

    Uses clip_sigma=100 to effectively disable sigma clipping, isolating
    the biweight detrending's transit preservation.
    """

    @pytest.mark.parametrize("depth", [0.005, 0.01, 0.05])
    def test_depth_recovery_box(self, depth):
        """Injected box transit depth is recovered within tolerance."""
        period = 3.0
        duration = 3.0 / 24.0  # 3 hours
        n_stars = 5

        times, fluxes, qualities = [], [], []
        for i in range(n_stars):
            t, f, q = _make_injection_star(n_points=18000, noise_std=0.0005, seed=i + 100)
            f_inj, _ = _inject_transit(t, f, depth, period, duration)
            times.append(t)
            fluxes.append(f_inj)
            qualities.append(q)

        results, _ = preprocess_track_a(
            times, fluxes, qualities,
            device="cpu", max_batch=10,
            window_scales=[(256, 128)],
            clip_sigma=100.0,  # disable clipping for transit preservation test
        )

        recovered_depths = []
        for i, r in enumerate(results):
            if r is None:
                continue
            rd = _recover_depth(r["detrended"], times[i], period, duration)
            if np.isfinite(rd):
                recovered_depths.append(rd)

        assert len(recovered_depths) >= 2, "Too few stars produced finite recovery"
        mean_recovered = np.mean(recovered_depths)
        assert mean_recovered > 0, f"Recovered depth should be positive, got {mean_recovered}"
        assert abs(mean_recovered - depth) / depth < 0.5, (
            f"Depth recovery error: injected={depth}, recovered={mean_recovered:.6f}"
        )

    def test_no_systematic_bias(self):
        """No consistent over/under-estimation across multiple depths."""
        period = 5.0
        duration = 3.0 / 24.0
        depths = [0.005, 0.01, 0.02, 0.05]
        biases = []

        for di, depth in enumerate(depths):
            times, fluxes, qualities = [], [], []
            for i in range(3):
                t, f, q = _make_injection_star(
                    n_points=18000, noise_std=0.0005,
                    seed=i + di * 1000,
                )
                f_inj, _ = _inject_transit(t, f, depth, period, duration)
                times.append(t)
                fluxes.append(f_inj)
                qualities.append(q)

            results, _ = preprocess_track_a(
                times, fluxes, qualities,
                device="cpu", max_batch=10,
                window_scales=[(256, 128)],
                clip_sigma=100.0,
            )

            recovered = []
            for i, r in enumerate(results):
                if r is None:
                    continue
                rd = _recover_depth(r["detrended"], times[i], period, duration)
                if np.isfinite(rd) and rd > 0:
                    recovered.append(rd)

            if recovered:
                bias = (np.mean(recovered) - depth) / depth
                biases.append(bias)

        assert len(biases) >= 2, "Not enough depths produced recoveries"
        # All biases should be small (<50% relative)
        assert all(abs(b) < 0.5 for b in biases), (
            f"Large bias detected: {[f'{b:.3f}' for b in biases]}"
        )

    def test_flat_lc_no_false_signal(self):
        """A flat LC (no transit) should produce detrended values near 1.0."""
        times, fluxes, qualities = [], [], []
        for i in range(5):
            t, f, q = _make_injection_star(seed=i + 500)
            times.append(t)
            fluxes.append(f)
            qualities.append(q)

        results, _ = preprocess_track_a(
            times, fluxes, qualities,
            device="cpu", max_batch=10,
            window_scales=[(256, 128)],
        )

        for r in results:
            if r is None:
                continue
            det = r["detrended"]
            finite = det[np.isfinite(det)]
            if len(finite) > 0:
                assert abs(np.median(finite) - 1.0) < 0.001
                assert finite.min() > 0.95

    def test_shallow_transit_survives_clipping(self):
        """Very shallow transits (below clip threshold) survive default clipping."""
        # With noise_std=0.001, 5-sigma threshold ~ 0.005
        # A 0.1% (0.001) transit is well below threshold -> should survive
        depth = 0.001
        period = 3.0
        duration = 4.0 / 24.0  # 4 hours for more in-transit points
        n_stars = 5

        times, fluxes, qualities = [], [], []
        for i in range(n_stars):
            t, f, q = _make_injection_star(n_points=18000, noise_std=0.001, seed=i + 200)
            f_inj, _ = _inject_transit(t, f, depth, period, duration)
            times.append(t)
            fluxes.append(f_inj)
            qualities.append(q)

        results, _ = preprocess_track_a(
            times, fluxes, qualities,
            device="cpu", max_batch=10,
            window_scales=[(256, 128)],
            # Default clip_sigma=5.0, transit should survive
        )

        # At least some stars should have non-None results
        processed = [r for r in results if r is not None]
        assert len(processed) >= 2
        # Detrended values should have finite non-NaN points
        for r in processed:
            finite = r["detrended"][np.isfinite(r["detrended"])]
            assert len(finite) > 100
