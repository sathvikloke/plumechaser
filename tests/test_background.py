"""Background climatology + robust z-score tests (pure numpy)."""

from __future__ import annotations

import numpy as np
import pytest

from plumechaser.detect.background import (
    robust_zscores,
    rolling_background,
    threshold_mask,
)


def _synthetic_stack(t: int = 20, h: int = 8, w: int = 8, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 1800.0 + rng.normal(0.0, 8.0, size=(t, h, w))


def test_anomaly_detected_and_baseline_uncontaminated():
    stack = _synthetic_stack()
    # Inject a strong enhancement on the final day at (3, 4).
    stack[-1, 3, 4] += 120.0
    med, mad = rolling_background(stack, window_days=30, exclude_center=True)
    z = robust_zscores(stack, med, mad)

    assert z[-1, 3, 4] > 5.0, "injected anomaly must clear detection threshold"
    # Background pixels elsewhere stay quiet. Bound accounts for the extreme
    # value of ~1200 correlated-noise samples (~N(0,1) modified z-scores).
    assert np.nanmax(z[:-1]) < 4.5, "no false anomalies in clean period"
    # Exclusion matters: the anomaly's own baseline must not include itself.
    med_inc, _ = rolling_background(stack, window_days=30, exclude_center=False)
    assert med[-1, 3, 4] < med_inc[-1, 3, 4], (
        "excluding centre day should lower the anomaly pixel's baseline"
    )


def test_mad_floor_prevents_explosion():
    stack = _synthetic_stack(t=6)
    stack[:] = 1800.0  # zero-variance scene => MAD == 0 everywhere
    med, mad = rolling_background(stack, window_days=5)
    z = robust_zscores(stack, med, mad, floor_mad_ppb=2.0)
    assert np.isnan(z).all(), "degenerate windows must yield NaN, not inf"


def test_threshold_mask_semantics():
    z = np.array([[np.nan, 2.9], [3.0, 10.0]])
    mask = threshold_mask(z, z_threshold=3.0)
    assert mask.tolist() == [[False, False], [True, True]]


def test_window_too_short_raises():
    with pytest.raises(ValueError):
        rolling_background(np.zeros((3, 4, 4)), window_days=0)


def test_robustness_to_outlier_day():
    """A single corrupted early day must not poison later backgrounds."""
    stack = _synthetic_stack()
    stack[2] += 500.0  # sensor glitch on one full frame
    med, mad = rolling_background(stack, window_days=30, exclude_center=True)
    z_after = robust_zscores(stack[10:], med[10:], mad[10:])
    assert np.nanmax(z_after) < 4.5, "median/MAD should absorb one bad frame"
