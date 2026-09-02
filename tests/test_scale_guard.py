"""The guard against the 2026-08-25 root cause.

A 10^4 input-scale error survived a full campaign because band ratios are
invariant to it. These tests pin the guard that makes it loud instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from plumechaser.data.scale import (
    assert_dn_scale,
    infer_reflectance_scale,
)


def _scene(rng, scale=1.0):
    """A plausible arid-surface SWIR window: reflectance 0.15-0.45."""
    return rng.uniform(0.15, 0.45, size=(64, 64)) * scale


def test_dn_scene_is_accepted():
    rng = np.random.default_rng(20270307)
    v = assert_dn_scale(_scene(rng, 1e4), "target")
    assert v.is_dn
    assert v.p99 > 1000


def test_reflectance_scene_is_rejected_with_the_fix_in_the_message():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="0-1 reflectance") as exc:
        assert_dn_scale(_scene(rng), "target")
    # the message must say what to do, not just that something is wrong
    assert "multiply by 1e4" in str(exc.value)
    assert "invariant" in str(exc.value)


def test_the_exact_regression_the_guard_exists_for():
    """mars2l_demo divided DN by 1e4 before handing bands to MARS-S2L."""
    rng = np.random.default_rng(2)
    dn = _scene(rng, 1e4)

    assert infer_reflectance_scale(dn).scale == "dn"
    assert infer_reflectance_scale(dn / 1e4).scale == "reflectance"


def test_band_ratio_really_is_invariant_so_the_guard_is_the_only_defence():
    """Justifies the guard: the downstream signal cannot reveal the error."""
    rng = np.random.default_rng(3)
    b11 = _scene(rng, 1e4)
    b12 = _scene(rng, 1e4)

    ratio_dn = b12 / b11
    ratio_refl = (b12 / 1e4) / (b11 / 1e4)

    np.testing.assert_allclose(ratio_dn, ratio_refl, rtol=1e-12)


def test_saturated_pixels_do_not_flip_the_verdict():
    """p99, not max — a few corrupt pixels must not rescue a bad array."""
    rng = np.random.default_rng(4)
    refl = _scene(rng)
    refl.flat[:5] = 9999.0  # a handful of garbage pixels

    assert infer_reflectance_scale(refl).scale == "reflectance"


def test_all_dark_window_is_indeterminate_not_silently_passed():
    dark = np.full((32, 32), 5.0)
    v = infer_reflectance_scale(dark)
    assert v.scale == "indeterminate"
    with pytest.raises(ValueError, match="cannot determine"):
        assert_dn_scale(dark)


def test_empty_and_all_nan_are_indeterminate():
    assert infer_reflectance_scale(np.full((8, 8), np.nan)).scale == "indeterminate"
    assert infer_reflectance_scale(np.array([])).scale == "indeterminate"


def test_nodata_does_not_break_inference():
    rng = np.random.default_rng(5)
    dn = _scene(rng, 1e4)
    dn[:20, :] = np.nan  # a partial granule

    v = infer_reflectance_scale(dn)
    assert v.is_dn
    assert v.finite_fraction == pytest.approx(44 / 64, rel=1e-6)
