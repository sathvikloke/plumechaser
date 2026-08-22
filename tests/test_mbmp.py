"""MBMP retrieval tests: artifact cancellation + synthetic-plume recovery."""

from __future__ import annotations

import numpy as np
import pytest

from plumechaser.retrieve.mbmp import (
    column_mass_kg_m2,
    log_band_ratio,
    mbmp_enhancement_ppb,
    plume_mask,
    robust_scene_sigma,
)

ALPHA_11, ALPHA_12 = 3.0e-5, 1.2e-4  # config defaults (per ppb)


def _scene_with_texture(h=32, w=32, seed=3):
    rng = np.random.default_rng(seed)
    texture = 0.9 + 0.2 * rng.random((h, w))          # static surface structure
    b11 = 0.25 * texture
    b12 = 0.15 * texture
    return b11, b12


def test_artifact_cancellation_and_recovery():
    """Same-surface passes cancel; injected CH4 recovers to ~truth."""
    b11_ref, b12_ref = _scene_with_texture()
    b11_t, b12_t = b11_ref.copy(), b12_ref.copy()

    truth = np.zeros((32, 32))
    yy, xx = np.mgrid[0:32, 0:32]
    plume = 800.0 * np.exp(-(((yy - 16) ** 2) + ((xx - 20) ** 2)) / 18.0)
    truth += plume

    # CH4 transmittance dips both bands by their alpha coefficients.
    b12_t *= np.exp(-ALPHA_12 * truth)
    b11_t *= np.exp(-ALPHA_11 * truth)

    d_xch4 = mbmp_enhancement_ppb(b11_t, b12_t, b11_ref, b12_ref, ALPHA_11, ALPHA_12)
    peak_cell = np.unravel_index(np.nanargmax(truth), truth.shape)
    assert d_xch4[peak_cell] == pytest.approx(truth[peak_cell], rel=0.15)
    # Static texture cancels exactly in the true zero-gas far field.
    # (Pixels with truth in [0.005, 1) hold real tail gas; recovering it is
    # correct behaviour, not a cancellation error -- see residuals ~1e-12.)
    far = truth < 0.005
    assert np.nanmax(np.abs(d_xch4[far])) < 0.01


def test_positive_enhancement_sign():
    """More CH4 in target pass => positive enhancement (both bands attenuate)."""
    b11_ref, b12_ref = _scene_with_texture()
    b11_t, b12_t = b11_ref.copy(), b12_ref.copy()
    b12_t *= np.exp(-ALPHA_12 * 1000.0)
    b11_t *= np.exp(-ALPHA_11 * 1000.0)
    out = mbmp_enhancement_ppb(b11_t, b12_t, b11_ref, b12_ref, ALPHA_11, ALPHA_12)
    assert np.allclose(out, 1000.0, rtol=1e-6)


def test_alpha_ordering_enforced():
    with pytest.raises(ValueError):
        mbmp_enhancement_ppb(
            np.ones((4, 4)), np.ones((4, 4)),
            np.ones((4, 4)), np.ones((4, 4)),
            alpha_b11_per_ppb=1e-4, alpha_b12_per_ppb=5e-5,
        )


def test_plume_mask_finds_blob_and_despeckles():
    field = np.zeros((40, 40))
    field[10:14, 20:26] = 120.0        # coherent blob (24 px)
    field[33, 3] = 150.0               # isolated single-pixel spike
    mask = plume_mask(field, threshold_sigma=3.0, median_size=3)
    assert mask[11:13, 21:25].all(), "blob interior must survive despeckling"
    assert not mask[33, 3], "median filter should kill the lone spike"
    assert mask.sum() >= 12, "majority of blob area preserved"


def test_robust_scene_sigma_floor():
    assert robust_scene_sigma(np.full((8, 8), 50.0)) >= 2.0


def test_column_mass_monotone_in_pressure():
    lo = column_mass_kg_m2(500.0, surface_pressure_hpa=900.0)
    hi = column_mass_kg_m2(500.0, surface_pressure_hpa=1050.0)
    assert float(lo) > 0 and float(hi) > float(lo)


def test_log_band_ratio_nan_on_zero():
    b11 = np.array([[1.0, 0.0]])
    b12 = np.array([[1.0, 1.0]])
    out = log_band_ratio(b11, b12)
    assert np.isfinite(out[0, 0]) and np.isnan(out[0, 1])
