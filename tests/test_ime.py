"""IME quantification tests against hand-computed values."""

from __future__ import annotations

import numpy as np
import pytest

from plumechaser.retrieve.ime import (
    effective_wind_speed,
    integrated_mass_kg,
    plume_length_m,
    quantitate,
    source_rate_q,
)


def test_effective_wind_speed_formula():
    # Varon et al. 2021: Ueff = 0.33*U10 + 0.45
    assert effective_wind_speed(4.0) == pytest.approx(1.77)
    assert effective_wind_speed(0.0) == pytest.approx(0.45)
    with pytest.raises(ValueError):
        effective_wind_speed(-1.0)


def test_column_mass_and_ime_hand_computed():
    d_xch4 = np.full((4, 4), np.nan)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True          # 4 pixels at 1000 ppb each
    d_xch4[mask] = 1000.0

    ime = integrated_mass_kg(d_xch4, mask, pixel_area_m2=400.0, surface_pressure_hpa=1013.0)
    # per-pixel mass = 1000e-9 * (16.043/28.965) * (101300/9.80665) kg/m2 * 400 m2
    col_air = 1013.0 * 100.0 / 9.80665
    per_px = 1000e-9 * (16.043 / 28.965) * col_air * 400.0
    assert ime == pytest.approx(4 * per_px, rel=1e-9)


def test_source_rate_q_hand_computed():
    q = source_rate_q(ime_kg=22.9, ueff_ms=1.77, length_m=63.25)
    assert q == pytest.approx(22.9 * 1.77 * 3600 / 63.25, rel=1e-6)


def test_plume_length():
    assert plume_length_m(n_pixels=10, pixel_area_m2=400.0) == pytest.approx(np.sqrt(4000.0))
    with pytest.raises(ValueError):
        plume_length_m(0, 400.0)


def test_quantitate_ci_brackets_point_estimate():
    rng = np.random.default_rng(42)
    grid = rng.normal(0.0, 5.0, size=(40, 40))
    mask = np.zeros_like(grid, dtype=bool)
    mask[15:25, 18:26] = True       # 80 pixels
    grid[mask] += 900.0             # strong coherent enhancement

    res = quantitate(
        grid,
        mask,
        u10_ms=4.0,
        pixel_area_m2=400.0,
        mc_samples=300,
        seed=123,
        surface_pressure_hpa=1013.0,
    )
    assert res.ci_low < res.q_kg_h < res.ci_high
    assert res.n_pixels == 80
    assert res.length_m == pytest.approx(np.sqrt(80 * 400.0))
    # Determinism under a fixed seed.
    res2 = quantitate(
        grid,
        mask,
        u10_ms=4.0,
        pixel_area_m2=400.0,
        mc_samples=300,
        seed=123,
        surface_pressure_hpa=1013.0,
    )
    assert (res2.ci_low, res2.ci_high) == (res.ci_low, res.ci_high)


def test_quantitate_mask_shape_enforced():
    with pytest.raises(ValueError):
        integrated_mass_kg(
            np.zeros((4, 4)), np.zeros((3, 4), dtype=bool), pixel_area_m2=400.0
        )
