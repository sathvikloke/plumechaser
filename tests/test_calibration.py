"""RTM-derived calibration and the calibration-independent gate anchor.

The stored fits in config/rtm_calibration.json were measured by
scripts/calibrate_alpha.py against marss2l's RTM LUT; these tests pin the
properties the retrieval and the honesty gate depend on.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from plumechaser.config import load_config
from plumechaser.retrieve.calibration import (
    CalibrationError,
    load_calibration,
    simplified_c1,
)
from plumechaser.retrieve.gates import sigma_ppb_limit_for_scale

CAL_PATH = "config/rtm_calibration.json"


@pytest.fixture(scope="module")
def cal():
    return load_calibration(CAL_PATH)


def test_calibration_covers_the_geometry_we_observe_at(cal):
    assert "S2A" in cal.satellites and "S2B" in cal.satellites
    assert cal.sza_grid.min() <= 10.0
    assert cal.sza_grid.max() >= 70.0
    assert cal.vza_grid.min() <= 0.0
    assert cal.vza_grid.max() >= 10.0
    # the cubic fit has to actually describe the curve
    assert cal.max_relative_residual < 0.05


def test_zero_log_ratio_is_zero_enhancement(cal):
    assert cal.ppb_from_log_ratio(0.0, "S2B", 30.0, 5.0) == pytest.approx(0.0)


def test_curve_is_monotonic_and_superlinear(cal):
    """The RTM is markedly non-linear; a linear alpha understates big plumes."""
    x = np.linspace(0.0, 0.15, 40)
    ppb = cal.ppb_from_log_ratio(x, "S2B", 30.0, 10.0)
    assert np.all(np.diff(ppb) > 0)

    c1 = cal.c1("S2B", 30.0, 10.0)
    linear_only = c1 * x[-1]
    assert ppb[-1] > linear_only * 1.1  # >10% above the linear extrapolation


def test_noise_stays_symmetric_about_zero(cal):
    """Negative excursions must not be rectified into a positive bias.

    A MAD-based sigma over a rectified field would be badly wrong, and the
    honesty gate is computed from exactly that statistic.
    """
    pos = cal.ppb_from_log_ratio(0.01, "S2B", 30.0, 5.0)
    neg = cal.ppb_from_log_ratio(-0.01, "S2B", 30.0, 5.0)
    assert neg == pytest.approx(-pos)


def test_simplified_chain_understates_columns_by_the_measured_factor(cal):
    """The audit's headline number, pinned as a regression test."""
    c1_simple = simplified_c1(alpha_b12_per_ppb=1.2e-4, alpha_b11_per_ppb=3.0e-5)
    assert c1_simple == pytest.approx(11111.1, rel=1e-4)

    factors = [
        cal.c1(sat, sza, vza) / c1_simple
        for sat in cal.satellites
        for sza in cal.sza_grid
        for vza in cal.vza_grid
    ]
    assert min(factors) == pytest.approx(2.5, abs=0.3)
    assert max(factors) == pytest.approx(6.3, abs=0.3)
    # the spread is why a single scalar correction was rejected
    assert max(factors) / min(factors) > 2.0


def test_sensitivity_increases_with_solar_zenith(cal):
    """Longer slant path at high SZA means fewer ppb per unit ratio change."""
    low = cal.c1("S2B", 10.0, 5.0)
    high = cal.c1("S2B", 70.0, 5.0)
    assert high < low


def test_viewing_zenith_barely_matters(cal):
    """VZA spans only 0-10 deg for S2; it should be a second-order term."""
    a = cal.c1("S2A", 30.0, 0.0)
    b = cal.c1("S2A", 30.0, 10.0)
    assert abs(a - b) / a < 0.05


def test_geometry_outside_the_grid_clamps_rather_than_extrapolates(cal):
    edge = cal.c1("S2B", 70.0, 10.0)
    beyond = cal.c1("S2B", 89.0, 10.0)
    assert beyond == pytest.approx(edge)


def test_unknown_satellite_falls_back_instead_of_failing(cal):
    """S2C shares the MSI design and must not abort a campaign."""
    assert cal.c1("S2C", 30.0, 5.0) > 0


def test_gate_anchor_reproduces_the_pre_registered_80_ppb():
    """The frozen operating point must be unchanged by the reparameterisation."""
    cfg = load_config("config/default.yaml")
    c1_simple = simplified_c1(cfg.mbmp.alpha_b12_per_ppb, cfg.mbmp.alpha_b11_per_ppb)

    limit = sigma_ppb_limit_for_scale(c1_simple, cfg.gates.sigma_log_ratio_limit)

    assert limit == pytest.approx(cfg.gates.sigma_col_ppb_limit, rel=1e-6)


def test_gate_anchor_scales_up_on_the_rtm_scale(cal):
    """Same physical noise, RTM calibration -> a proportionally larger ppb limit."""
    cfg = load_config("config/default.yaml")
    rtm_limit = sigma_ppb_limit_for_scale(
        cal.c1("S2B", 28.0, 10.0), cfg.gates.sigma_log_ratio_limit
    )
    # ~500 ppb, versus 80 ppb on the simplified chain
    assert 300.0 < rtm_limit < 700.0
    assert rtm_limit > cfg.gates.sigma_col_ppb_limit


def test_gate_anchor_rejects_nonsense():
    with pytest.raises(ValueError):
        sigma_ppb_limit_for_scale(0.0, 0.0072)
    with pytest.raises(ValueError):
        sigma_ppb_limit_for_scale(11111.0, -1.0)


def test_missing_calibration_is_a_clear_error(tmp_path):
    with pytest.raises(CalibrationError, match="not found"):
        load_calibration(tmp_path / "nope.json")


def test_malformed_calibration_is_a_clear_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    with pytest.raises(CalibrationError):
        load_calibration(bad)

    holey = tmp_path / "holey.json"
    holey.write_text(json.dumps({"entries": [
        {"satellite": "S2B", "sza": 10.0, "vza": 0.0, "c1": 1.0, "c2": 0.0, "c3": 0.0},
        {"satellite": "S2B", "sza": 20.0, "vza": 5.0, "c1": 1.0, "c2": 0.0, "c3": 0.0},
    ]}))
    with pytest.raises(CalibrationError, match="incomplete"):
        load_calibration(holey)
