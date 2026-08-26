"""Arithmetic audit of the production flux path.

The regression anchors below are the numbers marss2l actually wrote into
bundles/EVT-*-MARSS2L/provenance.json during the 2026-08-23 campaign. They
are stored rounded to 2 dp, hence the 1% tolerances.
"""

from __future__ import annotations

import numpy as np
import pytest

from plumechaser.retrieve.flux_audit import (
    audit_q_output,
    first_principles_kg_m2_per_ppb,
    marss2l_kg_m2_per_ppb,
    q_kg_h,
)

# Korpezhe 2026-08-05, catalog 26 t/h (bundle EVT-20260805-K26-MARSS2L-v2)
KORPEZHE_Q = {
    "Q": 336620.76,
    "L": 4416.88,
    "npix_plume": 48772,
    "IME": 243315.33,
    "sum_enhancement": 849479963.88,
    "pixel_size": 400.0,
    "u_eff": 1.7,
}
# Permian 2026-04-27, catalog 82 t/h (bundle EVT-20260427-P82-MARSS2L)
PERMIAN_Q = {
    "Q": 478172.03,
    "L": 3793.68,
    "npix_plume": 35980,
    "IME": 224268.44,
    "sum_enhancement": 782982093.36,
    "pixel_size": 400.0,
    "u_eff": 2.25,
}


def test_unit_path_agrees_with_first_principles():
    """marss2l's 8000 m / 22.4 L-per-mol path vs our hydrostatic P/g column.

    Agreement here is what clears 'units' as a suspect for the overestimate.
    """
    marss = marss2l_kg_m2_per_ppb()
    ours = first_principles_kg_m2_per_ppb(surface_pressure_hpa=1013.0)
    assert marss == pytest.approx(ours, rel=0.01)
    # sanity: ~5.7e-6 kg per m^2 per ppb of column enhancement
    assert 5.0e-6 < marss < 6.5e-6


def test_forward_model_reproduces_recorded_flux():
    for q in (KORPEZHE_Q, PERMIAN_Q):
        a = audit_q_output(q)
        recomputed = q_kg_h(
            a.mean_enhancement_ppb, a.n_mask_px, a.pixel_area_m2, a.u_eff_ms
        )
        assert recomputed == pytest.approx(q["Q"], rel=0.01)


def test_recorded_ime_and_length_are_internally_consistent():
    """No reconstruction notes means marss2l's own bookkeeping is coherent."""
    for q in (KORPEZHE_Q, PERMIAN_Q):
        assert audit_q_output(q).notes == ()


def test_mean_enhancement_is_the_implausible_term():
    a = audit_q_output(KORPEZHE_Q, event_id="korpezhe", catalog_rate_t_h=26.0)

    assert a.q_t_h == pytest.approx(336.6, rel=0.01)
    assert a.ratio_to_catalog == pytest.approx(12.9, rel=0.02)
    # ~2200 ppb sustained across ~20 km^2
    assert a.mean_enhancement_ppb == pytest.approx(2177.0, rel=0.01)
    assert a.plume_area_km2 == pytest.approx(19.5, rel=0.01)
    # the column would have to be more than doubled over that whole area
    assert a.column_enhancement_factor > 1.0
    # a catalog-consistent plume over the same mask needs ~170 ppb
    assert a.mean_ppb_for_catalog_rate == pytest.approx(168.0, rel=0.05)


def test_flux_scales_as_sqrt_area_at_fixed_enhancement():
    """The mask is a first-order lever on Q, not a second-order detail."""
    base = q_kg_h(500.0, 10_000, 400.0, 2.0)
    quadrupled = q_kg_h(500.0, 40_000, 400.0, 2.0)
    assert quadrupled == pytest.approx(2.0 * base, rel=1e-9)

    a = audit_q_output(KORPEZHE_Q, catalog_rate_t_h=26.0)
    # closing a 12.9x gap by mask alone needs a ~167x smaller area
    assert a.mask_shrink_for_catalog_rate() == pytest.approx(12.9**2, rel=0.05)


def test_flux_is_linear_in_enhancement():
    assert q_kg_h(1000.0, 5_000, 400.0, 2.0) == pytest.approx(
        2.0 * q_kg_h(500.0, 5_000, 400.0, 2.0), rel=1e-9
    )


def test_mask_fraction_uses_the_retrieval_window():
    # mars2l_demo default: half_km=8 at 20 m -> 800 x 800 window
    a = audit_q_output(KORPEZHE_Q, catalog_rate_t_h=26.0, window_px=800 * 800)
    assert a.mask_fraction == pytest.approx(48772 / 640000, rel=1e-6)
    # under the 15% mask gate, so the mask gate alone would NOT have caught it
    assert a.mask_fraction < 0.15


def test_reconstruction_mismatch_is_flagged():
    bad = dict(KORPEZHE_Q, IME=KORPEZHE_Q["IME"] * 3.0)
    assert any("IME reconstruction mismatch" in n for n in audit_q_output(bad).notes)

    bad_l = dict(KORPEZHE_Q, L=123.0)
    assert any("not sqrt(plume area)" in n for n in audit_q_output(bad_l).notes)


def test_missing_keys_raise():
    with pytest.raises(KeyError, match="missing keys"):
        audit_q_output({"Q": 1.0})


def test_forward_model_rejects_degenerate_masks():
    with pytest.raises(ValueError):
        q_kg_h(100.0, 0, 400.0, 2.0)
    with pytest.raises(ValueError):
        q_kg_h(100.0, 10, 0.0, 2.0)


def test_audit_serialises():
    d = audit_q_output(PERMIAN_Q, event_id="permian", catalog_rate_t_h=82.0).as_dict()
    assert d["event_id"] == "permian"
    assert d["ratio_to_catalog"] == pytest.approx(5.83, rel=0.02)
    assert np.isfinite(d["mean_enhancement_ppb"])
