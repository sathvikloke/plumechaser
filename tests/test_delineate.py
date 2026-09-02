"""Delineation is the dominant absolute-flux error term, so it gets real tests.

The synthetic scenes here are not decorative: two of them reproduce the
measured 2026-08-25 controlled-release failure numerically — a 2,626-pixel
(1.05 km^2) mask with an 11,718 ppb mean in-mask enhancement, against a
**metered** 7.18 t/h release, giving a 45x flux overestimate. The enhancement
of the true plume is not hand-picked: it is inverted from the ``flux_audit``
identity so that the true mask carries exactly the metered rate. That makes
"did delineation recover the truth?" a fair question to ask of the output.
"""

from __future__ import annotations

import numpy as np
import pytest

from plumechaser.retrieve.delineate import (
    RULE_ORDER,
    delineate_plume,
    downwind_unit_pixel_vector,
    rejection_report,
)
from plumechaser.retrieve.flux_audit import (
    BACKGROUND_CH4_PPB,
    marss2l_kg_m2_per_ppb,
    q_kg_h,
)
from plumechaser.retrieve.gates import evaluate_gates
from plumechaser.retrieve.ime import effective_wind_speed

# --- the measured Ehrenberg 2021-10-19 failure ------------------------------
PIXEL_M = 20.0  # Sentinel-2 SWIR
SHAPE = (320, 320)
SOURCE_RC = (160.0, 80.0)
U10_MS = 2.6
WIND_MS = (U10_MS, 0.0)  # blowing toward the east
METERED_TRUTH_KG_H = 7_180.0
MEASURED_MASK_PX = 2_626
MEASURED_MEAN_PPB = 11_718.0
MEASURED_FLUX_RATIO = 45.0

U_EFF_MS = effective_wind_speed(U10_MS)
KG_M2_PER_PPB = marss2l_kg_m2_per_ppb()


def _blank() -> np.ndarray:
    return np.zeros(SHAPE, dtype=bool)


def _true_plume() -> np.ndarray:
    """A compact, laterally spreading plume anchored at the source, blowing east."""
    d_row = (np.arange(SHAPE[0])[:, None] - SOURCE_RC[0]).astype(float)
    d_col = (np.arange(SHAPE[1])[None, :] - SOURCE_RC[1]).astype(float)
    wedge = (d_col >= 0) & (d_col <= 55) & (np.abs(d_row) <= 1.5 + 0.10 * d_col)
    return np.broadcast_to(wedge, SHAPE).copy()


def _mean_ppb_for_rate(n_px: int, rate_kg_h: float) -> float:
    """Invert ``Q = 3600 U_eff C mean_ppb sqrt(A)`` for the mean enhancement."""
    return rate_kg_h / (
        3600.0 * U_EFF_MS * KG_M2_PER_PPB * np.sqrt(n_px * PIXEL_M * PIXEL_M)
    )


def _ehrenberg_scene(seed: int = 20260825):
    """The measured failure: a true plume buried in a high-amplitude artifact.

    The artifact is a compact bright patch (the release rig / hard standing),
    placed off the plume axis and not touching it. It is 14,370 ppb because
    that is what makes the combined mask reproduce the recorded 11,718 ppb
    mean over 2,626 pixels.
    """
    plume = _true_plume()
    rr = np.arange(SHAPE[0])[:, None]
    cc = np.arange(SHAPE[1])[None, :]
    artifact = ((rr - 70.0) ** 2 + (cc - 100.0) ** 2) <= 26.0**2
    artifact = np.asarray(artifact) & ~plume

    n_plume = int(plume.sum())
    plume_ppb = _mean_ppb_for_rate(n_plume, METERED_TRUTH_KG_H)
    artifact_ppb = (
        MEASURED_MASK_PX * MEASURED_MEAN_PPB - n_plume * plume_ppb
    ) / int(artifact.sum())

    field = np.random.default_rng(seed).normal(0.0, 40.0, size=SHAPE)
    field[plume] = plume_ppb
    field[artifact] = artifact_ppb
    return plume, artifact, plume | artifact, field, plume_ppb


def _ablation_scene(seed: int = 7):
    """One blob per rule: each is removable by exactly one of the constraints.

    * ``amplitude``   - a bright lobe fused to the plume: connected, downwind,
      close. Only its 14,370 ppb amplitude betrays it.
    * ``connectivity``- a moderate isolated patch, on-axis and within range.
      Only its detachment from the source betrays it.
    * ``sector``      - a moderate lobe fused to the plume but sitting
      **upwind** of the source.
    * ``distance``    - a moderate tail fused to the plume running far beyond
      the advection bound.
    """
    plume = _true_plume()
    amplitude = _blank()
    amplitude[148:172, 110:134] = True
    amplitude &= ~plume
    connectivity = _blank()
    connectivity[190:201, 150:171] = True
    sector = _blank()
    sector[155:166, 54:81] = True
    sector &= ~plume
    distance = _blank()
    distance[158:163, 136:230] = True
    distance &= ~plume

    blobs = {
        "amplitude": amplitude,
        "connectivity": connectivity,
        "sector": sector,
        "distance": distance,
    }
    raw = plume.copy()
    for blob in blobs.values():
        raw |= blob

    field = np.random.default_rng(seed).normal(0.0, 40.0, size=SHAPE)
    for blob in blobs.values():
        field[blob] = 1_361.0  # the measured Casa Grande diffuse-artifact level
    field[amplitude] = 14_370.0
    field[plume] = _mean_ppb_for_rate(int(plume.sum()), METERED_TRUTH_KG_H)
    return plume, blobs, raw, field


# --- wind geometry ----------------------------------------------------------


def test_downwind_unit_vector_converts_wind_to_raster_directions():
    # Wind blowing east -> downwind is +columns, no row component.
    d_row, d_col = downwind_unit_pixel_vector((5.0, 0.0))
    assert d_col == pytest.approx(1.0)
    assert d_row == pytest.approx(0.0)

    # Wind blowing north -> downwind is -rows on a north-up raster.
    assert downwind_unit_pixel_vector((0.0, 5.0))[0] == pytest.approx(-1.0)
    assert downwind_unit_pixel_vector((0.0, 5.0), north_up=False)[0] == pytest.approx(1.0)

    # Always unit length.
    d_row, d_col = downwind_unit_pixel_vector((3.0, -4.0))
    assert np.hypot(d_row, d_col) == pytest.approx(1.0)
    assert d_row == pytest.approx(0.8)  # southward wind -> +rows on a north-up grid


def test_zero_or_malformed_wind_is_an_error_not_a_silent_direction():
    with pytest.raises(ValueError, match="positive, finite speed"):
        downwind_unit_pixel_vector((0.0, 0.0))
    with pytest.raises(ValueError, match="two components"):
        downwind_unit_pixel_vector([3.0])


# --- the headline cases -----------------------------------------------------


def test_compact_plume_kept_and_bright_artifact_blob_removed():
    plume, artifact, raw, field, _ = _ehrenberg_scene()

    res = delineate_plume(
        raw,
        enhancement_ppb=field,
        source_rc=SOURCE_RC,
        wind_vector_ms=WIND_MS,
        pixel_size_m=PIXEL_M,
    )

    assert int((res.mask & artifact).sum()) == 0, "artifact blob must not survive"
    recall = int((res.mask & plume).sum()) / int(plume.sum())
    assert recall > 0.98, f"true plume must be retained, recall={recall:.3f}"
    assert res.dominant_rule == "amplitude"


def test_ehrenberg_failure_shape_becomes_physically_plausible():
    """The measured 2,626 px / 11,718 ppb mask, delineated."""
    _, _, raw, field, plume_ppb = _ehrenberg_scene()

    # First confirm the synthetic really is the measured failure.
    assert int(raw.sum()) == pytest.approx(MEASURED_MASK_PX, rel=0.02)
    assert float(field[raw].mean()) == pytest.approx(MEASURED_MEAN_PPB, rel=0.02)
    assert float(raw.sum()) * PIXEL_M**2 / 1e6 == pytest.approx(1.05, rel=0.02)

    res = delineate_plume(
        raw,
        enhancement_ppb=field,
        source_rc=SOURCE_RC,
        wind_vector_ms=WIND_MS,
        pixel_size_m=PIXEL_M,
    )

    # Far smaller mask...
    assert res.n_kept_px < 0.25 * res.n_input_px
    assert res.area_shrink_factor > 4.0
    assert res.area_km2_after < 0.25
    # ...and a mean enhancement that a point source could actually produce.
    assert res.mean_ppb_after < BACKGROUND_CH4_PPB
    assert res.mean_ppb_after == pytest.approx(plume_ppb, rel=0.02)


def test_delineated_flux_lands_near_metered_truth():
    """The quantitative claim: 45x off becomes within a factor of ~1.2.

    Both fluxes are computed with ``flux_audit.q_kg_h`` on the same field, the
    same wind and the same pixel size. The only difference is the mask.
    """
    _, _, raw, field, _ = _ehrenberg_scene()

    q_raw = q_kg_h(
        float(field[raw].mean()), int(raw.sum()), PIXEL_M**2, U_EFF_MS
    )
    res = delineate_plume(
        raw,
        enhancement_ppb=field,
        source_rc=SOURCE_RC,
        wind_vector_ms=WIND_MS,
        pixel_size_m=PIXEL_M,
    )
    q_delineated = q_kg_h(
        res.mean_ppb_after, res.n_kept_px, res.pixel_area_m2, U_EFF_MS
    )

    ratio_raw = q_raw / METERED_TRUTH_KG_H
    ratio_delineated = q_delineated / METERED_TRUTH_KG_H

    # The raw mask reproduces the campaign's measured 45x overestimate.
    assert ratio_raw == pytest.approx(MEASURED_FLUX_RATIO, rel=0.1)
    # The delineated mask lands inside the ~2x band the published teams
    # (Harvard, Kayrros, LARS, SRON) achieved on these same pixels.
    assert 0.5 < ratio_delineated < 2.0
    # ...which is at least a 20x improvement in absolute log-distance terms.
    assert ratio_raw / ratio_delineated > 20.0


def test_delineation_turns_a_gate_failure_into_a_gate_pass():
    """Delineate first, gate second: the gates then judge a physical mask."""
    _, _, raw, field, _ = _ehrenberg_scene()

    before = evaluate_gates(field, raw)
    assert before.artifact_dominated
    assert any("unphysical" in reason for reason in before.reasons)

    res = delineate_plume(
        raw,
        enhancement_ppb=field,
        source_rc=SOURCE_RC,
        wind_vector_ms=WIND_MS,
        pixel_size_m=PIXEL_M,
    )
    after = evaluate_gates(field, res.mask)

    assert not after.artifact_dominated, after.reasons
    assert after.mean_enhancement_ppb < BACKGROUND_CH4_PPB


def test_upwind_blob_rejected_and_the_same_blob_downwind_is_kept():
    source = (60.0, 60.0)
    core = _blank()[:120, :120].copy()
    core[58:63, 58:71] = True  # a small body anchored at the source

    downwind_blob = np.zeros_like(core)
    downwind_blob[55:66, 70:90] = True
    upwind_blob = np.zeros_like(core)
    upwind_blob[55:66, 30:50] = True

    kept = {}
    for name, blob in (("downwind", downwind_blob), ("upwind", upwind_blob)):
        res = delineate_plume(
            core | blob,
            source_rc=source,
            wind_vector_ms=(3.0, 0.0),
            pixel_size_m=PIXEL_M,
        )
        kept[name] = int((res.mask & blob).sum()) / int(blob.sum())

    assert kept["downwind"] == pytest.approx(1.0)
    assert kept["upwind"] == 0.0


# --- rule attribution -------------------------------------------------------


def test_each_rule_is_individually_switchable_and_removes_its_own_blob():
    plume, blobs, raw, field = _ablation_scene()
    switches = {
        "amplitude": "apply_amplitude",
        "connectivity": "apply_source_connectivity",
        "sector": "apply_downwind_sector",
        "distance": "apply_distance_bound",
    }
    common = dict(
        enhancement_ppb=field,
        source_rc=SOURCE_RC,
        wind_vector_ms=WIND_MS,
        pixel_size_m=PIXEL_M,
    )

    def retained(res, blob):
        return int((res.mask & blob).sum()) / int(blob.sum())

    all_on = delineate_plume(raw, **common)
    assert retained(all_on, plume) > 0.98

    for name, switch in switches.items():
        off = delineate_plume(raw, **common, **{switch: False})
        # Disabling a rule leaks exactly its own blob back in...
        assert retained(off, blobs[name]) == pytest.approx(1.0), name
        # ...and nothing else changes: the other blobs stay as they were.
        for other in switches:
            if other != name:
                assert retained(off, blobs[other]) == pytest.approx(
                    retained(all_on, blobs[other])
                ), f"{name} off perturbed {other}"
        # The true plume survives every configuration.
        assert retained(off, plume) > 0.98, name

    # With every rule on, each blob is at least largely gone.
    assert retained(all_on, blobs["amplitude"]) == 0.0
    assert retained(all_on, blobs["connectivity"]) == 0.0
    # The sector keeps a near-source pad by design (the cone apex sits upwind),
    # and the distance bound only clips the far half of a tail that starts
    # inside the bound. Both are documented behaviour, not leakage.
    assert retained(all_on, blobs["sector"]) < 0.2
    assert retained(all_on, blobs["distance"]) < 0.7


def test_rejection_report_accounts_for_every_pixel():
    _, _, raw, field, _ = _ehrenberg_scene()
    res = delineate_plume(
        raw,
        enhancement_ppb=field,
        source_rc=SOURCE_RC,
        wind_vector_ms=WIND_MS,
        pixel_size_m=PIXEL_M,
    )
    report = rejection_report(res)

    assert report["balanced"] is True
    assert res.n_kept_px == res.n_input_px - res.n_dropped_px + res.n_added_px
    assert report["n_input_px"] == int(raw.sum())
    assert report["dominant_rule"] == "amplitude"

    names = [row["rule"] for row in report["rules"]]
    assert names == [r for r in RULE_ORDER if r in names]  # reported in rule order
    assert all(row["status"] in {"applied", "skipped"} for row in report["rules"])
    by_rule = {row["rule"]: row["dropped_px"] for row in report["rules"]}
    assert by_rule["amplitude"] == int((raw & (field > 2 * BACKGROUND_CH4_PPB)).sum())
    assert sum(by_rule.values()) == res.n_dropped_px


def test_implied_flux_factor_matches_the_flux_audit_identity():
    _, _, raw, field, _ = _ehrenberg_scene()
    res = delineate_plume(
        raw,
        enhancement_ppb=field,
        source_rc=SOURCE_RC,
        wind_vector_ms=WIND_MS,
        pixel_size_m=PIXEL_M,
    )
    q_raw = q_kg_h(float(field[raw].mean()), int(raw.sum()), PIXEL_M**2, U_EFF_MS)
    q_new = q_kg_h(res.mean_ppb_after, res.n_kept_px, res.pixel_area_m2, U_EFF_MS)

    assert res.implied_flux_factor == pytest.approx(q_new / q_raw, rel=1e-9)
    assert res.as_dict()["implied_flux_factor"] == pytest.approx(
        round(q_new / q_raw, 4)
    )


# --- individual techniques in isolation -------------------------------------


def test_amplitude_rule_drops_impossible_pixels_and_keeps_a_plume_core():
    mask = np.zeros((60, 60), bool)
    mask[20:40, 20:40] = True
    field = np.zeros((60, 60))
    field[mask] = 300.0
    field[20:25, 20:40] = 12_000.0  # 6.7x the ambient column: not methane
    field[30:32, 30:32] = np.nan
    field[35:37, 35:37] = -400.0  # a negative retrieval is not an enhancement

    res = delineate_plume(
        mask,
        enhancement_ppb=field,
        apply_morphology=False,
        apply_source_connectivity=False,
        apply_downwind_sector=False,
        apply_distance_bound=False,
    )

    assert res.dropped["amplitude"] == 100 + 4 + 4
    assert res.mean_ppb_after == pytest.approx(300.0)
    # The ceiling is a multiple of the ambient column and is caller-adjustable.
    loose = delineate_plume(
        mask,
        enhancement_ppb=field,
        max_enhancement_ratio=10.0,
        apply_morphology=False,
        apply_source_connectivity=False,
        apply_downwind_sector=False,
        apply_distance_bound=False,
    )
    assert loose.dropped["amplitude"] == 8  # only the NaNs and the negatives


def test_morphology_removes_specks_severs_bridges_and_fills_dropouts():
    mask = np.zeros((60, 80), bool)
    mask[20:40, 10:30] = True  # plume body
    mask[28:31, 18:21] = False  # a 9 px retrieval dropout inside it
    mask[29, 30:40] = True  # a one-pixel bridge...
    mask[20:40, 40:60] = True  # ...to a second bright surface
    mask[5, 5] = True  # specks
    mask[50, 70] = True

    res = delineate_plume(
        mask,
        apply_amplitude=False,
        apply_source_connectivity=False,
        apply_downwind_sector=False,
        apply_distance_bound=False,
    )

    assert res.added["morphology_fill"] == 9  # dropout repaired
    assert not res.mask[5, 5] and not res.mask[50, 70]  # specks gone
    assert not res.mask[29, 35]  # bridge severed
    assert res.mask[20:40, 10:30].sum() > 350  # body essentially intact

    # With connectivity switched back on, severing the bridge is what lets the
    # second surface be dropped: that is the two rules composing.
    linked = delineate_plume(
        mask, source_rc=(30.0, 20.0), apply_amplitude=False,
        apply_downwind_sector=False, apply_distance_bound=False,
    )
    assert int(linked.mask[:, 40:60].sum()) == 0


def test_distance_bound_derives_from_the_wind_and_can_be_overridden():
    mask = np.zeros((200, 400), bool)
    mask[98:103, 100:340] = True  # a 4,800 m streak from the source

    derived = delineate_plume(
        mask,
        source_rc=(100.0, 100.0),
        wind_vector_ms=(2.0, 0.0),
        pixel_size_m=PIXEL_M,
    )
    # 2 m/s x 900 s of advection = 1,800 m of plausible extent.
    assert derived.max_extent_m == pytest.approx(1_800.0)
    assert derived.dropped["distance_bound"] > 0
    assert derived.n_kept_px == pytest.approx(5 * 1_800 / PIXEL_M, rel=0.05)

    # A stronger wind carries the plume further, so the bound relaxes.
    windy = delineate_plume(
        mask, source_rc=(100.0, 100.0), wind_vector_ms=(6.0, 0.0), pixel_size_m=PIXEL_M
    )
    assert windy.max_extent_m > derived.max_extent_m
    assert windy.n_kept_px > derived.n_kept_px

    # An explicit bound wins over the derivation.
    explicit = delineate_plume(
        mask,
        source_rc=(100.0, 100.0),
        wind_vector_ms=(2.0, 0.0),
        pixel_size_m=PIXEL_M,
        max_extent_m=600.0,
    )
    assert explicit.max_extent_m == 600.0
    assert explicit.n_kept_px < derived.n_kept_px

    # A near-calm scene must not clip the source neighbourhood to nothing.
    calm = delineate_plume(
        mask,
        source_rc=(100.0, 100.0),
        wind_vector_ms=(0.001, 0.0),
        pixel_size_m=PIXEL_M,
    )
    assert calm.max_extent_m == pytest.approx(2.0 * PIXEL_M)
    assert calm.n_kept_px > 0


def test_source_connectivity_keeps_one_component_not_the_scene():
    """The Casa Grande mode: several bright pads, only one at the source."""
    mask = np.zeros((120, 120), bool)
    mask[58:63, 58:80] = True  # at the source, downwind
    mask[20:35, 60:90] = True  # unrelated pad
    mask[90:110, 65:95] = True  # another
    res = delineate_plume(
        mask,
        source_rc=(60.0, 60.0),
        wind_vector_ms=(3.0, 0.0),
        pixel_size_m=PIXEL_M,
        apply_downwind_sector=False,
        apply_distance_bound=False,
    )
    assert res.dropped["source_connectivity"] > 800
    assert int(res.mask[20:35, 60:90].sum()) == 0
    assert int(res.mask[90:110, 65:95].sum()) == 0
    assert int(res.mask[58:63, 58:80].sum()) > 100


def test_geometric_clipping_cannot_leave_an_island_detached_from_the_source():
    """An island that is downwind and in range, but only reachable off-cone.

    The island sits on the plume axis and inside the distance bound, so neither
    geometric rule touches it directly. It reaches the source only through a
    detour that the sector cut removes — which leaves it floating. A floating
    body is not a plume either, which is why connectivity is re-checked after
    the geometric clips rather than only before them.
    """
    mask = np.zeros((200, 200), bool)
    mask[78:83, 60:101] = True  # plume body, on the axis
    mask[78:83, 130:151] = True  # the island, also on the axis
    mask[83:141, 96:101] = True  # detour: south...
    mask[136:141, 96:146] = True  # ...east...
    mask[83:141, 141:146] = True  # ...and back north to the island

    res = delineate_plume(
        mask,
        source_rc=(80.0, 60.0),
        wind_vector_ms=(3.0, 0.0),
        pixel_size_m=PIXEL_M,
        sector_half_angle_deg=20.0,
    )

    assert res.dropped["downwind_sector"] > 0
    assert res.dropped.get("reconnect", 0) > 0, "severed island must be dropped"
    assert int(res.mask[78:83, 130:151].sum()) == 0
    assert int(res.mask[78:83, 60:101].sum()) > 190  # the body is untouched


def test_valid_mask_drops_unusable_pixels_first():
    mask = np.zeros((60, 60), bool)
    mask[20:40, 20:40] = True
    valid = np.ones((60, 60), bool)
    valid[20:25, :] = False  # a nodata band across the mask

    res = delineate_plume(
        mask,
        valid=valid,
        apply_amplitude=False,
        apply_morphology=False,
        apply_source_connectivity=False,
        apply_downwind_sector=False,
        apply_distance_bound=False,
    )
    assert res.dropped["invalid"] == 100
    assert res.n_kept_px == 300


# --- degradation, not crashes -----------------------------------------------


def test_empty_input_degrades_to_an_empty_mask():
    res = delineate_plume(
        np.zeros((60, 60), bool),
        enhancement_ppb=np.zeros((60, 60)),
        source_rc=(30.0, 30.0),
        wind_vector_ms=(3.0, 0.0),
    )
    assert res.n_input_px == 0
    assert res.n_kept_px == 0
    assert not res.mask.any()
    assert np.isnan(res.mean_ppb_before) and np.isnan(res.mean_ppb_after)
    assert np.isnan(res.implied_flux_factor)
    assert rejection_report(res)["balanced"] is True


def test_all_true_input_reduces_to_the_admissible_cone():
    mask = np.ones((80, 80), bool)
    res = delineate_plume(
        mask, source_rc=(40.0, 10.0), wind_vector_ms=(3.0, 0.0), pixel_size_m=PIXEL_M
    )
    assert 0 < res.n_kept_px < res.n_input_px
    assert res.dropped["downwind_sector"] > 0
    # Nothing upwind of the tolerance survives.
    assert int(res.mask[:, :4].sum()) == 0
    # The source neighbourhood does.
    assert res.mask[40, 12]


def test_source_outside_the_mask_keeps_the_nearest_component_and_says_so():
    mask = np.zeros((80, 80), bool)
    mask[10:16, 10:16] = True
    res = delineate_plume(
        mask,
        source_rc=(60.0, 60.0),
        wind_vector_ms=(-3.0, 3.0),  # blowing north-west, i.e. toward the blob
        pixel_size_m=PIXEL_M,
    )
    assert res.n_kept_px > 0
    assert any("not inside the candidate mask" in note for note in res.notes)


def test_missing_context_skips_rules_instead_of_guessing():
    """No source means no source-relative rule — and no 'largest blob' fallback.

    Falling back to "keep the biggest component" is exactly how the Ehrenberg
    artifact (2,121 px) would have been kept and the plume (480 px) discarded.
    """
    _, artifact, raw, field, _ = _ehrenberg_scene()

    res = delineate_plume(raw, enhancement_ppb=field, pixel_size_m=PIXEL_M)

    assert set(res.rules_skipped) >= {
        "source_connectivity",
        "downwind_sector",
        "distance_bound",
    }
    assert any("no source position" in note for note in res.notes)
    # The amplitude rule still runs and still removes the artifact.
    assert int((res.mask & artifact).sum()) == 0

    # No wind: the sector is skipped, but the rest still applies.
    no_wind = delineate_plume(
        raw, enhancement_ppb=field, source_rc=SOURCE_RC, pixel_size_m=PIXEL_M
    )
    assert "downwind_sector" in no_wind.rules_skipped
    assert "distance_bound" in no_wind.rules_skipped
    assert "source_connectivity" in no_wind.rules_applied

    # No enhancement field: the amplitude rule is skipped, not silently passed.
    no_field = delineate_plume(
        raw, source_rc=SOURCE_RC, wind_vector_ms=WIND_MS, pixel_size_m=PIXEL_M
    )
    assert "amplitude" in no_field.rules_skipped
    assert np.isnan(no_field.mean_ppb_before)


def test_everything_removed_is_reported_as_a_valid_outcome():
    """A metered-zero scene should be allowed to come back empty."""
    mask = np.zeros((80, 80), bool)
    mask[10:20, 10:20] = True
    field = np.zeros((80, 80))
    field[mask] = 9_000.0  # the Ehrenberg zero-control amplitude

    res = delineate_plume(
        mask, enhancement_ppb=field, source_rc=(15.0, 15.0), wind_vector_ms=(3.0, 0.0)
    )
    assert res.n_kept_px == 0
    assert res.implied_flux_factor == 0.0
    assert res.area_shrink_factor == float("inf")
    assert any("valid outcome" in note for note in res.notes)


def test_result_serialises_for_provenance():
    _, _, raw, field, _ = _ehrenberg_scene()
    res = delineate_plume(
        raw,
        enhancement_ppb=field,
        source_rc=SOURCE_RC,
        wind_vector_ms=WIND_MS,
        pixel_size_m=PIXEL_M,
    )
    d = res.as_dict()
    assert set(d) >= {
        "n_input_px",
        "n_kept_px",
        "dropped_px",
        "implied_flux_factor",
        "max_extent_m",
        "downwind_unit_rc",
    }
    assert isinstance(d["dropped_px"], dict)
    assert d["downwind_unit_rc"] == [0.0, 1.0]
    import json

    json.dumps(d)  # must be JSON-safe
    json.dumps(rejection_report(res))


# --- validation -------------------------------------------------------------


def test_shape_and_parameter_validation():
    mask = np.zeros((10, 10), bool)
    with pytest.raises(ValueError, match="enhancement and mask shapes differ"):
        delineate_plume(mask, enhancement_ppb=np.zeros((11, 11)))
    with pytest.raises(ValueError, match="valid and mask shapes differ"):
        delineate_plume(mask, valid=np.ones((11, 11), bool))
    with pytest.raises(ValueError, match="must be 2-D"):
        delineate_plume(np.zeros((3, 4, 5), bool))
    with pytest.raises(ValueError, match="pixel_size_m must be positive"):
        delineate_plume(mask, pixel_size_m=0.0)
    with pytest.raises(ValueError, match="half-angle"):
        delineate_plume(
            mask, source_rc=(5.0, 5.0), wind_vector_ms=(1.0, 0.0),
            sector_half_angle_deg=95.0,
        )
    with pytest.raises(ValueError, match="connectivity must be"):
        delineate_plume(mask, connectivity=3)
    with pytest.raises(ValueError, match="max_extent_m must be positive"):
        delineate_plume(mask, source_rc=(5.0, 5.0), max_extent_m=-1.0)
