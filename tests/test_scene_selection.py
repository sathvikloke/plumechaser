"""Scene-selection and pixel-validity rules for the simplified S2 chain.

These are the rules the 2026-08-25 flux audit found broken in the sibling
production chain (docs/S2_REAL_DATA_FINDINGS.md, "Three compounding
preprocessing defects"): a background granule that covered only 57% of the
retrieval window, a reference from the wrong platform on the wrong relative
orbit, and nodata pixels entering the retrieval as -1000 DN. All of them are
decidable from names and arrays, so all of them are tested offline — nothing
here touches the network.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import real_s2_demo as demo  # noqa: E402

from plumechaser.retrieve.mbmp import log_band_ratio  # noqa: E402

TARGET = "S2B_MSIL1C_20260803T072619_N0511_R049_T40TFK_20260803T081205.SAFE"


def safe(day: date, platform: str = "S2B", orbit: str = "R049",
         stamp: str = "072619") -> str:
    """A SAFE directory name shaped exactly like the GCS mirror's."""
    return (f"{platform}_MSIL1C_{day:%Y%m%d}T{stamp}_N0511_{orbit}"
            f"_T40TFK_{day:%Y%m%d}T081205.SAFE")


# ------------------------------------------------- relative-orbit parsing

def test_relative_orbit_is_parsed_from_a_real_safe_name():
    assert demo.rel_orbit(TARGET) == "R049"
    assert demo.rel_orbit(safe(date(2026, 7, 24), orbit="R006")) == "R006"


def test_relative_orbit_is_unknown_rather_than_wrong_when_absent():
    """A missing orbit token must never silently read as "same orbit"."""
    assert demo.rel_orbit("S2B_MSIL1C_20260803T072619_N0511_T40TFK.SAFE") == "R???"
    # Two unknowns are not evidence of a shared orbit; the caller compares
    # against the target's token, which is a real Rnnn on any real name.
    assert demo.rel_orbit(TARGET) != "R???"


def test_relative_orbit_ignores_other_four_character_tokens():
    name = ("S2B_MSIL1C_20260803T072619_RXYZ_N0511_R049_T40TFK"
            "_20260803T081205.SAFE")
    assert demo.rel_orbit(name) == "R049"


def test_platform_comes_from_the_leading_token():
    assert demo.safe_platform(TARGET) == "S2B"
    assert demo.safe_platform(safe(date(2026, 8, 3), platform="S2A")) == "S2A"
    # S2C exists on other mirrors; it must not be misread as S2B.
    assert demo.safe_platform(safe(date(2026, 8, 3), platform="S2C")) == "S2C"


# --------------------------------------------------- window coverage rule

def test_window_nodata_frac_counts_pixels_the_granule_does_not_contain():
    window = np.full((10, 10), 0.25)
    assert demo.window_nodata_frac(window) == 0.0
    window[:4, :] = np.nan
    assert demo.window_nodata_frac(window) == pytest.approx(0.40)


def test_pick_covering_skips_the_swath_edge_granule_that_comes_first():
    """The 43%-missing-background defect: safes[0] is not safe."""
    fracs = {"edge.SAFE": 0.43, "full.SAFE": 0.001}
    picked, frac, covered = demo.pick_covering(
        ["edge.SAFE", "full.SAFE"], fracs.__getitem__)
    assert (picked, covered) == ("full.SAFE", True)
    assert frac == 0.001


def test_pick_covering_stops_at_the_first_covering_granule():
    probed: list[str] = []

    def probe(s):
        probed.append(s)
        return 0.0

    assert demo.pick_covering(["a", "b", "c"], probe)[0] == "a"
    assert probed == ["a"]  # one cheap read, not three


def test_pick_covering_reports_the_best_effort_when_nothing_covers():
    fracs = {"a": 0.9, "b": 0.43}
    picked, frac, covered = demo.pick_covering(["a", "b"], fracs.__getitem__)
    assert (picked, frac, covered) == ("b", 0.43, False)


def test_pick_covering_on_a_day_with_no_granules():
    assert demo.pick_covering([], lambda s: 0.0) == (None, 1.0, False)


def test_pick_covering_threshold_is_a_bound_not_an_optimum():
    # 2% nodata is accepted as covering even though a cleaner one follows;
    # the point is to reject partial granules, not to rank clean ones.
    picked, _, covered = demo.pick_covering(
        ["ok", "cleaner"], {"ok": 0.02, "cleaner": 0.0}.__getitem__)
    assert (picked, covered) == ("ok", True)


# ------------------------------------------------ nodata / negative DN

def test_nodata_is_excluded_and_filled_with_zero_not_left_negative():
    window = np.full((4, 4), 0.25)   # DN 2500 as returned by vrt_window
    window[0, 0] = np.nan            # nodata
    dn, valid, nodata, negative = demo.dn_with_valid(window, -1000.0)

    assert not valid[0, 0] and nodata[0, 0]
    assert dn[0, 0] == 0.0           # NOT -1000
    assert dn[1, 1] == pytest.approx(1500.0)
    assert valid[1, 1] and not nodata[1, 1] and not negative[1, 1]


def test_pixels_driven_non_positive_by_the_offset_are_invalid_too():
    window = np.array([[0.05, 0.25], [0.10, 0.02]])  # DN 500 / 2500 / 1000 / 200
    dn, valid, nodata, negative = demo.dn_with_valid(window, -1000.0)

    assert valid.tolist() == [[False, True], [False, False]]
    assert negative.tolist() == [[True, False], [True, True]]  # 1000-1000 = 0
    assert not nodata.any()
    assert dn.tolist() == [[0.0, 1500.0], [0.0, 0.0]]


def test_two_negative_dns_would_otherwise_forge_a_positive_band_ratio():
    """The reason the fill matters: -400/-900 is a perfectly good ratio.

    Left negative, an all-nodata corner produces a finite log ratio that the
    MBMP retrieval cannot distinguish from real surface signal.
    """
    b11 = np.array([[0.06, 0.25]])
    b12 = np.array([[0.01, 0.20]])

    naive = log_band_ratio(b11 * 1e4 - 1000, b12 * 1e4 - 1000)
    assert np.isfinite(naive[0, 0])          # the bug: a believable number

    dn11, ok11, _, _ = demo.dn_with_valid(b11, -1000.0)
    dn12, ok12, _, _ = demo.dn_with_valid(b12, -1000.0)
    valid = ok11 & ok12
    masked = log_band_ratio(demo.nan_invalid(dn11, valid),
                            demo.nan_invalid(dn12, valid))
    assert np.isnan(masked[0, 0])            # excluded, as it must be
    assert np.isfinite(masked[0, 1])         # the real pixel survives


def test_zero_fill_alone_already_keeps_invalid_pixels_out_of_the_ratio():
    window = np.array([[np.nan, 0.25]])
    dn, _, _, _ = demo.dn_with_valid(window, -1000.0)
    u = log_band_ratio(dn, dn)
    assert np.isnan(u[0, 0])


def test_a_fully_valid_window_is_unchanged_apart_from_the_offset():
    rng = np.random.default_rng(20270307)
    window = rng.uniform(0.15, 0.35, size=(32, 32))
    dn, valid, nodata, negative = demo.dn_with_valid(window, -1000.0)

    assert valid.all()
    assert not nodata.any() and not negative.any()
    assert dn == pytest.approx(window * 1e4 - 1000.0)


# ------------------------------------------------- reference preference

def _selector(scenes: dict[date, list[str]]):
    """(safes_on, choose) over a fixed catalogue; every granule covers."""
    def safes_on(day):
        return scenes.get(day, [])

    def choose(safes):
        return (safes[0], 0.0)

    return safes_on, choose


def test_same_platform_is_required_not_merely_preferred():
    t_day = date(2026, 8, 3)
    scenes = {
        t_day - timedelta(days=5): [safe(t_day - timedelta(days=5), "S2A")],
        t_day - timedelta(days=10): [safe(t_day - timedelta(days=10), "S2B")],
    }
    picked, n_same_orbit = demo.select_references(
        TARGET, t_day, *_selector(scenes), want=4)

    assert [s for _, s, _ in picked] == [safe(t_day - timedelta(days=10), "S2B")]
    assert n_same_orbit == 1
    # the closer S2A pass is never used, at any priority
    assert all(demo.safe_platform(s) == "S2B" for _, s, _ in picked)


def test_same_orbit_beats_a_temporally_closer_cross_orbit_pass():
    t_day = date(2026, 8, 3)
    scenes = {
        t_day - timedelta(days=5): [safe(t_day - timedelta(days=5), orbit="R006")],
        t_day - timedelta(days=10): [safe(t_day - timedelta(days=10), orbit="R049")],
    }
    picked, n_same_orbit = demo.select_references(
        TARGET, t_day, *_selector(scenes), want=1)

    assert demo.rel_orbit(picked[0][1]) == "R049"
    assert (t_day - picked[0][0]).days == 10   # the exact 10-day multiple
    assert n_same_orbit == 1


def test_cross_orbit_is_used_only_to_make_up_the_shortfall():
    t_day = date(2026, 8, 3)
    scenes = {
        t_day - timedelta(days=5): [safe(t_day - timedelta(days=5), orbit="R006")],
        t_day - timedelta(days=10): [safe(t_day - timedelta(days=10), orbit="R049")],
        t_day - timedelta(days=15): [safe(t_day - timedelta(days=15), orbit="R006")],
        t_day - timedelta(days=20): [safe(t_day - timedelta(days=20), orbit="R049")],
    }
    picked, n_same_orbit = demo.select_references(
        TARGET, t_day, *_selector(scenes), want=3)

    assert n_same_orbit == 2                       # both R049 passes first
    assert len(picked) == 3
    assert [(t_day - d).days for d, _, _ in picked] == [5, 10, 20]  # newest first
    cross = [s for _, s, _ in picked if demo.rel_orbit(s) != "R049"]
    assert len(cross) == 1                         # only the shortfall


def test_no_cross_orbit_when_same_orbit_already_satisfies_the_request():
    t_day = date(2026, 8, 3)
    scenes = {
        t_day - timedelta(days=5): [safe(t_day - timedelta(days=5), orbit="R006")],
        t_day - timedelta(days=10): [safe(t_day - timedelta(days=10), orbit="R049")],
        t_day - timedelta(days=20): [safe(t_day - timedelta(days=20), orbit="R049")],
    }
    picked, n_same_orbit = demo.select_references(
        TARGET, t_day, *_selector(scenes), want=2)

    assert n_same_orbit == 2
    assert all(demo.rel_orbit(s) == "R049" for _, s, _ in picked)


def test_reference_lag_window_is_respected():
    t_day = date(2026, 8, 3)
    scenes = {
        t_day - timedelta(days=2): [safe(t_day - timedelta(days=2))],   # too close
        t_day - timedelta(days=22): [safe(t_day - timedelta(days=22))],  # too old
        t_day - timedelta(days=10): [safe(t_day - timedelta(days=10))],
    }
    picked, _ = demo.select_references(TARGET, t_day, *_selector(scenes), want=4)

    assert [(t_day - d).days for d, _, _ in picked] == [10]


def test_uncovered_days_are_dropped_from_the_reference_set():
    """A day whose only granule misses the window is not a usable date."""
    t_day = date(2026, 8, 3)
    partial = safe(t_day - timedelta(days=10))
    scenes = {
        t_day - timedelta(days=10): [partial],
        t_day - timedelta(days=20): [safe(t_day - timedelta(days=20))],
    }
    safes_on, _ = _selector(scenes)

    def choose(safes):
        picked, frac, covered = demo.pick_covering(
            safes, lambda s: 0.43 if s == partial else 0.0)
        return None if not covered else (picked, frac)

    picked, n_same_orbit = demo.select_references(
        TARGET, t_day, safes_on, choose, want=4)

    assert [s for _, s, _ in picked] == [safe(t_day - timedelta(days=20))]
    assert n_same_orbit == 1


def test_coverage_is_checked_within_the_orbit_class_not_across_it():
    """A partial same-orbit granule must not drag in a cross-orbit day early."""
    t_day = date(2026, 8, 3)
    day10 = t_day - timedelta(days=10)
    partial = safe(day10, orbit="R049")
    scenes = {
        day10: [partial, safe(day10, orbit="R049", stamp="072620")],
        t_day - timedelta(days=5): [safe(t_day - timedelta(days=5), orbit="R006")],
    }
    safes_on, _ = _selector(scenes)

    def choose(safes):
        picked, frac, covered = demo.pick_covering(
            safes, lambda s: 0.43 if s == partial else 0.0)
        return None if not covered else (picked, frac)

    picked, n_same_orbit = demo.select_references(
        TARGET, t_day, safes_on, choose, want=1)

    assert picked[0][1] == safe(day10, orbit="R049", stamp="072620")
    assert n_same_orbit == 1


def test_no_usable_comparison_dates_returns_empty_for_the_caller_to_reject():
    t_day = date(2026, 8, 3)
    scenes = {t_day - timedelta(days=10): [safe(t_day - timedelta(days=10), "S2A")]}
    picked, n_same_orbit = demo.select_references(
        TARGET, t_day, *_selector(scenes), want=4)

    assert picked == [] and n_same_orbit == 0
