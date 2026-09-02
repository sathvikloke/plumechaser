"""Observability-atlas surface: geometry, measured anchors, and the two floors.

The atlas is the project's headline deliverable, so these tests guard meaning as
much as arithmetic:

* the community detection floor and OUR artifact floor are different quantities
  and must never collapse into one another;
* the artifact floor cannot be produced without the label saying it is a
  limitation and not an emission (project rule 4);
* anything derived from ``atlas.measured_sigma_log_ratio`` carries n=1 and a
  provisional flag, because three single scenes are not a climatology.

Bundle-backed assertions run against synthetic bundles in ``tmp_path`` so they
hold on a fresh clone: ``bundles/`` is gitignored.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from plumechaser.atlas.limits import (
    PROVISIONAL_N_NOTE,
    S2_MIN_BLOB_PIXELS,
    MeasuredNoiseScene,
    community_detection_floor,
    limit_surface,
    load_measured_scenes,
    min_detectable_rate,
    our_artifact_floor,
    parse_measured_sigma_key,
    qmin_curve,
    sigma_col_ppb_from_log_ratio,
    solar_zenith_deg,
)
from plumechaser.config import Basin, load_config
from plumechaser.retrieve.calibration import load_calibration, simplified_c1

REPO = Path(__file__).resolve().parents[1]
CALIBRATION = REPO / "config" / "rtm_calibration.json"
TRUTH = REPO / "config" / "controlled_release_truth.json"
BUNDLES = REPO / "bundles"


@pytest.fixture(scope="module")
def cal():
    return load_calibration(CALIBRATION)


# --------------------------------------------------------------------------
# scene geometry
# --------------------------------------------------------------------------


class TestSolarGeometry:
    def test_greenwich_solstice_noon(self):
        """At solar noon SZA collapses to |latitude - declination|.

        Greenwich (51.4778 N) on the 2020 June solstice: declination +23.44 deg,
        solar noon within two minutes of 12:00 UTC, so SZA ~ 28.04 deg.
        """
        sza = solar_zenith_deg(
            datetime(2020, 6, 21, 12, 0, tzinfo=timezone.utc), 51.4778, 0.0
        )
        assert sza == pytest.approx(28.04, abs=0.25)

    def test_equator_equinox_noon(self):
        sza = solar_zenith_deg(
            datetime(2021, 3, 20, 12, 0, tzinfo=timezone.utc), 0.0, 0.0
        )
        assert sza < 2.0

    def test_naive_datetime_is_read_as_utc(self):
        naive = datetime(2020, 6, 21, 12, 0)
        aware = datetime(2020, 6, 21, 12, 0, tzinfo=timezone.utc)
        assert solar_zenith_deg(naive, 51.4778, 0.0) == pytest.approx(
            solar_zenith_deg(aware, 51.4778, 0.0)
        )

    def test_longitude_shifts_the_hour_angle(self):
        """Same UTC instant, 60 deg further east: four hours past solar noon."""
        at_meridian = solar_zenith_deg(
            datetime(2021, 6, 21, 12, 0, tzinfo=timezone.utc), 30.0, 0.0
        )
        far_east = solar_zenith_deg(
            datetime(2021, 6, 21, 12, 0, tzinfo=timezone.utc), 30.0, 60.0
        )
        assert far_east > at_meridian + 30.0

    def test_night_side_is_beyond_ninety_degrees(self):
        assert (
            solar_zenith_deg(datetime(2021, 6, 21, 0, 0, tzinfo=timezone.utc), 45.0, 0.0)
            > 90.0
        )


# --------------------------------------------------------------------------
# config key parsing -- this is what keeps make_atlas.py re-runnable
# --------------------------------------------------------------------------


class TestSceneKeyParsing:
    @pytest.mark.parametrize(
        ("key", "basin", "day", "screened", "baseline"),
        [
            ("korpezhe_2026-08-03_cloudscreened_50d", "korpezhe",
             date(2026, 8, 3), True, 50),
            ("korpezhe_2026-08-03_unscreened_50d", "korpezhe",
             date(2026, 8, 3), False, 50),
            ("permian_2026-04-24_cloudscreened_30d", "permian",
             date(2026, 4, 24), True, 30),
            # a basin whose own name contains an underscore
            ("region_c_2027-01-05_cloudscreened_10d", "region_c",
             date(2027, 1, 5), True, 10),
            # a future key with neither optional token
            ("permian_2027-06-01", "permian", date(2027, 6, 1), None, None),
        ],
    )
    def test_parses(self, key, basin, day, screened, baseline):
        parsed = parse_measured_sigma_key(key)
        assert (parsed.basin, parsed.day) == (basin, day)
        assert parsed.cloud_screened is screened
        assert parsed.baseline_days == baseline

    def test_unknown_trailing_tokens_are_kept_not_rejected(self):
        parsed = parse_measured_sigma_key("permian_2027-06-01_cloudscreened_5d_l2a")
        assert parsed.extra == ("l2a",)
        assert parsed.baseline_days == 5

    @pytest.mark.parametrize("bad", ["korpezhe", "2026-08-03", "korpezhe_20260803"])
    def test_rejects_unparseable(self, bad):
        with pytest.raises(ValueError):
            parse_measured_sigma_key(bad)


# --------------------------------------------------------------------------
# measured noise -> ppb at real geometry
# --------------------------------------------------------------------------


class TestSigmaFromLogRatio:
    def test_monotone_in_noise(self, cal):
        low = sigma_col_ppb_from_log_ratio(0.005, cal, "S2B", 30.0, 5.0)
        high = sigma_col_ppb_from_log_ratio(0.010, cal, "S2B", 30.0, 5.0)
        assert 0 < low < high

    def test_rtm_scale_exceeds_the_simplified_chain(self, cal):
        """The 2026-08-25 audit: our simplified alpha understates ppb 2.5-6.3x."""
        cfg = load_config(REPO / "config" / "default.yaml")
        simple_c1 = simplified_c1(
            cfg.mbmp.alpha_b12_per_ppb, cfg.mbmp.alpha_b11_per_ppb
        )
        sigma_ratio = 0.005
        rtm = sigma_col_ppb_from_log_ratio(sigma_ratio, cal, "S2B", 30.0, 5.0)
        ratio = rtm / (simple_c1 * sigma_ratio)
        assert 2.4 < ratio < 6.4

    @pytest.mark.parametrize("bad", [0.0, -0.01])
    def test_rejects_non_positive(self, cal, bad):
        with pytest.raises(ValueError):
            sigma_col_ppb_from_log_ratio(bad, cal, "S2B", 30.0, 5.0)


def _scene(**over) -> MeasuredNoiseScene:
    base = {
        "key": "testbasin_2026-01-01_cloudscreened_10d",
        "basin": "testbasin",
        "surface_class": "homogeneous_arid",
        "day": date(2026, 1, 1),
        "sigma_log_ratio": 0.007,
        "satellite": "S2B",
        "sza_deg": 30.0,
        "vza_deg": 5.0,
        "lat": 39.0,
        "lon": 59.0,
        "surface_pressure_hpa": 1013.0,
        "u10_ms": 3.0,
        "geometry_source": "scene_timestamp",
    }
    base.update(over)
    return MeasuredNoiseScene(**base)


class TestQminCurve:
    def test_rises_with_wind_and_with_noise(self, cal):
        quiet = _scene()
        noisy = _scene(sigma_log_ratio=0.014)
        winds = [2.0, 5.0, 9.0]
        q_quiet = qmin_curve(quiet, winds, cal, k_sigma=3.0)
        q_noisy = qmin_curve(noisy, winds, cal, k_sigma=3.0)
        assert np.all(np.diff(q_quiet) > 0)
        assert np.all(q_noisy > q_quiet)

    def test_larger_blob_requires_a_larger_source(self, cal):
        scene = _scene()
        small = qmin_curve(scene, [4.0], cal, k_sigma=3.0, min_pixels=9)[0]
        large = qmin_curve(scene, [4.0], cal, k_sigma=3.0, min_pixels=100)[0]
        assert large > small

    def test_default_blob_size_is_the_documented_one(self, cal):
        scene = _scene()
        assert qmin_curve(scene, [4.0], cal, k_sigma=3.0)[0] == pytest.approx(
            qmin_curve(scene, [4.0], cal, k_sigma=3.0,
                       min_pixels=S2_MIN_BLOB_PIXELS)[0]
        )

    def test_scene_carries_n_and_is_flagged_provisional(self):
        scene = _scene()
        assert scene.n_scenes == 1
        assert scene.provisional is True
        assert "n=1" in scene.label()


def test_provisional_note_says_what_it_must():
    lowered = PROVISIONAL_N_NOTE.lower()
    assert "provisional" in lowered
    assert "n=1" in lowered
    assert "climatology" in lowered


# --------------------------------------------------------------------------
# joining measured sigma to scene geometry
# --------------------------------------------------------------------------


def _stub_cfg(measured: dict, basins: dict[str, Basin]):
    return SimpleNamespace(
        raw={"atlas": {"measured_sigma_log_ratio": measured}}, basins=basins
    )


TEST_BASIN = Basin(
    name="testbasin",
    role="champion",
    bbox=(58.0, 38.5, 60.0, 40.0),
    surface_class="homogeneous_arid",
    elevation_hpa=1013.0,
)


def _write_bundle(root: Path, name: str, meta: dict) -> None:
    (root / name).mkdir(parents=True, exist_ok=True)
    (root / name / "provenance.json").write_text(json.dumps(meta))


class TestLoadMeasuredScenes:
    def test_falls_back_to_nominal_overpass_when_no_bundle_matches(self, tmp_path):
        cfg = _stub_cfg(
            {"testbasin_2027-07-01_cloudscreened_20d": 0.008},
            {"testbasin": TEST_BASIN},
        )
        (scene,) = load_measured_scenes(cfg, bundles_dir=tmp_path)
        assert scene.geometry_source == "nominal_overpass"
        assert scene.satellite == "S2B"
        assert scene.lon == pytest.approx(59.0)  # basin centroid
        assert scene.lat == pytest.approx(39.25)
        assert scene.event_id is None
        assert 0.0 < scene.sza_deg < 90.0
        assert scene.n_scenes == 1 and scene.provisional

    def test_uses_the_matching_bundle_geometry_when_present(self, tmp_path):
        _write_bundle(
            tmp_path,
            "EVT-TEST",
            {
                "event_id": "EVT-TEST",
                "basin": "testbasin",
                "det_date": "2027-07-01",
                "lat": 39.68,
                "lon": 58.52,
                "u10_ms": 4.25,
                "cloud_screening": "CloudSEN12",
                "pixels_target_safe":
                    "S2A_MSIL1C_20270701T065619_N0512_R063_T40TFK_x.SAFE",
                "pixels_background_safe":
                    "S2A_MSIL1C_20270611T065619_N0512_R063_T40TFK_x.SAFE",
                "gates": {"sigma_col_ppb": 912.5},
            },
        )
        cfg = _stub_cfg(
            {"testbasin_2027-07-01_cloudscreened_20d": 0.008},
            {"testbasin": TEST_BASIN},
        )
        (scene,) = load_measured_scenes(cfg, bundles_dir=tmp_path)
        assert scene.geometry_source == "scene_timestamp"
        assert scene.satellite == "S2A"
        assert scene.u10_ms == pytest.approx(4.25)
        assert scene.event_id == "EVT-TEST"
        assert scene.recorded_sigma_col_ppb == pytest.approx(912.5)

    def test_screening_and_baseline_tokens_disambiguate_sibling_runs(self, tmp_path):
        common = {
            "basin": "testbasin",
            "det_date": "2027-07-01",
            "lat": 39.68,
            "lon": 58.52,
            "pixels_background_safe":
                "S2B_MSIL1C_20270611T065619_N0512_R063_T40TFK_x.SAFE",
            "pixels_target_safe":
                "S2B_MSIL1C_20270701T065619_N0512_R063_T40TFK_x.SAFE",
        }
        _write_bundle(tmp_path, "EVT-SCREENED",
                      {**common, "event_id": "EVT-SCREENED",
                       "cloud_screening": "CloudSEN12"})
        _write_bundle(tmp_path, "EVT-RAW", {**common, "event_id": "EVT-RAW"})
        cfg = _stub_cfg(
            {
                "testbasin_2027-07-01_cloudscreened_20d": 0.008,
                "testbasin_2027-07-01_unscreened_20d": 0.05,
            },
            {"testbasin": TEST_BASIN},
        )
        by_key = {s.key: s for s in load_measured_scenes(cfg, bundles_dir=tmp_path)}
        assert by_key["testbasin_2027-07-01_cloudscreened_20d"].event_id == \
            "EVT-SCREENED"
        assert by_key["testbasin_2027-07-01_unscreened_20d"].event_id == "EVT-RAW"

    def test_a_withdrawn_or_diagnostic_source_run_is_surfaced(self, tmp_path):
        _write_bundle(
            tmp_path,
            "EVT-OLD",
            {
                "event_id": "EVT-OLD",
                "basin": "testbasin",
                "det_date": "2027-07-01",
                "lat": 39.0,
                "lon": 59.0,
                "result_status": "diagnostic",
                "pixels_target_safe":
                    "S2B_MSIL1C_20270701T065619_N0512_R063_T40TFK_x.SAFE",
            },
        )
        cfg = _stub_cfg(
            {"testbasin_2027-07-01": 0.008}, {"testbasin": TEST_BASIN}
        )
        (scene,) = load_measured_scenes(cfg, bundles_dir=tmp_path)
        assert scene.bundle_result_status == "diagnostic"
        assert scene.quotable is False
        assert "DIAGNOSTIC" in scene.label()

    def test_unreadable_provenance_is_skipped_not_fatal(self, tmp_path):
        (tmp_path / "EVT-HALF").mkdir()
        (tmp_path / "EVT-HALF" / "provenance.json").write_text('{"event_id": ')
        cfg = _stub_cfg(
            {"testbasin_2027-07-01": 0.008}, {"testbasin": TEST_BASIN}
        )
        (scene,) = load_measured_scenes(cfg, bundles_dir=tmp_path)
        assert scene.geometry_source == "nominal_overpass"

    def test_unknown_basin_without_a_bundle_is_an_error(self, tmp_path):
        cfg = _stub_cfg({"atlantis_2027-07-01": 0.008}, {})
        with pytest.raises(ValueError, match="atlantis"):
            load_measured_scenes(cfg, bundles_dir=tmp_path)


# --------------------------------------------------------------------------
# the two empirical floors -- different quantities, never merged
# --------------------------------------------------------------------------


class TestCommunityDetectionFloor:
    def test_matches_the_smallest_metered_release_per_campaign(self):
        floor = community_detection_floor(TRUTH)
        assert floor.low_t_h == pytest.approx(1.047, abs=1e-3)
        assert floor.high_t_h == pytest.approx(1.387, abs=1e-3)
        assert floor.n == 2

    def test_is_attributed_to_the_published_teams_not_to_us(self):
        floor = community_detection_floor(TRUTH)
        assert floor.is_ours is False
        assert "published by others" in floor.attribution
        assert "NOT BY US" in floor.label
        assert "DEMONSTRATED" in floor.label

    def test_carries_the_vendor_figure_and_design_caveats(self):
        caveats = " ".join(community_detection_floor(TRUTH).caveats).lower()
        assert "vendor" in caveats
        assert "not a measurement from these tests" in caveats
        assert "known-location" in caveats


ZERO_BUNDLES = {
    "CR-ehrenberg_2021-20211101": ("ehrenberg_2021", "2021-11-01", 156608.9),
    "CR-casa_grande_2022-20221125": ("casa_grande_2022", "2022-11-25", 145875.8),
}
NO_RELEASE_BUNDLE = ("CR-casa_grande_2022-20221011", "casa_grande_2022",
                     "2022-10-11", 18058.7)


@pytest.fixture
def zero_control_bundles(tmp_path) -> Path:
    for name, (basin, day, q_kg_h) in ZERO_BUNDLES.items():
        _write_bundle(
            tmp_path,
            name,
            {
                "event_id": name,
                "basin": basin,
                "det_date": day,
                "q_output": None,
                "q_output_withheld_artifact_dominated": {"Q": q_kg_h},
            },
        )
    name, basin, day, q_kg_h = NO_RELEASE_BUNDLE
    _write_bundle(
        tmp_path,
        name,
        {
            "event_id": name,
            "basin": basin,
            "det_date": day,
            "q_output": None,
            "q_output_withheld_artifact_dominated": {"Q": q_kg_h},
        },
    )
    return tmp_path


class TestOurArtifactFloor:
    def test_built_from_metered_zero_scenes_only(self, zero_control_bundles):
        floor = our_artifact_floor(zero_control_bundles, TRUTH)
        assert floor.n == 2
        assert floor.low_t_h == pytest.approx(145.876, abs=1e-2)
        assert floor.high_t_h == pytest.approx(156.609, abs=1e-2)
        # the 2022 no-release day is OUR inference of a zero, not the papers';
        # it must not silently drag the floor down an order of magnitude
        assert NO_RELEASE_BUNDLE[0] not in {s for s, _ in floor.members}

    def test_no_release_days_are_opt_in_and_labelled(self, zero_control_bundles):
        floor = our_artifact_floor(
            zero_control_bundles, TRUTH, kinds=("zero_control", "no_release")
        )
        assert floor.n == 3
        assert floor.low_t_h == pytest.approx(18.059, abs=1e-2)

    def test_label_cannot_be_stripped_of_rule_four_wording(self, zero_control_bundles):
        """Rule 4: this number may appear only as OUR limitation."""
        label = our_artifact_floor(zero_control_bundles, TRUTH).label
        assert "ARTIFACT FLOOR" in label
        assert "METERED-ZERO" in label
        assert "NOT A MEASUREMENT OF AN EMISSION" in label
        assert "withheld by the honesty gates" in label
        assert "n=2" in label

    def test_is_attributed_to_us(self, zero_control_bundles):
        assert our_artifact_floor(zero_control_bundles, TRUTH).is_ours is True

    def test_raises_rather_than_inventing_a_floor(self, tmp_path):
        with pytest.raises(ValueError, match="metered-zero"):
            our_artifact_floor(tmp_path, TRUTH)

    def test_a_zero_scene_we_got_right_is_not_a_floor_point(self, tmp_path):
        _write_bundle(
            tmp_path,
            "CR-ehrenberg_2021-20211101",
            {
                "event_id": "CR-ehrenberg_2021-20211101",
                "basin": "ehrenberg_2021",
                "det_date": "2021-11-01",
                "q_output": None,
                "q_output_withheld_artifact_dominated": None,
            },
        )
        with pytest.raises(ValueError):
            our_artifact_floor(tmp_path, TRUTH)


class TestTheFloorsAreDifferentQuantities:
    def test_our_floor_is_orders_of_magnitude_above_the_community_floor(
        self, zero_control_bundles
    ):
        """The atlas's central assertion, guarded as an invariant.

        If these two ever converge, either the chain improved enormously or the
        two quantities have been conflated. Both deserve a failing test.
        """
        community = community_detection_floor(TRUTH)
        ours = our_artifact_floor(zero_control_bundles, TRUTH)
        assert ours.low_t_h / community.high_t_h > 50.0
        assert ours.attribution != community.attribution
        assert ours.is_ours and not community.is_ours

    def test_every_rate_this_study_targets_is_below_our_floor(
        self, zero_control_bundles
    ):
        truth = json.loads(Path(TRUTH).read_text())
        metered = [
            float(o["kg_h"]) / 1000.0
            for c in truth["campaigns"]
            for o in c["overpasses"]
            if o.get("kind") == "release" and float(o.get("kg_h", 0.0)) > 0.0
        ]
        ours = our_artifact_floor(zero_control_bundles, TRUTH)
        assert max(metered) < ours.low_t_h


# --------------------------------------------------------------------------
# backward compatibility -- cli.py and scripts/make_figures.py call these
# --------------------------------------------------------------------------


class TestLegacySurface:
    def test_min_detectable_rate_monotonicity(self):
        base = dict(
            sigma_col_ppb=12.0, k_sigma=3.0, min_pixels=3, pixel_area_m2=1113.0**2,
            u10_ms=3.0, typical_plume_length_m=800.0,
        )
        q0 = min_detectable_rate(**base)
        assert q0 > 0
        assert min_detectable_rate(**{**base, "sigma_col_ppb": 24.0}) > q0
        assert min_detectable_rate(**{**base, "min_pixels": 6}) > q0
        assert min_detectable_rate(**{**base, "u10_ms": 6.0}) > q0

    def test_plume_length_never_shorter_than_the_blob(self):
        """L = max(class length, sqrt(N*A)); a 3 px TROPOMI blob is 1.9 km wide."""
        tiny_class = min_detectable_rate(
            sigma_col_ppb=12.0, k_sigma=3.0, min_pixels=3,
            pixel_area_m2=1113.0**2, u10_ms=3.0, typical_plume_length_m=1.0,
        )
        stated = min_detectable_rate(
            sigma_col_ppb=12.0, k_sigma=3.0, min_pixels=3,
            pixel_area_m2=1113.0**2, u10_ms=3.0,
            typical_plume_length_m=math.sqrt(3 * 1113.0**2),
        )
        assert tiny_class == pytest.approx(stated)

    def test_limit_surface_still_skips_missing_seasons(self):
        surface = limit_surface(
            {"korpezhe": {"surface_class": "homogeneous_arid"},
             "permian": {"surface_class": "heterogeneous"}},
            seasons=["JJA", "DJF"],
            u10_by_basin_season={("korpezhe", "JJA"): 4.0},
            sigma_by_class_season={
                ("homogeneous_arid", "JJA"): 12.0,
                ("heterogeneous", "JJA"): 25.0,
            },
            k_sigma=3.0, min_pixels=3, pixel_size_m=1113,
            lengths_by_class={"homogeneous_arid": 800.0, "heterogeneous": 1200.0},
        )
        assert set(surface) == {"korpezhe", "permian"}
        assert set(surface["korpezhe"]) == {"JJA"}


# --------------------------------------------------------------------------
# against the real repo state (skipped on a fresh clone: bundles/ is gitignored)
# --------------------------------------------------------------------------

_real_bundles = pytest.mark.skipif(
    not BUNDLES.is_dir(), reason="bundles/ is gitignored; run a campaign first"
)


class TestAgainstRealMeasuredAnchors:
    @_real_bundles
    def test_every_configured_sigma_resolves_to_a_real_scene(self, cal):
        cfg = load_config(REPO / "config" / "default.yaml")
        scenes = load_measured_scenes(cfg, bundles_dir=BUNDLES)
        assert len(scenes) == len(cfg.raw["atlas"]["measured_sigma_log_ratio"])
        for scene in scenes:
            assert scene.geometry_source == "scene_timestamp"
            assert scene.satellite.startswith("S2")
            # inside the sampled RTM envelope, so no coefficient extrapolation
            assert cal.sza_grid[0] <= scene.sza_deg <= cal.sza_grid[-1]
            assert scene.n_scenes == 1 and scene.provisional
            assert scene.sigma_col_ppb(cal) > 0

    @_real_bundles
    def test_measured_noise_is_far_worse_than_the_assumed_literature_value(self, cal):
        """The atlas's own finding: 12-25 ppb was optimistic by 1-2 decades."""
        cfg = load_config(REPO / "config" / "default.yaml")
        for scene in load_measured_scenes(cfg, bundles_dir=BUNDLES):
            assert scene.sigma_col_ppb(cal) > 10 * 25.0

    @_real_bundles
    def test_noise_limited_qmin_sits_between_the_two_empirical_floors(self, cal):
        """Our chain is nowhere near its own noise floor -- the headline gap."""
        cfg = load_config(REPO / "config" / "default.yaml")
        community = community_detection_floor(TRUTH)
        ours = our_artifact_floor(BUNDLES, TRUTH)
        lengths = cfg.raw["atlas"]["typical_plume_length_m"]
        for scene in load_measured_scenes(cfg, bundles_dir=BUNDLES):
            q_t_h = qmin_curve(
                scene,
                [scene.u10_ms],
                cal,
                k_sigma=float(cfg.raw["atlas"]["k_roc"]),
                pixel_size_m=cfg.sentinel2.pixel_size_m,
                typical_plume_length_m=float(lengths[scene.surface_class]),
            )[0] / 1000.0
            assert q_t_h < ours.low_t_h / 10.0
            assert q_t_h < 10.0 * community.high_t_h


# --------------------------------------------------------------------------
# the builder script
# --------------------------------------------------------------------------


def _load_make_atlas():
    import importlib.util

    path = REPO / "scripts" / "make_atlas.py"
    spec = importlib.util.spec_from_file_location("make_atlas_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestMakeAtlasScript:
    def test_help_exits_clean(self, capsys):
        module = _load_make_atlas()
        with pytest.raises(SystemExit) as exc:
            module.main(["--help"])
        assert exc.value.code == 0
        assert "observability atlas" in capsys.readouterr().out

    @_real_bundles
    def test_summary_json_states_provenance_and_provisional_n(self, tmp_path):
        module = _load_make_atlas()
        assert module.main(["--out-dir", str(tmp_path), "--json-only"]) == 0
        payload = json.loads((tmp_path / "atlas_summary.json").read_text())

        assert payload["provisional_note"] == PROVISIONAL_N_NOTE
        assert payload["settings"]["min_blob_pixels_is_a_function_default"] is True

        for record in payload["measured_scenes"]:
            assert record["n_scenes"] == 1
            assert record["provisional"] is True
            assert record["sigma_col_ppb_measured_at_geometry"] > 0
            assert record["geometry_source"] in (
                "scene_timestamp", "nominal_overpass"
            )

        floors = payload["floors"]
        assert floors["our_quantification_artifact"]["attribution"].startswith("ours")
        assert "NOT A MEASUREMENT OF AN EMISSION" in \
            floors["our_quantification_artifact"]["label"]
        assert floors["community_detection"]["attribution"].startswith(
            "published by others"
        )
        assert "NOT BY US" in floors["community_detection"]["label"]
        assert (
            floors["our_quantification_artifact"]["low_t_h"]
            > 50 * floors["community_detection"]["high_t_h"]
        )

    @_real_bundles
    def test_figures_are_written(self, tmp_path):
        module = _load_make_atlas()
        assert module.main(["--out-dir", str(tmp_path), "--dpi", "60"]) == 0
        for name in (
            "atlas_summary.json",
            "atlas_observability.png",
            "atlas_noise_anchors.png",
        ):
            assert (tmp_path / name).stat().st_size > 0
        # must never clobber the figure scripts/make_figures.py owns
        assert not (tmp_path / "atlas_limits.png").exists()
