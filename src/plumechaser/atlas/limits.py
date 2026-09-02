"""Detection-limit surfaces and empirical floors for the observability atlas.

The atlas is the project's primary deliverable. The 2026-08-25 controlled-release
audit closed the absolute-flux question with a negative result -- against METERED
releases this chain's spurious-flux floor is ~150 t/h, above every rate the study
targets -- so quantification is not a claimable capability. What remains claimable
is detection plus a measured map of what free sensors can and cannot see. That map
is this module.

What the atlas asserts, and on what evidence
--------------------------------------------
Three quantities live on the same axis and must never be conflated.

1. **Analytic noise-limited Q_min** (ours, derived). For each surface class and
   geometry, the smallest emission rate whose plume would clear ``k`` sigmas of
   *retrieval noise*. Computed from quantities we measure:

       sigma_log_ratio  robust 1-sigma of ln(B11/B12)_target - ln(B11/B12)_ref
                        over plume-free pixels -- calibration-independent, and
                        the only noise number that is comparable between chains
       sigma_col        that noise in ppb, via the RTM curve at the SCENE's own
                        solar/viewing geometry (retrieve.calibration)
       k                ROC operating point in sigmas (config atlas.k_roc)
       N_min            minimum coherent blob [pixels]
       IME_min          N_min * column_mass(k*sigma_col) * pixel_area
       Q_min            IME_min * Ueff * 3600 / L_typical(class)

   This is a *floor set by noise alone*. It is optimistic by construction: it
   assumes the only obstacle is radiometry.

2. **Community detection floor, ~1.0-1.4 t/h** (published by others, measured).
   The smallest metered releases the published teams actually detected in the
   Stanford/EDF single-blind tests. Not ours, and not a vendor spec sheet --
   see :func:`community_detection_floor` and the caveats it carries.

3. **Our quantification artifact floor, ~150 t/h** (ours, measured). The
   spurious flux this chain reports on scenes where the meter read ZERO. It is
   a limitation of our pipeline, never a measurement of an emission
   (project rule 4). :func:`our_artifact_floor` bakes that wording into the
   label so it cannot be plotted without it.

The gap between (1) and (3) is the atlas's central finding: our chain is two to
three orders of magnitude away from its own noise floor, so what limits it is
plume delineation and surface structure, not radiometric noise. The community
floor (2) sitting near (1) is what shows the analytic model is not the problem.

Honesty about n
---------------
``config/default.yaml -> atlas.measured_sigma_log_ratio`` holds THREE SINGLE
SCENES, not a climatology. Every object built from them carries ``n_scenes``
and a ``provisional`` flag, and :data:`PROVISIONAL_N_NOTE` is the text that must
appear on any figure derived from them. Do not present a single scene as a basin
or seasonal average.

A note on the units of sigma_col
--------------------------------
Q_min is linear in sigma_col, so the atlas is only as meaningful as that number.
The 2026-08-25 audit showed ppb is **not** comparable between retrieval chains:
our simplified absorption coefficients understate columns by 2.5-6.3x versus the
production RTM, so the same scene yields two very different sigma_col values and
therefore two very different atlas surfaces. Start from
:func:`sigma_col_ppb_from_log_ratio`; a bare ppb value still works, but only
compare such numbers within one chain.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from plumechaser.report.status import STATUS_QUOTABLE, bundle_status
from plumechaser.retrieve.ime import effective_wind_speed
from plumechaser.retrieve.mbmp import column_mass_kg_m2

if TYPE_CHECKING:  # pragma: no cover - typing only
    from plumechaser.retrieve.calibration import RtmCalibration

__all__ = [
    "PROVISIONAL_N_NOTE",
    "S2_MEAN_LOCAL_SOLAR_TIME_H",
    "S2_MIN_BLOB_PIXELS",
    "S2_NOMINAL_SATELLITE",
    "S2_NOMINAL_VZA_DEG",
    "EmpiricalFloor",
    "MeasuredNoiseScene",
    "SceneKey",
    "community_detection_floor",
    "limit_surface",
    "load_measured_scenes",
    "min_detectable_rate",
    "our_artifact_floor",
    "parse_measured_sigma_key",
    "qmin_curve",
    "sigma_col_ppb_from_log_ratio",
    "solar_zenith_deg",
]

# --------------------------------------------------------------------------
# Documented function defaults.
#
# These are deliberately NOT config keys. config/default.yaml is frozen and
# referenced by commit hash from the analysis plan; adding keys to it after the
# freeze would break that reference for a value that is a modelling choice, not
# a pre-registered threshold. Each is stated here with its justification so the
# choice is auditable, and each is overridable at the call site.
# --------------------------------------------------------------------------

#: Minimum coherent blob for a Sentinel-2 (20 m) detection, in pixels.
#: ``tropomi.min_blob_pixels`` (3) is calibrated for 1113 m screening pixels --
#: 3 px there is 3.7 km2, 3 px at 20 m is 1200 m2, which is single-pixel noise.
#: 25 px = 0.01 km2 = a coherent 100 m x 100 m structure, at the small end of
#: what the published S2 plume studies delineate. Sensitivity to this choice is
#: reported by scripts/make_atlas.py rather than hidden.
S2_MIN_BLOB_PIXELS = 25

#: Sentinel-2 mean viewing zenith, degrees. The RTM curve moves under 5% across
#: the whole VZA 0-10 envelope (2026-08-25 calibration), so mid-swath is a safe
#: stand-in when the scene metadata is not on disk.
S2_NOMINAL_VZA_DEG = 5.0

#: Sentinel-2 mean local solar time at descending node. Used ONLY when no bundle
#: records the scene's real acquisition timestamp; scenes resolved this way are
#: flagged ``geometry_source="nominal_overpass"``.
S2_MEAN_LOCAL_SOLAR_TIME_H = 10.5

#: Platform assumed when the SAFE name is unavailable. S2A and S2B differ by
#: ~25% in ppb-per-log-ratio, so this is a real approximation, and it is flagged.
S2_NOMINAL_SATELLITE = "S2B"

#: Must appear ON any figure built from ``atlas.measured_sigma_log_ratio``.
PROVISIONAL_N_NOTE = (
    "PROVISIONAL: each sigma is ONE scene (n=1), not a climatology. "
    "Not a basin or seasonal average."
)

_SAFE_RE = re.compile(r"^(S2[ABCD])_MSIL1C_(\d{8}T\d{6})_")
_ISO_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


# --------------------------------------------------------------------------
# Scene geometry
# --------------------------------------------------------------------------


def solar_zenith_deg(when: datetime, lat: float, lon: float) -> float:
    """Solar zenith angle [deg] at a UTC instant and location.

    Low-precision NOAA solar-position algorithm (Meeus, abridged): better than
    0.1 deg over the modern era, which is far finer than the 10 deg RTM
    calibration grid it feeds. Implemented here rather than pulled in as a
    dependency because it is fifteen lines and the atlas must run offline.

    ``when`` must be timezone-aware; a naive datetime is read as UTC.
    """
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    jd = when.timestamp() / 86400.0 + 2440587.5
    n = jd - 2451545.0
    mean_anomaly = math.radians((357.528 + 0.9856003 * n) % 360.0)
    mean_long = math.radians((280.460 + 0.9856474 * n) % 360.0)
    ecl_long = (
        mean_long
        + math.radians(1.915) * math.sin(mean_anomaly)
        + math.radians(0.020) * math.sin(2.0 * mean_anomaly)
    )
    obliquity = math.radians(23.439 - 0.0000004 * n)
    declination = math.asin(math.sin(obliquity) * math.sin(ecl_long))
    right_asc = math.atan2(
        math.cos(obliquity) * math.sin(ecl_long), math.cos(ecl_long)
    )
    gmst_h = (18.697374558 + 24.06570982441908 * n) % 24.0
    local_sidereal = (gmst_h * 15.0 + lon) % 360.0
    hour_angle = math.radians(
        ((local_sidereal - math.degrees(right_asc)) + 180.0) % 360.0 - 180.0
    )
    lat_r = math.radians(lat)
    cos_z = math.sin(lat_r) * math.sin(declination) + math.cos(lat_r) * math.cos(
        declination
    ) * math.cos(hour_angle)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_z))))


def _nominal_overpass_utc(day: date, lon: float) -> datetime:
    """UTC instant of the nominal Sentinel-2 descending-node overpass."""
    utc_hour = S2_MEAN_LOCAL_SOLAR_TIME_H - lon / 15.0
    midnight = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    return midnight + timedelta(hours=utc_hour)


# --------------------------------------------------------------------------
# Measured band-ratio noise anchors
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SceneKey:
    """Parsed ``atlas.measured_sigma_log_ratio`` key.

    Grammar: ``<basin>_<YYYY-MM-DD>[_cloudscreened|_unscreened][_<N>d]``.
    Unknown trailing tokens are kept in ``extra`` rather than raising, so a key
    added by a later campaign still resolves to at least a basin and a date.
    """

    raw: str
    basin: str
    day: date
    cloud_screened: bool | None = None
    baseline_days: int | None = None
    extra: tuple[str, ...] = ()


def parse_measured_sigma_key(key: str) -> SceneKey:
    """Parse one measured-sigma config key. Raises ValueError if unparseable."""
    match = _ISO_DATE_RE.search(key)
    if not match:
        raise ValueError(f"measured-sigma key {key!r} has no ISO date")
    basin = key[: match.start()].rstrip("_")
    if not basin:
        raise ValueError(f"measured-sigma key {key!r} has no basin prefix")
    day = date.fromisoformat(match.group(1))
    screened: bool | None = None
    baseline: int | None = None
    extra: list[str] = []
    for token in key[match.end() :].strip("_").split("_"):
        if not token:
            continue
        if token == "cloudscreened":
            screened = True
        elif token == "unscreened":
            screened = False
        elif re.fullmatch(r"\d+d", token):
            baseline = int(token[:-1])
        else:
            extra.append(token)
    return SceneKey(
        raw=key,
        basin=basin,
        day=day,
        cloud_screened=screened,
        baseline_days=baseline,
        extra=tuple(extra),
    )


@dataclass(frozen=True)
class MeasuredNoiseScene:
    """One measured band-ratio noise anchor, resolved to a real geometry.

    ``n_scenes`` is 1 by construction and is carried so no consumer can quietly
    average or relabel it. ``provisional`` is True whenever ``n_scenes`` < 2.
    """

    key: str
    basin: str
    surface_class: str
    day: date
    sigma_log_ratio: float
    satellite: str
    sza_deg: float
    vza_deg: float
    lat: float
    lon: float
    surface_pressure_hpa: float
    u10_ms: float
    geometry_source: str
    cloud_screened: bool | None = None
    baseline_days: int | None = None
    event_id: str | None = None
    bundle_result_status: str = "absent"
    recorded_sigma_col_ppb: float | None = None
    n_scenes: int = 1

    @property
    def provisional(self) -> bool:
        return self.n_scenes < 2

    @property
    def quotable(self) -> bool:
        """False for scenes whose source run is withdrawn or diagnostic."""
        return self.bundle_result_status in (STATUS_QUOTABLE, "absent")

    def label(self) -> str:
        screen = {True: "cloud-screened", False: "UNSCREENED", None: "screening n/k"}[
            self.cloud_screened
        ]
        base = f", {self.baseline_days} d baseline" if self.baseline_days else ""
        status = (
            ""
            if self.bundle_result_status in (STATUS_QUOTABLE, "absent")
            else f" [{self.bundle_result_status.upper()} RUN]"
        )
        return f"{self.basin} {self.day.isoformat()} ({screen}{base}, n=1){status}"

    def sigma_col_ppb(self, calibration: RtmCalibration) -> float:
        """Measured band-ratio noise expressed in ppb at THIS scene's geometry."""
        return sigma_col_ppb_from_log_ratio(
            self.sigma_log_ratio,
            calibration,
            self.satellite,
            self.sza_deg,
            self.vza_deg,
        )


def _parse_safe(safe: str) -> tuple[str, datetime] | None:
    match = _SAFE_RE.match(safe or "")
    if not match:
        return None
    stamp = datetime.strptime(match.group(2), "%Y%m%dT%H%M%S")
    return match.group(1), stamp.replace(tzinfo=timezone.utc)


def _index_bundles(bundles_dir: Path) -> list[dict]:
    """Read every ``provenance.json`` under ``bundles_dir``. Unreadable ones are
    skipped: a campaign may be writing one while the atlas is being built."""
    out: list[dict] = []
    if not bundles_dir.is_dir():
        return out
    for prov in sorted(bundles_dir.glob("*/provenance.json")):
        try:
            meta = json.loads(prov.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(meta, dict):
            meta["_bundle_name"] = prov.parent.name
            out.append(meta)
    return out


def _bundle_matches(meta: dict, key: SceneKey) -> bool:
    if str(meta.get("basin", "")) != key.basin:
        return False
    if str(meta.get("det_date", "")) != key.day.isoformat():
        return False
    if key.cloud_screened is not None:
        has_screen = bool(meta.get("cloud_screening"))
        if has_screen is not key.cloud_screened:
            return False
    if key.baseline_days is not None:
        target = _parse_safe(str(meta.get("pixels_target_safe", "")))
        background = _parse_safe(str(meta.get("pixels_background_safe", "")))
        if not target or not background:
            return False
        gap = (target[1].date() - background[1].date()).days
        if gap != key.baseline_days:
            return False
    return True


def load_measured_scenes(
    cfg,
    *,
    bundles_dir: Path | str = "bundles",
    vza_deg: float = S2_NOMINAL_VZA_DEG,
    default_u10_ms: float = 3.0,
) -> list[MeasuredNoiseScene]:
    """Join ``atlas.measured_sigma_log_ratio`` to the geometry of its own scene.

    For each config key the matching bundle supplies the real acquisition
    timestamp, platform, wind and (for cross-checking) the sigma_col the honesty
    gates recorded on that same run. When no bundle matches -- a sigma measured
    on a scene whose bundle was never written, or written after this ran -- the
    scene falls back to the basin centroid at the nominal Sentinel-2 overpass and
    is flagged ``geometry_source="nominal_overpass"``.

    New keys therefore resolve without editing this function, which is what
    keeps ``scripts/make_atlas.py`` re-runnable unchanged as scenes arrive.
    """
    measured = dict(cfg.raw.get("atlas", {}).get("measured_sigma_log_ratio", {}) or {})
    bundles = _index_bundles(Path(bundles_dir))
    scenes: list[MeasuredNoiseScene] = []

    for raw_key, sigma in sorted(measured.items()):
        key = parse_measured_sigma_key(str(raw_key))
        basin = cfg.basins.get(key.basin)
        candidates = [m for m in bundles if _bundle_matches(m, key)]
        # A quotable run wins over a diagnostic one when both fit the key; the
        # unscreened anchor deliberately has only a diagnostic run to point at.
        candidates.sort(
            key=lambda m: bundle_status(m, m.get("_bundle_name", "")) != STATUS_QUOTABLE
        )
        meta = candidates[0] if candidates else None

        lat = lon = None
        satellite = S2_NOMINAL_SATELLITE
        acquired: datetime | None = None
        u10 = default_u10_ms
        event_id = None
        status = "absent"
        recorded = None

        if meta is not None:
            event_id = str(meta.get("event_id") or meta.get("_bundle_name"))
            status = bundle_status(meta, str(meta.get("_bundle_name", "")))
            lat = _as_float(meta.get("lat"))
            lon = _as_float(meta.get("lon"))
            u10 = _as_float(meta.get("u10_ms")) or default_u10_ms
            parsed = _parse_safe(str(meta.get("pixels_target_safe", "")))
            if parsed:
                satellite, acquired = parsed
            gates = meta.get("gates")
            if isinstance(gates, dict):
                recorded = _as_float(gates.get("sigma_col_ppb"))

        if lat is None or lon is None:
            if basin is None:
                raise ValueError(
                    f"measured-sigma key {raw_key!r}: basin {key.basin!r} is not in "
                    f"the config and no bundle supplies coordinates"
                )
            lon = 0.5 * (basin.bbox[0] + basin.bbox[2])
            lat = 0.5 * (basin.bbox[1] + basin.bbox[3])

        if acquired is None:
            acquired = _nominal_overpass_utc(key.day, lon)
            geometry_source = "nominal_overpass"
        else:
            geometry_source = "scene_timestamp"

        scenes.append(
            MeasuredNoiseScene(
                key=str(raw_key),
                basin=key.basin,
                surface_class=basin.surface_class if basin else "unknown",
                day=key.day,
                sigma_log_ratio=float(sigma),
                satellite=satellite,
                sza_deg=solar_zenith_deg(acquired, lat, lon),
                vza_deg=float(vza_deg),
                lat=float(lat),
                lon=float(lon),
                surface_pressure_hpa=float(basin.elevation_hpa) if basin else 1013.0,
                u10_ms=float(u10),
                geometry_source=geometry_source,
                cloud_screened=key.cloud_screened,
                baseline_days=key.baseline_days,
                event_id=event_id,
                bundle_result_status=status,
                recorded_sigma_col_ppb=recorded,
            )
        )
    return scenes


def _as_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


# --------------------------------------------------------------------------
# Empirical floors -- two different quantities, never to be merged
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EmpiricalFloor:
    """A measured emission-rate floor, with whose measurement it is.

    ``label`` is the string that goes on the figure. For our own artifact floor
    it carries the rule-4 wording, so the number cannot be plotted stripped of
    the fact that it is a limitation rather than an emission.
    """

    name: str
    label: str
    low_t_h: float
    high_t_h: float
    n: int
    attribution: str  # "published by others (measured)" | "ours (measured)"
    caveats: tuple[str, ...] = ()
    members: tuple[tuple[str, float], ...] = field(default=(), repr=False)

    @property
    def is_ours(self) -> bool:
        return self.attribution.startswith("ours")


def community_detection_floor(
    truth_path: Path | str = "config/controlled_release_truth.json",
) -> EmpiricalFloor:
    """Detection floor DEMONSTRATED by the published teams, not by us.

    Derived from ``controlled_release_truth.json``: the smallest metered release
    in each single-blind campaign. Those are exactly the emissions the papers
    report as detected (1.4 t/h in 2021 by 3 of 4 teams; 1.05 t/h in 2022 by
    Orbio, to within +/-47%). The file's ``detection_limit_note`` is carried
    through verbatim as ``caveats`` because it contains the two disclaimers that
    must travel with the number: the widely quoted "1-3 t/h" is a VENDOR figure
    and not a measurement, and the known-location test design may have inflated
    sensitivity relative to general practice.
    """
    truth = json.loads(Path(truth_path).read_text())
    members: list[tuple[str, float]] = []
    for campaign in truth.get("campaigns", []):
        rates = [
            float(o["kg_h"]) / 1000.0
            for o in campaign.get("overpasses", [])
            if o.get("kind") == "release" and float(o.get("kg_h", 0.0)) > 0.0
        ]
        if rates:
            members.append((str(campaign.get("name", "?")), min(rates)))
    if not members:
        raise ValueError(f"{truth_path}: no metered releases found")
    lows = [r for _, r in members]
    return EmpiricalFloor(
        name="community_detection_floor",
        label=(
            f"COMMUNITY DETECTION FLOOR {min(lows):.1f}-{max(lows):.1f} t/h — "
            f"DEMONSTRATED BY PUBLISHED TEAMS (Sherwin et al. 2023/2024), NOT BY US"
        ),
        low_t_h=min(lows),
        high_t_h=max(lows),
        n=len(members),
        attribution="published by others (measured)",
        caveats=tuple(truth.get("detection_limit_note", ())),
        members=tuple(members),
    )


def our_artifact_floor(
    bundles_dir: Path | str = "bundles",
    truth_path: Path | str = "config/controlled_release_truth.json",
    *,
    kinds: tuple[str, ...] = ("zero_control",),
) -> EmpiricalFloor:
    """OUR spurious-flux floor, measured on scenes the meter says are zero.

    Reads back the flux the honesty gates WITHHELD on each metered-zero
    overpass. Every one of these numbers is an artifact of our chain; none is an
    emission. Default ``kinds`` is the metered zero controls only. The 2022
    ``no_release`` overpasses are excluded by default because treating them as
    zeros is our inference and not the papers' (see ``kind_semantics`` in the
    truth file); pass them explicitly if that is what you want to show, and say
    so on the figure.
    """
    truth = json.loads(Path(truth_path).read_text())
    wanted: dict[tuple[str, str], str] = {}
    for campaign in truth.get("campaigns", []):
        name = str(campaign.get("name", ""))
        for over in campaign.get("overpasses", []):
            if over.get("kind") in kinds:
                day = str(over.get("date", "")).replace("-", "")
                wanted[(name, day)] = str(over.get("kind"))

    members: list[tuple[str, float]] = []
    for meta in _index_bundles(Path(bundles_dir)):
        name = str(meta.get("basin", ""))
        day = str(meta.get("det_date", "")).replace("-", "")
        if (name, day) not in wanted:
            continue
        withheld = meta.get("q_output_withheld_artifact_dominated")
        rate = _as_float(withheld.get("Q")) if isinstance(withheld, dict) else None
        if rate is None or rate <= 0:
            continue  # correctly reported nothing on this zero -- not a floor point
        members.append((str(meta.get("event_id") or meta.get("_bundle_name")), rate / 1000.0))

    if not members:
        raise ValueError(
            f"{bundles_dir}: no metered-zero bundles with a withheld flux; "
            f"run scripts/controlled_release.py first"
        )
    rates = sorted(r for _, r in members)
    return EmpiricalFloor(
        name="our_quantification_artifact_floor",
        label=(
            f"OUR ARTIFACT FLOOR {rates[0]:.0f}-{rates[-1]:.0f} t/h — spurious flux "
            f"THIS CHAIN reports on METERED-ZERO scenes (n={len(members)}).\n"
            f"A LIMITATION OF OUR PIPELINE, NOT A MEASUREMENT OF AN EMISSION. "
            f"All of it was withheld by the honesty gates."
        ),
        low_t_h=rates[0],
        high_t_h=rates[-1],
        n=len(members),
        attribution="ours (measured, artifact)",
        caveats=(
            "Every value is a false positive on a scene metered at zero emission.",
            "Quantification was withheld on all of them; none was ever reported "
            "as a flux.",
            "The mechanism is mask-wide mean enhancement of 1,361-8,667 ppb, up "
            "to 4.8x the entire ambient column — surface/infrastructure "
            "structure, not gas.",
        ),
        members=tuple(members),
    )


# --------------------------------------------------------------------------
# The analytic surface
# --------------------------------------------------------------------------


def sigma_col_ppb_from_log_ratio(
    sigma_log_ratio: float,
    calibration: RtmCalibration,
    satellite: str,
    sza: float,
    vza: float,
) -> float:
    """Column noise in ppb from calibration-independent band-ratio noise.

    ``sigma_log_ratio`` is the robust 1-sigma of
    ``ln(B11/B12)_target - ln(B11/B12)_reference`` over plume-free pixels --
    a property of the scene and the instrument, not of anyone's absorption
    coefficients.
    """
    if sigma_log_ratio <= 0:
        raise ValueError("sigma_log_ratio must be positive")
    return float(abs(calibration.ppb_from_log_ratio(
        sigma_log_ratio, satellite, sza, vza
    )))


def min_detectable_rate(
    *,
    sigma_col_ppb: float,
    k_sigma: float,
    min_pixels: int,
    pixel_area_m2: float,
    u10_ms: float,
    typical_plume_length_m: float,
    surface_pressure_hpa: float = 1013.0,
    ueff_slope: float = 0.33,
    ueff_intercept: float = 0.45,
) -> float:
    """Minimum detectable emission rate Q_min [kg/h] under the linear model."""
    if sigma_col_ppb <= 0 or min_pixels < 1 or pixel_area_m2 <= 0:
        raise ValueError("invalid limit inputs")
    dxch4_min = k_sigma * sigma_col_ppb
    ime_min = float(column_mass_kg_m2(dxch4_min, surface_pressure_hpa)) * (
        min_pixels * pixel_area_m2
    )
    ueff = effective_wind_speed(u10_ms, ueff_slope, ueff_intercept)
    length = max(typical_plume_length_m, math.sqrt(min_pixels * pixel_area_m2))
    return ime_min * ueff * 3600.0 / length


def qmin_curve(
    scene: MeasuredNoiseScene,
    u10_grid,
    calibration: RtmCalibration,
    *,
    k_sigma: float,
    min_pixels: int = S2_MIN_BLOB_PIXELS,
    pixel_size_m: int = 20,
    typical_plume_length_m: float = 1000.0,
    ueff_slope: float = 0.33,
    ueff_intercept: float = 0.45,
) -> np.ndarray:
    """Q_min [kg/h] versus 10 m wind for one measured-noise scene.

    Noise-limited by construction: the only obstacle it models is radiometric
    noise, so it is a LOWER bound on what a chain can achieve, not a prediction
    of what ours does. Compare it against :func:`our_artifact_floor`.
    """
    sigma_ppb = scene.sigma_col_ppb(calibration)
    area = float(pixel_size_m) ** 2
    return np.array(
        [
            min_detectable_rate(
                sigma_col_ppb=sigma_ppb,
                k_sigma=k_sigma,
                min_pixels=min_pixels,
                pixel_area_m2=area,
                u10_ms=float(u),
                typical_plume_length_m=typical_plume_length_m,
                surface_pressure_hpa=scene.surface_pressure_hpa,
                ueff_slope=ueff_slope,
                ueff_intercept=ueff_intercept,
            )
            for u in np.atleast_1d(u10_grid)
        ],
        dtype=float,
    )


def limit_surface(
    basins: dict[str, dict],
    seasons: list[str],
    u10_by_basin_season: dict[tuple[str, str], float],
    sigma_by_class_season: dict[tuple[str, str], float],
    *,
    k_sigma: float,
    min_pixels: int,
    pixel_size_m: int,
    lengths_by_class: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Assemble Q_min[basin][season] across the atlas grid.

    ``basins`` maps basin name -> {"surface_class": ...}; the two sigma/lookup
    dicts use (surface_class|basin, season) keys as indicated.
    """
    area = float(pixel_size_m) ** 2
    out: dict[str, dict[str, float]] = {}
    for basin, meta in basins.items():
        cls = meta["surface_class"]
        out[basin] = {}
        for season in seasons:
            key = (cls, season)
            if key not in sigma_by_class_season:
                continue
            u10 = u10_by_basin_season.get((basin, season), float("nan"))
            qmin = min_detectable_rate(
                sigma_col_ppb=sigma_by_class_season[key],
                k_sigma=k_sigma,
                min_pixels=min_pixels,
                pixel_area_m2=area,
                u10_ms=u10 if np.isfinite(u10) else 3.0,
                typical_plume_length_m=lengths_by_class.get(cls, 1000.0),
            )
            out[basin][season] = round(qmin, 1)
    return out
