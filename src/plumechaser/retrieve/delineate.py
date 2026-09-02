"""Physically-motivated plume delineation — shrinking the mask to what a plume can be.

Why this module exists
----------------------
The 2026-08-25 controlled-release campaign measured this chain's absolute-flux
artifact floor at ~150 t/h against **metered** ground truth
(docs/S2_REAL_DATA_FINDINGS.md). The dominant cause is not the sensor and not
the enhancement retrieval: it is the plume mask. For a 7.18 t/h metered release
at Ehrenberg the chain produced a **2,626-pixel (1.05 km^2)** mask whose mean
in-mask enhancement was **11,718 ppb** — 6.5x the entire ambient methane column.
On the same pixels the published teams retrieved within roughly 2x of truth.

Why the mask is the whole ballgame is the identity in
:mod:`plumechaser.retrieve.flux_audit`:

    Q [kg/h] = 3600 * U_eff * C * mean_ppb * sqrt(plume area)

An oversized mask hurts twice, and both terms move the same way:

1. Q grows as **sqrt(area)**, so area alone inflates the flux.
2. The extra area is where the surface artifacts live (release rig, hard
   standing, vehicles, pads), whose 10% band-ratio offset the RTM inverts into
   ~10,000 ppb — so **mean_ppb** rises too.

Segmentation models answer "does this look like a plume?". They do not answer
"is this shape reachable, by this gas, from this source, under this wind?".
That second question is pure physics and pure geometry, and it is what this
module asks. Everything here is a *constraint on the candidate mask*: it can
only ever remove area (plus small hole-filling), never invent it.

The five constraints, and why each one is physical
--------------------------------------------------
**1. Source connectivity.** A plume is a continuous body of gas advected from
its source. It is spatially contiguous with the source. A scatter of bright
pads across a scene is not one plume; it is several unrelated surfaces that a
threshold happened to catch. Keeping only the connected component that contains
(or is nearest to) the source is the single most effective cut against the
Casa Grande failure mode (1,361 ppb spread over 16.7 km^2).

**2. Downwind sector.** Advection is directional. Material released at the
source can only be found downwind of it. The mask is intersected with a cone
whose apex sits ``upwind_tolerance_m`` *behind* the source (absorbing
source-location error and wind-direction error) and whose half-angle bounds
lateral turbulent spread. A 60 deg half-angle is deliberately permissive:
Pasquill-Gifford lateral growth is sigma_y/x ~ 0.05-0.25 from stable to very
unstable, and a visible envelope of ~3 sigma_y implies a half-angle under 40 deg
even in convective conditions. The remaining margin covers reanalysis wind
direction uncertainty (~20-30 deg for a single hour).

**3. Distance bound.** A snapshot only shows material emitted within the last
tau; older material has been diluted below the retrieval noise by turbulent
spreading (the cross-wind-integrated column falls as ~1/sigma_y(x)). The
advection distance in that time is ``U * tau``, which bounds the plume extent.
This is a **loose outer bound**, not a plume shaper: its job is to remove
distant co-detected infrastructure that happens to be connected and roughly
downwind, not to sculpt the plume. Callers who can bound the extent better
(from a detection-limit calculation, or from analyst QC) pass ``max_extent_m``
directly.

**4. Amplitude sanity.** The ambient CH4 column is ~1,800 ppb
(:data:`plumechaser.retrieve.flux_audit.BACKGROUND_CH4_PPB`). A retrieved
*per-pixel* enhancement of several times that is not methane; it is a surface
whose B12/B11 ratio changed between passes. ``retrieve/gates.py`` already
encodes this idea for the mask-wide *mean* (and rightly bounds it at 1.0x,
since a mask-wide mean above ambient is impossible for a point source). The
per-pixel bound here must be looser than the gate's, because a real plume
*core* near the stack can legitimately approach the ambient column — hence a
default of 2.0x. Note the direction of this filter: it drops the **brightest**
pixels, which is the opposite of what a detection threshold does, and is
exactly why it removes artifacts a threshold cannot.

**5. Morphological cleanup.** Retrieval noise produces isolated specks that
inflate area, and thin one-pixel bridges that let a bright pad inherit the
plume's connectivity. An opening removes both. A closing plus small-hole fill
repairs dropouts *inside* the plume body, which otherwise understate IME.

Relationship to the existing chain
----------------------------------
This is new capability sitting alongside the existing path. It changes no
default, no threshold and no science constant anywhere else in the repo:
``retrieve/gates.py`` still decides whether a flux may be quoted at all, and
this module only decides which pixels the flux is computed over. Delineating
first and gating second is the intended order — the gates then judge a mask
that has already been asked to be physical.

Nothing here can rescue a scene that has no plume in it. Applied to the
Ehrenberg **metered-zero** control it should return a small mask or an empty
one, and an empty mask is the correct answer there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

from .flux_audit import BACKGROUND_CH4_PPB

__all__ = [
    "DEFAULT_MAX_TRANSPORT_TIME_S",
    "DEFAULT_PIXEL_SIZE_M",
    "DEFAULT_SECTOR_HALF_ANGLE_DEG",
    "DelineationResult",
    "RULE_ORDER",
    "delineate_plume",
    "downwind_unit_pixel_vector",
    "rejection_report",
]

# --- documented defaults (deliberately NOT config keys; see module docstring) ---
DEFAULT_PIXEL_SIZE_M = 20.0  # Sentinel-2 B11/B12 native ground sample distance
DEFAULT_SECTOR_HALF_ANGLE_DEG = 60.0
DEFAULT_UPWIND_TOLERANCE_M = 100.0
DEFAULT_MAX_TRANSPORT_TIME_S = 900.0  # 15 min of advection: a loose outer bound
DEFAULT_MAX_ENHANCEMENT_RATIO = 2.0  # per-pixel, as a multiple of ambient column
DEFAULT_MIN_COMPONENT_PX = 4
DEFAULT_FILL_HOLE_MAX_PX = 25

# Rule identifiers. Stable strings: campaigns log these.
RULE_INVALID = "invalid"
RULE_AMPLITUDE = "amplitude"
RULE_MORPH_OPEN = "morphology_open"
RULE_MORPH_CLOSE = "morphology_close"
RULE_MORPH_FILL = "morphology_fill"
RULE_MORPH_MIN_SIZE = "morphology_min_size"
RULE_SOURCE_CONNECTIVITY = "source_connectivity"
RULE_DOWNWIND_SECTOR = "downwind_sector"
RULE_DISTANCE_BOUND = "distance_bound"
RULE_RECONNECT = "reconnect"

#: Order in which the rules are applied. Ordering is itself a design decision:
#: amplitude first (so impossible pixels cannot anchor a component), morphology
#: next (so a one-pixel bridge cannot hand an artifact the plume's
#: connectivity), then the source-relative geometry, then a final connectivity
#: re-check because clipping a sector can sever the retained body from its
#: source.
RULE_ORDER = (
    RULE_INVALID,
    RULE_AMPLITUDE,
    RULE_MORPH_OPEN,
    RULE_MORPH_CLOSE,
    RULE_MORPH_FILL,
    RULE_MORPH_MIN_SIZE,
    RULE_SOURCE_CONNECTIVITY,
    RULE_DOWNWIND_SECTOR,
    RULE_DISTANCE_BOUND,
    RULE_RECONNECT,
)

_ADDITIVE_RULES = frozenset({RULE_MORPH_CLOSE, RULE_MORPH_FILL})


def downwind_unit_pixel_vector(
    wind_vector_ms: tuple[float, float] | np.ndarray,
    *,
    north_up: bool = True,
) -> tuple[float, float]:
    """Unit vector pointing **downwind** in ``(row, col)`` raster space.

    The repo's canonical wind representation (``data/openmeteo.py``,
    ``data/winds.py``) is ``(u10, v10)``: eastward and northward components of
    the vector the wind blows *toward*. A north-up raster has columns
    increasing eastward and rows increasing **southward**, so the northward
    component flips sign on its way into pixel space. That sign is the single
    easiest thing to get wrong in this module, so it lives in one tested place.

    Args:
        wind_vector_ms: ``(u_east, v_north)`` in m/s — the direction the wind
            blows toward, *not* the meteorological "direction from".
        north_up: True for a conventional north-up raster. Pass False if rows
            increase northward.

    Returns:
        ``(d_row, d_col)``, unit length.
    """
    vec = np.asarray(wind_vector_ms, dtype=float).ravel()
    if vec.size < 2:
        raise ValueError("wind_vector_ms must have two components (u_east, v_north)")
    u_east, v_north = float(vec[0]), float(vec[1])
    speed = float(np.hypot(u_east, v_north))
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("wind vector must have a positive, finite speed")
    d_col = u_east / speed
    d_row = (-v_north if north_up else v_north) / speed
    return (d_row, d_col)


@dataclass(frozen=True, eq=False)
class DelineationResult:
    """The retained mask plus a full, per-rule account of what was removed.

    The bookkeeping invariant, which :func:`rejection_report` re-checks and the
    tests assert, is exact::

        n_kept_px == n_input_px - sum(dropped.values()) + sum(added.values())

    Every stage records the actual change in mask population it caused, so no
    pixel is double-counted and none goes missing.
    """

    mask: np.ndarray
    n_input_px: int
    n_kept_px: int
    dropped: dict[str, int]
    added: dict[str, int]
    pixel_area_m2: float
    mean_ppb_before: float = float("nan")
    mean_ppb_after: float = float("nan")
    source_rc: tuple[float, float] | None = None
    downwind_unit: tuple[float, float] | None = None
    max_extent_m: float | None = None
    rules_applied: tuple[str, ...] = field(default_factory=tuple)
    rules_skipped: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_dropped_px(self) -> int:
        return int(sum(self.dropped.values()))

    @property
    def n_added_px(self) -> int:
        return int(sum(self.added.values()))

    @property
    def area_km2_before(self) -> float:
        return self.n_input_px * self.pixel_area_m2 / 1e6

    @property
    def area_km2_after(self) -> float:
        return self.n_kept_px * self.pixel_area_m2 / 1e6

    @property
    def area_shrink_factor(self) -> float:
        """Input area divided by retained area (inf if nothing is retained)."""
        if self.n_kept_px == 0:
            return float("inf")
        return self.n_input_px / self.n_kept_px

    @property
    def dominant_rule(self) -> str | None:
        """Rule that removed the most pixels — the one-word campaign summary."""
        if not self.dropped:
            return None
        best = max(self.dropped.items(), key=lambda kv: kv[1])
        return best[0] if best[1] > 0 else None

    @property
    def implied_flux_factor(self) -> float:
        """Factor by which delineation changes Q, via the flux_audit identity.

        ``Q ~ mean_ppb * sqrt(area)``, so the delineated flux is the raw flux
        times ``(mean_after / mean_before) * sqrt(n_after / n_before)``. This is
        the number a campaign wants in its log: it states, in one figure, what
        the mask correction did to the reported emission rate.
        """
        if self.n_input_px == 0:
            return float("nan")
        if self.n_kept_px == 0:
            return 0.0
        area_term = float(np.sqrt(self.n_kept_px / self.n_input_px))
        before, after = self.mean_ppb_before, self.mean_ppb_after
        if not (np.isfinite(before) and np.isfinite(after)) or before == 0.0:
            return area_term
        return float(after / before * area_term)

    def as_dict(self) -> dict[str, Any]:
        """Flat, JSON-safe form for provenance records."""
        return {
            "n_input_px": self.n_input_px,
            "n_kept_px": self.n_kept_px,
            "n_dropped_px": self.n_dropped_px,
            "n_added_px": self.n_added_px,
            "area_km2_before": round(self.area_km2_before, 4),
            "area_km2_after": round(self.area_km2_after, 4),
            "area_shrink_factor": (
                None if self.n_kept_px == 0 else round(self.area_shrink_factor, 2)
            ),
            "mean_ppb_before": (
                None if not np.isfinite(self.mean_ppb_before)
                else round(self.mean_ppb_before, 1)
            ),
            "mean_ppb_after": (
                None if not np.isfinite(self.mean_ppb_after)
                else round(self.mean_ppb_after, 1)
            ),
            "implied_flux_factor": (
                None if not np.isfinite(self.implied_flux_factor)
                else round(self.implied_flux_factor, 4)
            ),
            "dropped_px": dict(self.dropped),
            "added_px": dict(self.added),
            "dominant_rule": self.dominant_rule,
            "source_rc": list(self.source_rc) if self.source_rc else None,
            "downwind_unit_rc": (
                None if self.downwind_unit is None
                else [round(v, 4) for v in self.downwind_unit]
            ),
            "max_extent_m": (
                None if self.max_extent_m is None else round(self.max_extent_m, 1)
            ),
            "rules_applied": list(self.rules_applied),
            "rules_skipped": list(self.rules_skipped),
            "notes": list(self.notes),
        }


class _Ledger:
    """Applies one rule to the running mask and records the exact delta."""

    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask
        self.dropped: dict[str, int] = {}
        self.added: dict[str, int] = {}
        self.applied: list[str] = []
        self.skipped: list[str] = []
        self.notes: list[str] = []

    def apply(self, rule: str, new_mask: np.ndarray) -> None:
        before = int(self.mask.sum())
        after = int(new_mask.sum())
        self.mask = new_mask
        self.applied.append(rule)
        delta = before - after
        if delta > 0:
            self.dropped[rule] = self.dropped.get(rule, 0) + delta
        elif delta < 0:
            self.added[rule] = self.added.get(rule, 0) - delta
        else:
            # Record a zero so the log shows the rule ran and found nothing.
            target = self.added if rule in _ADDITIVE_RULES else self.dropped
            target.setdefault(rule, 0)

    def skip(self, rule: str, why: str) -> None:
        self.skipped.append(rule)
        self.notes.append(f"{rule} not applied: {why}")


def _binary_structure(connectivity: int) -> np.ndarray:
    if connectivity not in (1, 2):
        raise ValueError("connectivity must be 1 (4-neighbour) or 2 (8-neighbour)")
    return ndimage.generate_binary_structure(2, connectivity)


def _padded_morphology(mask: np.ndarray, op: Any, structure: np.ndarray, radius: int) -> np.ndarray:
    """Run a morphological op with enough zero padding to avoid border effects.

    scipy applies ``border_value=0`` outside the array, which lets a closing
    erode foreground that touches the array edge. Padding first makes both
    operations behave as they do mathematically: opening is anti-extensive,
    closing is extensive, at the border as everywhere else.
    """
    pad = radius + 1
    padded = np.pad(mask, pad, mode="constant", constant_values=False)
    out = op(padded, structure=structure, iterations=radius)
    return np.asarray(out[pad:-pad, pad:-pad], dtype=bool)


def _remove_small_components(
    mask: np.ndarray, structure: np.ndarray, min_px: int
) -> np.ndarray:
    if min_px <= 1 or not mask.any():
        return mask
    labels, n = ndimage.label(mask, structure=structure)
    if n == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = np.zeros(sizes.size, dtype=bool)
    keep[1:] = sizes[1:] >= min_px
    return keep[labels]


def _fill_small_holes(mask: np.ndarray, max_px: int) -> np.ndarray:
    """Fill enclosed background regions no larger than ``max_px``.

    A hole inside the plume body is a retrieval dropout, not a gap in the gas.
    Leaving it out understates IME and, through ``sqrt(area)``, understates Q.
    Only *small* holes are filled; a large enclosed void is real structure and
    filling it would be inventing plume.
    """
    if max_px < 1 or not mask.any():
        return mask
    filled = ndimage.binary_fill_holes(mask)
    holes = filled & ~mask
    if not holes.any():
        return mask
    # Holes are 4-connected background regions; label them the same way.
    labels, n = ndimage.label(holes, structure=_binary_structure(1))
    if n == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    small = np.zeros(sizes.size, dtype=bool)
    small[1:] = sizes[1:] <= max_px
    return mask | small[labels]


def _source_component(
    mask: np.ndarray,
    structure: np.ndarray,
    source_rc: tuple[float, float],
    pixel_size_m: float,
) -> tuple[np.ndarray, str | None]:
    """Connected component containing the source, else the nearest component."""
    labels, n = ndimage.label(mask, structure=structure)
    if n == 0:
        return mask, None
    n_rows, n_cols = mask.shape
    src_r = int(np.clip(round(source_rc[0]), 0, n_rows - 1))
    src_c = int(np.clip(round(source_rc[1]), 0, n_cols - 1))
    label_at_source = int(labels[src_r, src_c])
    note: str | None = None
    if label_at_source == 0:
        rows, cols = np.nonzero(mask)
        d2 = (rows - source_rc[0]) ** 2 + (cols - source_rc[1]) ** 2
        nearest = int(np.argmin(d2))
        label_at_source = int(labels[rows[nearest], cols[nearest]])
        offset_m = float(np.sqrt(d2[nearest])) * pixel_size_m
        note = (
            f"source pixel is not inside the candidate mask; kept the nearest "
            f"component, {offset_m:.0f} m away"
        )
    return labels == label_at_source, note


def _downwind_sector_mask(
    shape: tuple[int, int],
    source_rc: tuple[float, float],
    downwind_unit: tuple[float, float],
    pixel_size_m: float,
    half_angle_deg: float,
    upwind_tolerance_m: float,
) -> np.ndarray:
    """Cone of admissible pixels, apex placed ``upwind_tolerance_m`` upwind.

    Placing the apex behind the source rather than on it is what makes the
    constraint usable in practice: at the source itself the cone would have
    zero width, and both the source coordinate and the wind direction carry
    real uncertainty. With the apex offset, the cone is
    ``tan(half_angle) * upwind_tolerance_m`` wide where the source sits — 173 m
    at the defaults — which is the right order for a 20 m grid.
    """
    if not 0.0 < half_angle_deg < 90.0:
        raise ValueError("sector half-angle must be in (0, 90) degrees")
    if upwind_tolerance_m < 0.0:
        raise ValueError("upwind tolerance must be non-negative")
    rows = (np.arange(shape[0], dtype=float) - source_rc[0]) * pixel_size_m
    cols = (np.arange(shape[1], dtype=float) - source_rc[1]) * pixel_size_m
    dr = rows[:, None]
    dc = cols[None, :]
    d_row, d_col = downwind_unit
    # Along-wind coordinate, measured from the cone apex rather than the source.
    along = dr * d_row + dc * d_col + upwind_tolerance_m
    # Cross-wind distance (2-D cross product magnitude with a unit vector).
    cross = np.abs(dr * d_col - dc * d_row)
    return (along >= 0.0) & (cross <= np.tan(np.radians(half_angle_deg)) * along)


def _radial_distance_m(
    shape: tuple[int, int], source_rc: tuple[float, float], pixel_size_m: float
) -> np.ndarray:
    rows = (np.arange(shape[0], dtype=float) - source_rc[0]) * pixel_size_m
    cols = (np.arange(shape[1], dtype=float) - source_rc[1]) * pixel_size_m
    return np.hypot(rows[:, None], cols[None, :])


def _masked_mean(field_arr: np.ndarray | None, mask: np.ndarray) -> float:
    if field_arr is None or not mask.any():
        return float("nan")
    vals = field_arr[mask]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def delineate_plume(
    candidate_mask: np.ndarray,
    *,
    enhancement_ppb: np.ndarray | None = None,
    source_rc: tuple[float, float] | None = None,
    wind_vector_ms: tuple[float, float] | np.ndarray | None = None,
    u10_ms: float | None = None,
    pixel_size_m: float = DEFAULT_PIXEL_SIZE_M,
    valid: np.ndarray | None = None,
    north_up: bool = True,
    # --- rule switches: each technique is independently attributable ---------
    apply_amplitude: bool = True,
    apply_morphology: bool = True,
    apply_source_connectivity: bool = True,
    apply_downwind_sector: bool = True,
    apply_distance_bound: bool = True,
    # --- rule parameters ----------------------------------------------------
    max_enhancement_ratio: float = DEFAULT_MAX_ENHANCEMENT_RATIO,
    min_enhancement_ppb: float | None = 0.0,
    open_radius_px: int = 1,
    close_radius_px: int = 1,
    fill_hole_max_px: int = DEFAULT_FILL_HOLE_MAX_PX,
    min_component_px: int = DEFAULT_MIN_COMPONENT_PX,
    connectivity: int = 2,
    sector_half_angle_deg: float = DEFAULT_SECTOR_HALF_ANGLE_DEG,
    upwind_tolerance_m: float = DEFAULT_UPWIND_TOLERANCE_M,
    max_extent_m: float | None = None,
    max_transport_time_s: float = DEFAULT_MAX_TRANSPORT_TIME_S,
) -> DelineationResult:
    """Reduce a candidate plume mask to the part that could physically be a plume.

    Args:
        candidate_mask: (H, W) boolean/0-1 mask from any upstream detector —
            a segmentation model, a sigma threshold, an analyst polygon.
        enhancement_ppb: (H, W) column enhancement field, same grid. Required
            for the amplitude rule and for the before/after mean statistics;
            without it the amplitude rule is skipped and recorded as skipped.
        source_rc: known or estimated source position as ``(row, col)`` in
            pixel coordinates (fractional allowed). Required by the three
            source-relative rules; without it they are skipped, and the mask is
            **not** silently reduced to its largest component — doing that is
            precisely how the Ehrenberg artifact would have been retained.
        wind_vector_ms: ``(u_east, v_north)`` m/s, the direction the wind blows
            toward, matching ``data/openmeteo.py``. Required by the downwind
            sector; also supplies the speed for the distance bound when
            ``u10_ms`` is not given.
        u10_ms: 10 m wind speed for the distance bound, if the vector is not
            available or a different speed estimate is preferred.
        pixel_size_m: ground sample distance. Defaults to Sentinel-2 SWIR.
        valid: optional (H, W) mask of usable pixels; invalid pixels are always
            dropped, mirroring ``retrieve/gates.py``.
        north_up: raster row orientation; see
            :func:`downwind_unit_pixel_vector`.
        apply_amplitude: enable the per-pixel physical-plausibility filter.
        apply_morphology: enable opening, closing, small-hole fill and
            minimum-component-size removal.
        apply_source_connectivity: enable source-component selection (and the
            post-clip re-connection check).
        apply_downwind_sector: enable the downwind cone constraint.
        apply_distance_bound: enable the maximum-extent constraint.
        max_enhancement_ratio: per-pixel enhancement ceiling as a multiple of
            the ambient column (1,800 ppb). Looser than the mask-wide gate in
            ``retrieve/gates.py`` because a plume core may approach ambient.
        min_enhancement_ppb: per-pixel floor. Default 0.0 drops non-positive
            retrievals, since a plume is an *enhancement*; pass None to
            disable.
        open_radius_px: erosion/dilation iterations for the opening. 1 removes
            single-pixel specks and one-pixel bridges.
        close_radius_px: iterations for the closing.
        fill_hole_max_px: largest enclosed hole that is treated as a dropout
            and filled.
        min_component_px: components smaller than this are removed. At 20 m,
            4 px = 1,600 m^2 — below that a "plume" is not resolved.
        connectivity: 1 for 4-neighbour, 2 for 8-neighbour labelling.
        sector_half_angle_deg: half-angle of the downwind cone.
        upwind_tolerance_m: how far upwind of the source the cone apex sits.
        max_extent_m: explicit maximum plume extent. If None it is derived as
            ``wind speed * max_transport_time_s`` (floored at two pixels).
        max_transport_time_s: advection time used for that derivation.

    Returns:
        :class:`DelineationResult` — the retained mask plus per-rule counts.

    Raises:
        ValueError: on shape mismatches or out-of-range parameters.
    """
    mask0 = np.asarray(candidate_mask).astype(bool)
    if mask0.ndim != 2:
        raise ValueError(f"candidate_mask must be 2-D, got shape {mask0.shape}")
    shape: tuple[int, int] = (int(mask0.shape[0]), int(mask0.shape[1]))
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive")

    field_arr: np.ndarray | None = None
    if enhancement_ppb is not None:
        field_arr = np.asarray(enhancement_ppb, dtype=float)
        if field_arr.shape != mask0.shape:
            raise ValueError(
                f"enhancement and mask shapes differ: {field_arr.shape} != {mask0.shape}"
            )
    if valid is not None:
        valid_arr = np.asarray(valid).astype(bool)
        if valid_arr.shape != mask0.shape:
            raise ValueError(
                f"valid and mask shapes differ: {valid_arr.shape} != {mask0.shape}"
            )
    else:
        valid_arr = None

    structure = _binary_structure(connectivity)
    ledger = _Ledger(mask0)
    n_input = int(mask0.sum())
    mean_before = _masked_mean(field_arr, mask0)

    # -- wind geometry, resolved once ---------------------------------------
    downwind: tuple[float, float] | None = None
    if wind_vector_ms is not None:
        downwind = downwind_unit_pixel_vector(wind_vector_ms, north_up=north_up)
    speed_ms: float | None = None
    if u10_ms is not None:
        speed_ms = float(u10_ms)
    elif wind_vector_ms is not None:
        vec = np.asarray(wind_vector_ms, dtype=float).ravel()
        speed_ms = float(np.hypot(vec[0], vec[1]))
    if speed_ms is not None and (not np.isfinite(speed_ms) or speed_ms < 0):
        raise ValueError("wind speed must be non-negative and finite")

    extent_m: float | None = max_extent_m
    if extent_m is None and speed_ms is not None:
        if max_transport_time_s <= 0:
            raise ValueError("max_transport_time_s must be positive")
        # Advection distance in one snapshot's worth of transport time, floored
        # so that a near-calm scene cannot clip the source neighbourhood away.
        extent_m = max(speed_ms * max_transport_time_s, 2.0 * pixel_size_m)
    if extent_m is not None and extent_m <= 0:
        raise ValueError("max_extent_m must be positive")

    # -- rule 0: data validity (always on) ----------------------------------
    if valid_arr is not None:
        ledger.apply(RULE_INVALID, ledger.mask & valid_arr)

    # -- rule 1: amplitude sanity -------------------------------------------
    if apply_amplitude:
        if field_arr is None:
            ledger.skip(RULE_AMPLITUDE, "no enhancement field supplied")
        else:
            if max_enhancement_ratio <= 0:
                raise ValueError("max_enhancement_ratio must be positive")
            ceiling = max_enhancement_ratio * BACKGROUND_CH4_PPB
            with np.errstate(invalid="ignore"):
                keep = np.isfinite(field_arr) & (field_arr <= ceiling)
                if min_enhancement_ppb is not None:
                    keep &= field_arr > float(min_enhancement_ppb)
            ledger.apply(RULE_AMPLITUDE, ledger.mask & keep)
    else:
        ledger.skip(RULE_AMPLITUDE, "disabled by caller")

    # -- rule 2: morphological cleanup --------------------------------------
    if apply_morphology:
        if open_radius_px < 0 or close_radius_px < 0:
            raise ValueError("morphology radii must be non-negative")
        if open_radius_px > 0:
            ledger.apply(
                RULE_MORPH_OPEN,
                _padded_morphology(
                    ledger.mask, ndimage.binary_opening, structure, open_radius_px
                ),
            )
        if close_radius_px > 0:
            ledger.apply(
                RULE_MORPH_CLOSE,
                _padded_morphology(
                    ledger.mask, ndimage.binary_closing, structure, close_radius_px
                ),
            )
        if fill_hole_max_px > 0:
            ledger.apply(RULE_MORPH_FILL, _fill_small_holes(ledger.mask, fill_hole_max_px))
        if min_component_px > 1:
            ledger.apply(
                RULE_MORPH_MIN_SIZE,
                _remove_small_components(ledger.mask, structure, min_component_px),
            )
    else:
        for rule in (RULE_MORPH_OPEN, RULE_MORPH_CLOSE, RULE_MORPH_FILL, RULE_MORPH_MIN_SIZE):
            ledger.skip(rule, "disabled by caller")

    # -- rule 3: source connectivity ----------------------------------------
    if apply_source_connectivity:
        if source_rc is None:
            ledger.skip(RULE_SOURCE_CONNECTIVITY, "no source position supplied")
        elif not ledger.mask.any():
            ledger.skip(RULE_SOURCE_CONNECTIVITY, "mask already empty")
        else:
            kept, note = _source_component(ledger.mask, structure, source_rc, pixel_size_m)
            if note:
                ledger.notes.append(note)
            ledger.apply(RULE_SOURCE_CONNECTIVITY, kept)
    else:
        ledger.skip(RULE_SOURCE_CONNECTIVITY, "disabled by caller")

    # -- rule 4: downwind sector --------------------------------------------
    if apply_downwind_sector:
        if source_rc is None or downwind is None:
            ledger.skip(
                RULE_DOWNWIND_SECTOR,
                "needs both a source position and a wind vector",
            )
        else:
            sector = _downwind_sector_mask(
                shape,
                source_rc,
                downwind,
                pixel_size_m,
                sector_half_angle_deg,
                upwind_tolerance_m,
            )
            ledger.apply(RULE_DOWNWIND_SECTOR, ledger.mask & sector)
    else:
        ledger.skip(RULE_DOWNWIND_SECTOR, "disabled by caller")

    # -- rule 5: distance bound ---------------------------------------------
    if apply_distance_bound:
        if source_rc is None or extent_m is None:
            ledger.skip(
                RULE_DISTANCE_BOUND,
                "needs a source position and either max_extent_m or a wind speed",
            )
        else:
            within = _radial_distance_m(shape, source_rc, pixel_size_m) <= extent_m
            ledger.apply(RULE_DISTANCE_BOUND, ledger.mask & within)
    else:
        ledger.skip(RULE_DISTANCE_BOUND, "disabled by caller")

    # -- rule 6: re-check connectivity after the geometric clips ------------
    # Clipping a sector or a radius can sever the retained body, leaving an
    # island that is downwind and close but no longer continuous with the
    # source. That is not a plume either.
    clipped_geometrically = (
        RULE_DOWNWIND_SECTOR in ledger.applied or RULE_DISTANCE_BOUND in ledger.applied
    )
    if apply_source_connectivity and clipped_geometrically and source_rc is not None:
        if ledger.mask.any():
            kept, _ = _source_component(ledger.mask, structure, source_rc, pixel_size_m)
            ledger.apply(RULE_RECONNECT, kept)
        else:
            ledger.skip(RULE_RECONNECT, "mask empty after geometric clipping")

    final_mask = ledger.mask
    n_kept = int(final_mask.sum())
    if n_kept != n_input - sum(ledger.dropped.values()) + sum(ledger.added.values()):
        # Guard against a future edit silently breaking the attribution ledger.
        raise AssertionError("delineation ledger does not balance")

    if n_kept == 0 and n_input > 0:
        ledger.notes.append(
            "no candidate pixels survived delineation — an empty plume mask is a "
            "valid outcome and means no quantifiable plume, not a failure"
        )

    return DelineationResult(
        mask=final_mask,
        n_input_px=n_input,
        n_kept_px=n_kept,
        dropped=dict(ledger.dropped),
        added=dict(ledger.added),
        pixel_area_m2=pixel_size_m * pixel_size_m,
        mean_ppb_before=mean_before,
        mean_ppb_after=_masked_mean(field_arr, final_mask),
        source_rc=None if source_rc is None else (float(source_rc[0]), float(source_rc[1])),
        downwind_unit=downwind,
        max_extent_m=extent_m,
        rules_applied=tuple(ledger.applied),
        rules_skipped=tuple(ledger.skipped),
        notes=tuple(ledger.notes),
    )


def rejection_report(result: DelineationResult) -> dict[str, Any]:
    """Why pixels were dropped, in a form a campaign can log verbatim.

    Returns the per-rule counts in :data:`RULE_ORDER`, each rule's share of the
    input mask, and the consequence that actually matters: the factor by which
    the reported flux changes, from the ``flux_audit`` identity
    ``Q ~ mean_ppb * sqrt(area)``.

    The ``balanced`` field re-derives the bookkeeping invariant from the
    reported numbers, so a log entry is self-checking.
    """
    n_in = result.n_input_px
    rules: list[dict[str, Any]] = []
    for rule in RULE_ORDER:
        dropped = int(result.dropped.get(rule, 0))
        added = int(result.added.get(rule, 0))
        if rule in result.rules_skipped:
            status = "skipped"
        elif rule in result.rules_applied:
            status = "applied"
        else:
            status = "not_run"
        if status == "not_run" and dropped == 0 and added == 0:
            continue
        rules.append(
            {
                "rule": rule,
                "status": status,
                "dropped_px": dropped,
                "added_px": added,
                "dropped_frac_of_input": (round(dropped / n_in, 4) if n_in else 0.0),
            }
        )

    return {
        "n_input_px": n_in,
        "n_kept_px": result.n_kept_px,
        "n_dropped_px": result.n_dropped_px,
        "n_added_px": result.n_added_px,
        "kept_frac_of_input": round(result.n_kept_px / n_in, 4) if n_in else 0.0,
        "area_km2_before": round(result.area_km2_before, 4),
        "area_km2_after": round(result.area_km2_after, 4),
        "mean_ppb_before": (
            None if not np.isfinite(result.mean_ppb_before)
            else round(result.mean_ppb_before, 1)
        ),
        "mean_ppb_after": (
            None if not np.isfinite(result.mean_ppb_after)
            else round(result.mean_ppb_after, 1)
        ),
        "implied_flux_factor": (
            None if not np.isfinite(result.implied_flux_factor)
            else round(result.implied_flux_factor, 4)
        ),
        "dominant_rule": result.dominant_rule,
        "rules": rules,
        "balanced": (
            result.n_kept_px == n_in - result.n_dropped_px + result.n_added_px
        ),
        "notes": list(result.notes),
    }
