"""Audit of absolute IME fluxes reported by the production model (marss2l).

Why this module exists
----------------------
The first production campaign returned fluxes 6-13x above catalog rates
(docs/S2_REAL_DATA_FINDINGS.md). Before re-running anything we need to know
*which factor* in ``Q = 3600 * U_eff * IME / L`` carries the error. This
module rebuilds that chain from the numbers marss2l already recorded in each
bundle, so the arithmetic can be audited with no network and no re-run.

The central identity
--------------------
Substituting ``IME = mean_ppb * C * N * A_pix`` and ``L = sqrt(N * A_pix)``
into the IME formula collapses it to

    Q [kg/h] = 3600 * U_eff * C * mean_ppb * sqrt(N * A_pix)

with ``C`` the column mass per ppb [kg m^-2 ppb^-1]. Two consequences drive
the whole audit:

* Q is **linear** in the mean in-mask enhancement, so a retrieval bias maps
  straight into the flux.
* Q grows as **sqrt(plume area)**, so an over-large mask inflates the flux
  even when the per-pixel enhancements are untouched. A mask 4x too big
  doubles the reported flux by itself.

Nothing here re-derives methane physics; it re-derives *marss2l's own*
conversion so we can state, with numbers, that the unit path is sound and
the error lives upstream in the enhancement field and the mask.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .mbmp import column_mass_kg_m2

__all__ = [
    "BACKGROUND_CH4_PPB",
    "FluxAudit",
    "audit_q_output",
    "first_principles_kg_m2_per_ppb",
    "marss2l_kg_m2_per_ppb",
    "q_kg_h",
]

# --- marss2l's declared conversion constants (marshsi.quantification) -------
MARSS2L_ATMOSPHERE_HEIGHT_M = 8_000.0  # ppb -> ppm x m uses H/1000 = 8
MOLAR_VOLUME_STP_L_PER_MOL = 22.4
M_CH4_KG_PER_MOL = 0.01604

# Standard-atmosphere CH4 column mole fraction, used for plausibility only.
BACKGROUND_CH4_PPB = 1_800.0


def marss2l_kg_m2_per_ppb() -> float:
    """Column mass per ppb implied by marss2l's ppb -> ppm x m -> kg path.

    marss2l converts ppb to ppm x m with a factor ``H/1000 = 8`` and then
    applies ``ppm x m * 1e-6 * 1000 / 22.4 * M_CH4`` kg per m^2.
    """
    ppm_x_m_per_ppb = MARSS2L_ATMOSPHERE_HEIGHT_M / 1_000.0
    kg_per_m2_per_ppm_x_m = (
        1e-6 * 1_000.0 / MOLAR_VOLUME_STP_L_PER_MOL * M_CH4_KG_PER_MOL
    )
    return ppm_x_m_per_ppb * kg_per_m2_per_ppm_x_m


def first_principles_kg_m2_per_ppb(surface_pressure_hpa: float = 1013.0) -> float:
    """Column mass per ppb from the hydrostatic dry-air column, P/g.

    This is our own independent path (``retrieve.mbmp.column_mass_kg_m2``);
    agreement with :func:`marss2l_kg_m2_per_ppb` is what clears the units.
    """
    return float(column_mass_kg_m2(1.0, surface_pressure_hpa=surface_pressure_hpa))


def q_kg_h(
    mean_enhancement_ppb: float,
    n_mask_px: int,
    pixel_area_m2: float,
    u_eff_ms: float,
    kg_m2_per_ppb: float | None = None,
) -> float:
    """Forward IME flux model, collapsed to its mean-enhancement form.

    Reproduces ``marshsi.quantification.obtain_flux_rate`` exactly for a
    uniform enhancement, which is what makes it usable as a sensitivity tool.
    """
    if n_mask_px < 1 or pixel_area_m2 <= 0:
        raise ValueError("need >=1 masked pixel and positive pixel area")
    c = marss2l_kg_m2_per_ppb() if kg_m2_per_ppb is None else kg_m2_per_ppb
    area_m2 = n_mask_px * pixel_area_m2
    return float(3600.0 * u_eff_ms * c * mean_enhancement_ppb * np.sqrt(area_m2))


@dataclass(frozen=True)
class FluxAudit:
    """Decomposition of one recorded ``obtain_flux_rate`` output."""

    event_id: str
    q_t_h: float
    u_eff_ms: float
    n_mask_px: int
    pixel_area_m2: float
    plume_area_km2: float
    length_m: float
    ime_kg: float
    mean_enhancement_ppb: float
    catalog_t_h: float | None = None
    window_px: int | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    # -- derived diagnostics -------------------------------------------------
    @property
    def ratio_to_catalog(self) -> float | None:
        if not self.catalog_t_h:
            return None
        return self.q_t_h / self.catalog_t_h

    @property
    def mask_fraction(self) -> float | None:
        if not self.window_px:
            return None
        return self.n_mask_px / self.window_px

    @property
    def column_enhancement_factor(self) -> float:
        """Mean in-mask enhancement as a multiple of the background column.

        A value near or above 1 means the retrieval claims the total methane
        column is doubled, sustained across the whole mask -- the single most
        useful implausibility flag at basin scale.
        """
        return self.mean_enhancement_ppb / BACKGROUND_CH4_PPB

    @property
    def mean_ppb_for_catalog_rate(self) -> float | None:
        """Mean enhancement that would reproduce the catalog rate, same mask."""
        if not self.catalog_t_h:
            return None
        return self.mean_enhancement_ppb / self.ratio_to_catalog  # type: ignore[operator]

    def mask_shrink_for_catalog_rate(self) -> float | None:
        """Mask-area factor that alone would reconcile Q with the catalog.

        Since ``Q ~ sqrt(area)`` at fixed mean enhancement, closing a factor
        ``r`` by shrinking the mask needs the area cut by ``r^2``.
        """
        r = self.ratio_to_catalog
        return None if r is None else r**2

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "event_id": self.event_id,
            "q_t_h": round(self.q_t_h, 1),
            "catalog_t_h": self.catalog_t_h,
            "ratio_to_catalog": (
                None if self.ratio_to_catalog is None
                else round(self.ratio_to_catalog, 2)
            ),
            "u_eff_ms": round(self.u_eff_ms, 3),
            "n_mask_px": self.n_mask_px,
            "plume_area_km2": round(self.plume_area_km2, 3),
            "mask_fraction": (
                None if self.mask_fraction is None else round(self.mask_fraction, 4)
            ),
            "length_m": round(self.length_m, 1),
            "ime_kg": round(self.ime_kg, 1),
            "mean_enhancement_ppb": round(self.mean_enhancement_ppb, 1),
            "column_enhancement_factor": round(self.column_enhancement_factor, 2),
            "mean_ppb_for_catalog_rate": (
                None if self.mean_ppb_for_catalog_rate is None
                else round(self.mean_ppb_for_catalog_rate, 1)
            ),
            "mask_shrink_for_catalog_rate": (
                None if self.mask_shrink_for_catalog_rate() is None
                else round(self.mask_shrink_for_catalog_rate(), 1)
            ),
            "notes": list(self.notes),
        }
        return out


def audit_q_output(
    q: dict[str, Any],
    *,
    event_id: str = "unknown",
    catalog_rate_t_h: float | None = None,
    window_px: int | None = None,
) -> FluxAudit:
    """Decompose a recorded marss2l ``obtain_flux_rate`` dict.

    Args:
        q: the ``q_output`` dict stored in a bundle's provenance.json.
        event_id: label for reporting.
        catalog_rate_t_h: published rate for the same event, if known.
        window_px: total pixels in the retrieval window, for mask fraction.

    Returns:
        FluxAudit with the flux decomposed into mean enhancement and area.
    """
    required = {"Q", "L", "npix_plume", "IME", "sum_enhancement", "pixel_size", "u_eff"}
    missing = required - set(q)
    if missing:
        raise KeyError(f"q_output missing keys: {sorted(missing)}")

    npix = int(q["npix_plume"])
    pixel_area = float(q["pixel_size"])
    # marss2l stores sum_enhancement already converted to ppm x m.
    ppm_x_m_per_ppb = MARSS2L_ATMOSPHERE_HEIGHT_M / 1_000.0
    sum_ppb = float(q["sum_enhancement"]) / ppm_x_m_per_ppb
    mean_ppb = sum_ppb / npix if npix else float("nan")

    notes: list[str] = []
    # Cross-check the recorded IME against the reconstructed one; a mismatch
    # would mean the unit path is not what the source says it is.
    ime_recomputed = mean_ppb * marss2l_kg_m2_per_ppb() * npix * pixel_area
    if q["IME"] and abs(ime_recomputed - float(q["IME"])) / float(q["IME"]) > 0.01:
        notes.append(
            f"IME reconstruction mismatch: recorded {float(q['IME']):.0f} kg vs "
            f"reconstructed {ime_recomputed:.0f} kg"
        )

    l_recomputed = float(np.sqrt(npix * pixel_area))
    if abs(l_recomputed - float(q["L"])) > 1.0:
        notes.append(
            f"L is not sqrt(plume area): recorded {float(q['L']):.1f} m vs "
            f"sqrt(A) {l_recomputed:.1f} m"
        )

    return FluxAudit(
        event_id=event_id,
        q_t_h=float(q["Q"]) / 1000.0,
        u_eff_ms=float(q["u_eff"]),
        n_mask_px=npix,
        pixel_area_m2=pixel_area,
        plume_area_km2=npix * pixel_area / 1e6,
        length_m=float(q["L"]),
        ime_kg=float(q["IME"]),
        mean_enhancement_ppb=mean_ppb,
        catalog_t_h=catalog_rate_t_h,
        window_px=window_px,
        notes=tuple(notes),
    )
