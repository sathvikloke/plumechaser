"""Multi-band multi-pass (MBMP) methane column retrieval from SWIR band pairs.

Science basis
-------------
Varon et al. (2021, AMT 14, 2771) showed that Sentinel-2 bands B11
(~1600 nm, weak CH4 absorption) and B12 (~2200 nm, strong CH4 absorption)
can be combined into a methane column-enhancement retrieval. We implement the
multi-pass formulation in log space:

    u(pass) = ln(B11 / B12)

For modest enhancements the band reflectances scale as
B_b(pass) = B_b,surface * exp(-alpha_b * dX), so

    u = ln(B11_sfc / B12_sfc) + (alpha_12 - alpha_11) * dX

and differencing two passes cancels the unknown surface term exactly when the
surface is unchanged between passes:

    dX_ppb = [u(target) - u(reference)] / (alpha_12 - alpha_11)

Simplification vs. production systems
-------------------------------------
Varon et al. derive ``alpha`` from a 100-layer radiative-transfer model
spectrally convolved to the MSI response. We use single effective coefficients
(config ``mbmp.alpha_*_per_ppb``) seeded from literature magnitudes. Absolute
rates therefore carry a calibration caveat until the RTM-LUT step
(``retrieve/calibration.py``, roadmap) replaces them. Structure, artifact
cancellation, masking, and uncertainty propagation are faithful; only the
band-integrated absorption coefficient is simplified.

Reference handling follows the frozen analysis plan: the cleanest pass within
+/-12 days (excluding +/-2 days around the target), scored by cloud fraction,
surface stability, and absence of its own anomalies (see cue/reference.py).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# Molar masses, kg/kmol (IUPAC)
M_CH4 = 16.043
M_DRY_AIR = 28.965
G_ACCEL = 9.80665  # m s^-2


def log_band_ratio(b11: np.ndarray, b12: np.ndarray) -> np.ndarray:
    """u = ln(B11/B12); nonpositive or nonfinite radiances become NaN."""
    b11 = np.asarray(b11, dtype=np.float64)
    b12 = np.asarray(b12, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = b11 / b12
        return np.log(np.where((ratio > 0) & np.isfinite(ratio), ratio, np.nan))


def mbmp_enhancement_ppb(
    b11_target: np.ndarray,
    b12_target: np.ndarray,
    b11_reference: np.ndarray,
    b12_reference: np.ndarray,
    alpha_b11_per_ppb: float,
    alpha_b12_per_ppb: float,
) -> np.ndarray:
    """Methane column enhancement map (ppb) between target and reference passes.

    Positive values mean more methane in the target pass. Pixels invalid in
    either pass propagate as NaN.
    """
    if alpha_b12_per_ppb <= alpha_b11_per_ppb:
        raise ValueError("alpha_b12 must exceed alpha_b11 for a valid contrast")
    u_t = log_band_ratio(b11_target, b12_target)
    u_r = log_band_ratio(b11_reference, b12_reference)
    return (u_t - u_r) / (alpha_b12_per_ppb - alpha_b11_per_ppb)


def robust_scene_sigma(field: np.ndarray, floor_ppb: float = 2.0) -> float:
    """Robust noise estimate of an enhancement field via MAD (1-sigma equiv).

    MAD -> sigma conversion uses the normal-consistency factor 1.4826.
    """
    finite = field[np.isfinite(field)]
    if finite.size == 0:
        return floor_ppb
    med = np.median(finite)
    mad = np.median(np.abs(finite - med))
    return max(1.4826 * mad, floor_ppb)


def plume_mask(
    d_xch4_ppb: np.ndarray,
    threshold_sigma: float = 3.0,
    median_size: int = 3,
) -> np.ndarray:
    """Boolean plume mask: pixels above scene-noise threshold, despeckled.

    The threshold is applied on the raw field first; a median filter then
    removes isolated single-pixel spikes that survive it.
    """
    sigma = robust_scene_sigma(d_xch4_ppb)
    finite = np.nan_to_num(d_xch4_ppb, nan=-np.inf)
    raw = finite >= threshold_sigma * sigma
    if not raw.any():
        return raw.astype(bool)
    cleaned = ndimage.median_filter(raw.astype(np.uint8), size=median_size).astype(bool)
    return cleaned & np.isfinite(d_xch4_ppb)


def column_mass_kg_m2(
    d_xch4_ppb: np.ndarray | float,
    surface_pressure_hpa: float = 1013.0,
) -> np.ndarray | float:
    """Convert a dry-air mole-fraction enhancement (ppb) to excess mass (kg/m^2).

    Approximates the dry-air column as P/(M_air * ... ) collapsed to
    ``P_surface / g`` kilograms per square metre, times the CH4 mole fraction
    and molar-mass ratio. Water-vapour loading (<1% by mass) is neglected;
    this is consistent with the precision being reported anyway.
    """
    column_dry_air_kg_m2 = surface_pressure_hpa * 100.0 / G_ACCEL
    return (
        np.asarray(d_xch4_ppb, dtype=np.float64)
        * 1e-9
        * (M_CH4 / M_DRY_AIR)
        * column_dry_air_kg_m2
    )
