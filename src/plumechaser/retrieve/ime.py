"""Integrated Mass Enhancement (IME) emission-rate quantification.

Science basis
-------------
Frankenberg et al. (2016); Varon et al. (2018, 2021). The observed plume mass
resides downwind of the source for a residence time ~L/U_eff, giving

    Q [kg/h] = IME [kg] * U_eff [m/s] * 3600 / L [m]

with the Sentinel-2-calibrated effective wind speed (Varon et al. 2021):

    U_eff = 0.33 * U10 + 0.45   [m/s]

Uncertainty is propagated by Monte Carlo over three independent terms
(frozen analysis plan section 4.6): wind speed, plume-mask membership, and
retrieval noise. Percentile credible intervals are reported; the point
estimate always comes from the unperturbed inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImeResult:
    q_kg_h: float
    ci_low: float
    ci_high: float
    ime_kg: float
    ueff_ms: float
    length_m: float
    n_pixels: int


def effective_wind_speed(u10_ms: float, slope: float = 0.33, intercept: float = 0.45) -> float:
    """Varon et al. (2021) LES calibration mapping 10 m wind to effective wind."""
    if u10_ms < 0:
        raise ValueError("wind speed must be non-negative")
    return slope * u10_ms + intercept


def plume_length_m(n_pixels: int, pixel_area_m2: float) -> float:
    """Characteristic plume scale L = sqrt(plume area)."""
    if n_pixels < 1 or pixel_area_m2 <= 0:
        raise ValueError("need >=1 pixel and positive pixel area")
    return float(np.sqrt(n_pixels * pixel_area_m2))


def integrated_mass_kg(
    d_xch4_ppb: np.ndarray,
    mask: np.ndarray,
    pixel_area_m2: float,
    surface_pressure_hpa: float = 1013.0,
) -> float:
    """Total excess methane mass inside the plume mask, in kg."""
    from plumechaser.retrieve.mbmp import column_mass_kg_m2

    if mask.shape != d_xch4_ppb.shape:
        raise ValueError("mask/value shape mismatch")
    mass_per_px = column_mass_kg_m2(d_xch4_ppb, surface_pressure_hpa)
    vals = np.asarray(mass_per_px)[mask]
    vals = vals[np.isfinite(vals)]
    return float(vals.sum()) * pixel_area_m2


def source_rate_q(ime_kg: float, ueff_ms: float, length_m: float) -> float:
    """Q = IME * Ueff * 3600 / L  [kg/h]."""
    if length_m <= 0:
        raise ValueError("plume length must be positive")
    return ime_kg * ueff_ms * 3600.0 / length_m


def quantitate(
    d_xch4_ppb: np.ndarray,
    mask: np.ndarray,
    u10_ms: float,
    pixel_area_m2: float,
    *,
    ueff_slope: float = 0.33,
    ueff_intercept: float = 0.45,
    mc_samples: int = 500,
    wind_noise_frac: float = 0.25,
    mask_inclusion_prob: float = 0.9,
    retrieval_noise_ppb: float = 15.0,
    ci_percentiles: tuple[float, float] = (2.5, 97.5),
    surface_pressure_hpa: float = 1013.0,
    seed: int = 0,
) -> ImeResult:
    """Point estimate + Monte Carlo CI for a single-plume emission rate.

    Parameters mirror config ``ime:`` block; see module docstring for the
    three perturbation terms.
    """
    ime_point = integrated_mass_kg(d_xch4_ppb, mask, pixel_area_m2, surface_pressure_hpa)
    n_pixels = int(mask.sum())
    length = plume_length_m(max(n_pixels, 1), pixel_area_m2)
    ueff = effective_wind_speed(u10_ms, ueff_slope, ueff_intercept)
    q_point = source_rate_q(ime_point, ueff, length)

    rng = np.random.default_rng(seed)
    draws = np.empty(mc_samples, dtype=np.float64)
    for i in range(mc_samples):
        # 1) wind uncertainty (multiplicative, lognormal-ish symmetric clip)
        w_factor = 1.0 + rng.normal(0.0, wind_noise_frac)
        w_factor = float(np.clip(w_factor, 1.0 - 2 * wind_noise_frac, 1.0 + 2 * wind_noise_frac))
        # 2) plume-mask membership: Bernoulli keep per masked pixel
        keep = rng.random(mask.shape) < mask_inclusion_prob
        pert_mask = mask & keep
        if not pert_mask.any():
            draws[i] = 0.0
            continue
        # 3) retrieval noise added to the enhancement field
        noisy = d_xch4_ppb + rng.normal(0.0, retrieval_noise_ppb, size=d_xch4_ppb.shape)
        ime_i = integrated_mass_kg(noisy, pert_mask, pixel_area_m2, surface_pressure_hpa)
        n_i = int(pert_mask.sum())
        len_i = plume_length_m(max(n_i, 1), pixel_area_m2)
        draws[i] = source_rate_q(ime_i, ueff * w_factor, len_i)

    lo, hi = np.percentile(draws, ci_percentiles)
    return ImeResult(
        q_kg_h=q_point,
        ci_low=float(lo),
        ci_high=float(hi),
        ime_kg=ime_point,
        ueff_ms=ueff,
        length_m=length,
        n_pixels=n_pixels,
    )
