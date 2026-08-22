"""Analytic detection-limit surfaces for the free-sensor observability atlas.

Frozen plan section D3 / Round-8 spec A3. For each basin x season x surface
class we derive the minimum detectable emission rate from quantities we
already compute:

    sigma_col   robust retrieval noise of the enhancement field [ppb]
                (empirical, from plume-free reference windows of OUR pipeline;
                 external anchor points from Gorrono et al. 2023 are plotted
                 separately -- the layer is labelled 'instrument-and-pipeline')
    k           ROC operating point in sigmas (config atlas.k_roc)
    N_min       minimum blob size [pixels]      (config tropomi.min_blob_pixels)
    IME_min     N_min * column_mass(k*sigma_col) * pixel_area
    Q_min       IME_min * Ueff * 3600 / L_typical(class)

The empirical detections are then plotted ON this surface; agreement between
the analytic layer and observed detections is the atlas headline figure.
"""

from __future__ import annotations

import math

import numpy as np

from plumechaser.retrieve.ime import effective_wind_speed
from plumechaser.retrieve.mbmp import column_mass_kg_m2


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
