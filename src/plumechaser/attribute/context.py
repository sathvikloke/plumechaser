"""Infrastructure-context attribution with the honesty density rule.

This module deliberately does NOT claim source attribution. It joins a
detection against user-supplied infrastructure layers (EPA FLIGHT, Global
Energy Monitor exports, OSM energy tags) and produces a *confidence-ranked
context list*. When facility density is too high for individual attribution
(> ``density_rule_facilities_per_5km`` within 5 km), the output states
"multiple co-located infrastructure; individual attribution not supported"
verbatim -- that sentence appearing in dossiers IS the credibility feature.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from plumechaser.geo import haversine_km

MULTI_SOURCE_SENTINCEL = (
    "multiple co-located infrastructure; individual attribution not supported"
)


@dataclass(frozen=True)
class ContextResult:
    verdict: str                       # "single_candidate" | MULTI_SOURCE | "no_infrastructure"
    candidates: pd.DataFrame           # ranked: name, type, dist_km, wind_aligned


def _load_layer(path: str | None) -> pd.DataFrame:
    """Load an infrastructure CSV with columns name,type,lat,lon (empty ok)."""
    if not path:
        return pd.DataFrame(columns=["name", "type", "lat", "lon"])
    df = pd.read_csv(path)
    missing = {"name", "type", "lat", "lon"} - set(df.columns)
    if missing:
        raise ValueError(f"infrastructure layer {path} missing columns: {missing}")
    return df


def infrastructure_context(
    det_lon: float,
    det_lat: float,
    *,
    flight_csv: str | None = None,
    gem_csv: str | None = None,
    osm_csv: str | None = None,
    search_radius_km: float = 5.0,
    density_rule_per_radius: int = 5,
) -> ContextResult:
    """Rank nearby facilities and apply the frozen-plan density rule."""
    frames = [
        _load_layer(p).assign(layer=lbl)
        for lbl, p in (("flight", flight_csv), ("gem", gem_csv), ("osm", osm_csv))
    ]
    infra = pd.concat(frames, ignore_index=True)
    if infra.empty:
        return ContextResult(verdict="no_infrastructure", candidates=infra)

    infra["dist_km"] = [
        haversine_km(det_lon, det_lat, r.lon, r.lat) for r in infra.itertuples()
    ]
    near = infra[infra["dist_km"] <= search_radius_km].sort_values("dist_km")
    if len(near) > density_rule_per_radius:
        return ContextResult(verdict=MULTI_SOURCE_SENTINCEL, candidates=near)
    if near.empty:
        return ContextResult(verdict="no_infrastructure", candidates=near)
    return ContextResult(verdict="single_candidate", candidates=near)
