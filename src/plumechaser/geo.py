"""Shared geospatial helpers (no GDAL / rasterio dependency)."""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance between two WGS84 points, in kilometres."""
    phi1, lam1, phi2, lam2 = map(radians, (lat1, lon1, lat2, lon2))
    dphi, dlam = phi2 - phi1, lam2 - lam1
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(a))


def in_bbox(lon: float, lat: float, bbox: tuple[float, float, float, float]) -> bool:
    lon_min, lat_min, lon_max, lat_max = bbox
    return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max


@dataclass(frozen=True)
class GridTransform:
    """Affine mapping between pixel indices and lon/lat for a regular grid."""

    lon0: float  # longitude of pixel (0, 0) centre
    lat0: float  # latitude of pixel (0, 0) centre
    dlon: float  # degrees per column step
    dlat: float  # degrees per row step (negative for north-up rasters)

    def to_lonlat(self, row: int | float, col: int | float) -> tuple[float, float]:
        return self.lon0 + col * self.dlon, self.lat0 + row * self.dlat

    def from_lonlat(self, lon: float, lat: float) -> tuple[int, int]:
        return round((lat - self.lat0) / self.dlat), round((lon - self.lon0) / self.dlon)


def km_per_degree(lat: float) -> tuple[float, float]:
    """Approximate (km per degree lon, km per degree lat) at a latitude."""
    import math

    return EARTH_RADIUS_KM * math.cos(radians(lat)) * math.pi / 180.0, (
        EARTH_RADIUS_KM * math.pi / 180.0
    )
