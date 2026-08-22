"""Persistence verification: kill single-pass transients and artifacts.

A raw TROPOMI-tier candidate is promoted only if a compatible detection
appears again within ``gap_days`` of its first appearance. This mirrors the
persistence gate adopted in the frozen analysis plan and removes most
orbit-striping and coastline artifacts, which do not recur coherently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from plumechaser.geo import haversine_km


@dataclass
class Candidate:
    candidate_id: str
    det_date: date
    lon: float
    lat: float
    score: float
    source_pass_ids: list[str] = field(default_factory=list)

    def merge(self, other: Candidate) -> Candidate:
        """Merge another detection of the same underlying source into this one."""
        self.score = max(self.score, other.score)
        if other.det_date < self.det_date:
            self.det_date = other.det_date
            self.lon, self.lat = other.lon, other.lat
        self.source_pass_ids.extend(other.source_pass_ids)
        return self


def persist_candidates(
    detections: list[Candidate],
    min_passes: int = 2,
    gap_days: int = 3,
    match_radius_km: float = 25.0,
) -> tuple[list[Candidate], list[Candidate]]:
    """Cluster detections in space/time; keep clusters spanning >= min_passes days.

    Returns ``(confirmed, rejected)``. Cluster representative keeps the
    earliest date/location and the maximum score.
    """
    if min_passes < 1:
        raise ValueError("min_passes must be >= 1")
    remaining = sorted(detections, key=lambda d: (d.det_date, -d.score))
    clusters: list[list[Candidate]] = []
    for det in remaining:
        placed = False
        for cluster in clusters:
            anchor = cluster[0]
            if (
                haversine_km(det.lon, det.lat, anchor.lon, anchor.lat) <= match_radius_km
                and timedelta(days=gap_days) >= (det.det_date - anchor.det_date)
                and timedelta(days=0) <= (det.det_date - anchor.det_date)
            ):
                cluster.append(det)
                placed = True
                break
        if not placed:
            clusters.append([det])

    confirmed: list[Candidate] = []
    rejected: list[Candidate] = []
    for cluster in clusters:
        distinct_dates = {c.det_date for c in cluster}
        rep = cluster[0]
        for c in cluster[1:]:
            rep.merge(c)
        rep.source_pass_ids = sorted({*rep.source_pass_ids})
        if len(distinct_dates) >= min_passes:
            confirmed.append(rep)
        else:
            rejected.append(rep)
    return confirmed, rejected
