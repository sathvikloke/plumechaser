"""Spatiotemporal event matching between our detections and a reference catalog.

Matching semantics are fixed by the frozen analysis plan (section 4.3):
a detection matches a reference event when within ``match_radius_km``
(default 25 km -- TROPOMI localization uncertainty) AND ``match_window_days``
(default +/-3 d). Greedy assignment processes reference events in date order
and takes the nearest unused detection; this is deterministic and adequate at
the sparse event densities involved (documented alternative: Hungarian).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from plumechaser.geo import haversine_km


@dataclass(frozen=True)
class MatchResult:
    matches: pd.DataFrame          # columns: ref_id, det_id, dist_km, dt_days
    unmatched_ref: list[str]
    unmatched_det: list[str]


def match_events(
    detections: pd.DataFrame,   # id, date, lon, lat [, score]
    reference: pd.DataFrame,    # id, date, lon, lat
    radius_km: float = 25.0,
    window_days: int = 3,
) -> MatchResult:
    required_det = {"id", "date", "lon", "lat"}
    required_ref = {"id", "date", "lon", "lat"}
    if not required_det <= set(detections.columns):
        raise ValueError(f"detections missing columns: {required_det - set(detections.columns)}")
    if not required_ref <= set(reference.columns):
        raise ValueError(f"reference missing columns: {required_ref - set(reference.columns)}")

    dets = detections.copy()
    refs = reference.copy()
    dets["date"] = pd.to_datetime(dets["date"]).dt.date
    refs["date"] = pd.to_datetime(refs["date"]).dt.date

    pairs: list[dict] = []
    used_dets: set[str] = set()
    for ref in refs.sort_values("date").itertuples():
        best = None
        for det in dets.itertuples():
            if det.id in used_dets or abs((det.date - ref.date).days) > window_days:
                continue
            dist = haversine_km(ref.lon, ref.lat, det.lon, det.lat)
            if dist > radius_km:
                continue
            key = (abs((det.date - ref.date).days), dist)
            if best is None or key < best[0]:
                best = (key, det, dist)
        if best is not None:
            _, det, dist = best
            used_dets.add(det.id)
            pairs.append(
                {
                    "ref_id": ref.id,
                    "det_id": det.id,
                    "dist_km": round(dist, 3),
                    "dt_days": abs((det.date - ref.date).days),
                }
            )
    matches = pd.DataFrame(pairs, columns=["ref_id", "det_id", "dist_km", "dt_days"])
    return MatchResult(
        matches=matches,
        unmatched_ref=[str(r) for r in refs["id"] if r not in set(matches["ref_id"])]
        if len(matches)
        else [str(r) for r in refs["id"]],
        unmatched_det=[str(d) for d in dets["id"] if d not in set(matches["det_id"])],
    )
