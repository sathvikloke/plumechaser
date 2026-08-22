"""Agreement-study orchestrator: matching + metrics + sensitivity + branches.

Implements the frozen analysis plan end to end:

  * primary endpoint = transient-tier precision/recall on champion basins
  * dual-mode references (strict operational list vs lenient full set) with
    the mandated interpretation sentence attached to every report
  * cluster-parameter sensitivity sweep (distance x window, +/-50%)
  * power-branch assignment by rule on the central combo (n_eff >= 30 ->
    'full'; >= 15 -> 'intermediate'; else 'descriptive'); ALL branches are
    reported regardless of which is primary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from plumechaser.evaluate.matching import match_events
from plumechaser.evaluate.metrics import (
    bootstrap_precision,
    bootstrap_recall,
    precision_recall,
)

DUAL_MODE_SENTENCE = (
    "Strict mode benchmarks against operational human curation; lenient mode "
    "estimates the algorithmic ceiling; the strict-lenient gap is itself our "
    "quantification of curation effect."
)

BRANCH_RULES = {"full": 30, "intermediate": 15}


@dataclass(frozen=True)
class AgreementReport:
    mode: str
    branch: str
    n_reference_clusters: int
    metrics: dict
    precision_ci: tuple[float, float] | None
    recall_ci: tuple[float, float] | None
    sensitivity: pd.DataFrame
    interpretation: str


def cluster_ids(events: pd.DataFrame, distance_km: float, window_days: int) -> np.ndarray:
    """Greedy spatiotemporal clustering; returns a cluster id per event row.

    Events within ``distance_km`` of a cluster's anchor AND within
    ``window_days`` after it join that cluster; otherwise they seed a new one.
    Deterministic (date-then-lon sort order); ids are returned in the ORIGINAL
    row order of ``events`` so callers can align them with other per-row arrays.
    """
    from plumechaser.geo import haversine_km

    ids = np.full(len(events), -1, dtype=int)
    anchors: list[tuple[float, float, object]] = []  # lon, lat, date
    next_id = 0
    for orig_idx in events.sort_values(["date", "lon", "lat"]).index:
        r = events.loc[orig_idx]
        placed = False
        for cid, (alon, alat, adate) in enumerate(anchors):
            if haversine_km(r.lon, r.lat, alon, alat) <= distance_km and (
                0 <= (r.date - adate).days <= window_days
            ):
                ids[orig_idx] = cid
                placed = True
                break
        if not placed:
            ids[orig_idx] = next_id
            anchors.append((r.lon, r.lat, r.date))
            next_id += 1
    return ids


def _run_one_mode(
    detections: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    radius_km: float,
    window_days: int,
    cluster_distance_km: float,
    cluster_window_days: int,
    bootstrap_draws: int,
    seed: int,
) -> dict:
    mr = match_events(detections, reference, radius_km=radius_km, window_days=window_days)
    pr = precision_recall(mr.matches, len(detections), len(reference))

    ref_clusters = cluster_ids(reference, cluster_distance_km, cluster_window_days)
    det_clusters = cluster_ids(detections, cluster_distance_km, cluster_window_days)
    matched_ref_ids = set(mr.matches["ref_id"]) if len(mr.matches) else set()
    ref_matched_flags = reference["id"].isin(matched_ref_ids).to_numpy()
    det_matched_flags = detections["id"].isin(set(mr.matches["det_id"])).to_numpy()

    p_ci = (
        bootstrap_precision(det_clusters, det_matched_flags, draws=bootstrap_draws, seed=seed)
        if len(detections)
        else None
    )
    r_ci = (
        bootstrap_recall(ref_clusters, ref_matched_flags, draws=bootstrap_draws, seed=seed)
        if len(reference)
        else None
    )
    return {
        "metrics": pr,
        "precision_ci": p_ci,
        "recall_ci": r_ci,
        "n_ref_clusters": int(len(np.unique(ref_clusters))),
        "n_det_clusters": int(len(np.unique(det_clusters))),
    }


def assign_branch(n_effective: int) -> str:
    """Power-branch rule on effective (clustered) sample size."""
    if n_effective >= BRANCH_RULES["full"]:
        return "full"
    if n_effective >= BRANCH_RULES["intermediate"]:
        return "intermediate"
    return "descriptive"


def run_agreement(
    detections: pd.DataFrame,
    strict_reference: pd.DataFrame,
    lenient_reference: pd.DataFrame | None,
    *,
    radius_km: float = 25.0,
    window_days: int = 3,
    cluster_distance_km: float = 25.0,
    cluster_window_days: int = 14,
    bootstrap_draws: int = 2000,
    seed: int = 0,
) -> dict[str, AgreementReport]:
    """Full agreement study for strict (+ optional lenient) reference modes."""
    reports: dict[str, AgreementReport] = {}
    modes: list[tuple[str, pd.DataFrame]] = [("strict", strict_reference)]
    if lenient_reference is not None and len(lenient_reference):
        modes.append(("lenient", lenient_reference))

    for mode, ref in modes:
        central = _run_one_mode(
            detections,
            ref,
            radius_km=radius_km,
            window_days=window_days,
            cluster_distance_km=cluster_distance_km,
            cluster_window_days=cluster_window_days,
            bootstrap_draws=bootstrap_draws,
            seed=seed,
        )
        # Sensitivity sweep: +/-50% on both cluster parameters.
        rows = []
        for cd in (0.5, 1.0, 1.5):
            for cw in (7, 14, 21):
                sweep = _run_one_mode(
                    detections,
                    ref,
                    radius_km=radius_km,
                    window_days=window_days,
                    cluster_distance_km=cluster_distance_km * cd,
                    cluster_window_days=max(int(cluster_window_days * cw / 14), 1),
                    bootstrap_draws=max(bootstrap_draws // 4, 200),
                    seed=seed,
                )
                rows.append(
                    {
                        "cluster_dist_factor": cd,
                        "cluster_window_days": max(int(cluster_window_days * cw / 14), 1),
                        "n_ref_clusters": sweep["n_ref_clusters"],
                        "precision": sweep["metrics"]["precision"],
                        "recall": sweep["metrics"]["recall"],
                    }
                )
        sensitivity = pd.DataFrame(rows)
        flipped = int((sensitivity["n_ref_clusters"] >= BRANCH_RULES["full"]).sum())
        branch = assign_branch(central["n_ref_clusters"])
        note = ""
        if flipped >= 3 or flipped <= 6:
            note = f" (robustness: {flipped}/9 combos land in 'full' branch)"
        reports[mode] = AgreementReport(
            mode=mode,
            branch=branch + note,
            n_reference_clusters=central["n_ref_clusters"],
            metrics=central["metrics"],
            precision_ci=central["precision_ci"],
            recall_ci=central["recall_ci"],
            sensitivity=sensitivity,
            interpretation=DUAL_MODE_SENTENCE,
        )
    return reports
