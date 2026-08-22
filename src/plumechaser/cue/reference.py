"""Reference-pass selection for MBMP retrieval (frozen plan section B7).

Given a target date and candidate Sentinel-2 passes, select the cleanest
comparison pass:

    reject if  cloud_fraction            > max_cloud_fraction
    reject if  corr(B12, 12-mo median)   < min_surface_corr
    reject if  |MBSP anomaly sigma|      > max_reference_mbsp_sigma
               (the reference must not contain its own plume)
    score   = w_cloud*(1-cloud) + w_corr*corr + w_proximity*proximity
    proximity = 1 - |dt| / window_half_width

Returns the best scorer plus a logged manifest of all candidates and their
verdicts; ``None`` when every pass is rejected -> event stays DETECTION-ONLY.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidatePass:
    pass_id: str
    dt_days: int                      # signed distance from target date
    cloud_fraction: float
    surface_corr: float               # corr(B12, 12-month median B12) over ROI
    mbsp_abs_sigma: float             # strongest MBSP anomaly on the pass, in sigma


@dataclass(frozen=True)
class SelectionVerdict:
    chosen: str | None
    score: float | None
    rejected: dict[str, str]          # pass_id -> reason
    scores: dict[str, float]


def select_reference_pass(
    candidates: list[CandidatePass],
    *,
    margin_days: int = 2,
    max_window_days: int = 12,
    max_cloud_fraction: float = 0.05,
    min_surface_corr: float = 0.95,
    max_reference_mbsp_sigma: float = 3.0,
    w_cloud: float = 0.5,
    w_corr: float = 0.3,
    w_proximity: float = 0.2,
) -> SelectionVerdict:
    """Apply the frozen rejection rules then rank survivors by weighted score."""
    rejected: dict[str, str] = {}
    scores: dict[str, float] = {}
    half_window = float(max(max_window_days - margin_days, 1))

    for c in candidates:
        if abs(c.dt_days) < margin_days:
            rejected[c.pass_id] = f"inside {margin_days}d exclusion zone"
            continue
        if abs(c.dt_days) > max_window_days:
            rejected[c.pass_id] = "outside +/-window"
            continue
        if c.cloud_fraction > max_cloud_fraction:
            rejected[c.pass_id] = f"cloud {c.cloud_fraction:.2f} > {max_cloud_fraction:.2f}"
            continue
        if c.surface_corr < min_surface_corr:
            rejected[c.pass_id] = f"surface corr {c.surface_corr:.3f} < {min_surface_corr}"
            continue
        if c.mbsp_abs_sigma > max_reference_mbsp_sigma:
            rejected[c.pass_id] = f"own plume {c.mbsp_abs_sigma:.1f}sigma"
            continue
        proximity = 1.0 - (abs(c.dt_days) / half_window)
        score = (
            w_cloud * (1.0 - c.cloud_fraction)
            + w_corr * c.surface_corr
            + w_proximity * max(proximity, 0.0)
        )
        scores[c.pass_id] = round(score, 5)

    if not scores:
        return SelectionVerdict(chosen=None, score=None, rejected=rejected, scores=scores)

    best = max(scores.items(), key=lambda kv: kv[1])
    return SelectionVerdict(chosen=best[0], score=best[1], rejected=rejected, scores=scores)
