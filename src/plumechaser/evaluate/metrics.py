"""Metrics: precision/recall, ROC sweep, and cluster-bootstrap CIs.

Statistical discipline (frozen plan sections 4.1-4.5):
  * Detections of the SAME underlying source are correlated; naive bootstrap
    over detections understates uncertainty. We resample *source clusters*
    instead (block bootstrap), for precision over detection-clusters and for
    recall over reference-clusters.
  * All reported CIs are percentile intervals over ``bootstrap_draws``
    (default 2000) with a fixed seed recorded in the manifest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def precision_recall(
    matches: pd.DataFrame,
    n_detections: int,
    n_reference: int,
) -> dict[str, float | int | None]:
    """Point estimates from a :func:`match_events` result.

    Precision = matched / detected, Recall = matched / reference events.
    Returns None values when denominators are zero (never fabricate 0.0).
    """
    n_matched = len(matches)
    precision = n_matched / n_detections if n_detections else None
    recall = n_matched / n_reference if n_reference else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    return {
        "n_matched": n_matched,
        "n_detected": n_detections,
        "n_reference": n_reference,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def roc_points(
    scores: np.ndarray,
    is_true: np.ndarray,
) -> pd.DataFrame:
    """ROC curve by sweeping the score threshold from high to low.

    ``scores`` are candidate confidences; ``is_true`` flags candidates that
    match any reference event within tolerance. Ties are handled by grouping
    equal scores at one threshold step.
    """
    order = np.argsort(-np.asarray(scores, dtype=float))
    s_sorted = np.asarray(scores)[order]
    y_sorted = np.asarray(is_true, dtype=bool)[order]
    total_pos = int(y_sorted.sum())
    total_neg = int((~y_sorted).sum())
    rows: list[tuple[float, float, float]] = []
    tp = fp = 0
    i = 0
    while i < len(s_sorted):
        thr = s_sorted[i]
        j = i
        while j < len(s_sorted) and s_sorted[j] == thr:
            tp += int(y_sorted[j])
            fp += int(not y_sorted[j])
            j += 1
        tpr = tp / total_pos if total_pos else float("nan")
        fpr = fp / total_neg if total_neg else float("nan")
        rows.append((float(thr), tpr, fpr))
        i = j
    return pd.DataFrame(rows, columns=["threshold", "tpr", "fpr"])


def _percentile_ci(draws: np.ndarray, ci: tuple[float, float]) -> tuple[float, float]:
    lo, hi = np.percentile(draws, ci)
    return float(lo), float(hi)


def bootstrap_precision(
    det_cluster_ids: np.ndarray,
    tp_flags: np.ndarray,
    draws: int = 2000,
    seed: int = 0,
    ci: tuple[float, float] = (2.5, 97.5),
) -> tuple[float, float]:
    """Percentile CI on precision via block bootstrap over detection clusters."""
    clusters = np.unique(det_cluster_ids)
    rng = np.random.default_rng(seed)
    stats = np.empty(draws)
    cid_index = {c: np.where(det_cluster_ids == c)[0] for c in clusters}
    for d in range(draws):
        sample = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([cid_index[c] for c in sample])
        denom = len(idx)
        stats[d] = tp_flags[idx].sum() / denom if denom else 0.0
    return _percentile_ci(stats, ci)


def bootstrap_recall(
    ref_cluster_ids: np.ndarray,
    matched_flags: np.ndarray,
    draws: int = 2000,
    seed: int = 0,
    ci: tuple[float, float] = (2.5, 97.5),
) -> tuple[float, float]:
    """Percentile CI on recall via block bootstrap over reference clusters."""
    return bootstrap_precision(ref_cluster_ids, matched_flags, draws, seed, ci)
