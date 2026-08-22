"""Matching + metrics + bootstrap tests."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from plumechaser.evaluate.matching import match_events
from plumechaser.evaluate.metrics import (
    bootstrap_precision,
    bootstrap_recall,
    precision_recall,
    roc_points,
)


def _df(rows):
    return pd.DataFrame(rows, columns=["id", "date", "lon", "lat"])


def test_match_within_tolerance():
    dets = _df([("d1", date(2025, 3, 2), 59.0, 39.0)])
    refs = _df([("r1", date(2025, 3, 3), 59.1, 39.05)])
    mr = match_events(dets, refs, radius_km=25.0, window_days=3)
    assert list(mr.matches["det_id"]) == ["d1"]
    assert mr.unmatched_det == []


def test_time_window_enforced():
    dets = _df([("d1", date(2025, 3, 10), 59.0, 39.0)])
    refs = _df([("r1", date(2025, 3, 2), 59.0, 39.0)])
    mr = match_events(dets, refs, radius_km=25.0, window_days=3)
    assert mr.matches.empty
    assert mr.unmatched_ref == ["r1"] and mr.unmatched_det == ["d1"]


def test_radius_enforced_and_greedy_uniqueness():
    # Two references near one detection: only the closer-in-time one wins;
    # second detection is free to match the other reference.
    dets = _df(
        [
            ("d1", date(2025, 5, 1), 59.00, 39.00),
            ("d2", date(2025, 5, 4), 59.30, 39.20),
        ]
    )
    refs = _df(
        [
            ("r1", date(2025, 5, 1), 59.02, 39.01),
            ("r2", date(2025, 5, 4), 59.28, 39.18),
        ]
    )
    mr = match_events(dets, refs, radius_km=25.0, window_days=3)
    assert len(mr.matches) == 2
    assert set(mr.matches["det_id"]) == {"d1", "d2"}


def test_precision_recall_including_zero_denominators():
    matches = pd.DataFrame([{"det_id": "d", "ref_id": "r"}])
    pr = precision_recall(matches, n_detections=4, n_reference=8)
    assert pr["precision"] == 0.25 and pr["recall"] == 0.125 and pr["f1"]
    empty = precision_recall(pd.DataFrame(columns=["a"]), n_detections=0, n_reference=5)
    assert empty["precision"] is None and empty["recall"] == 0.0


def test_roc_points_monotone():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    truth = np.array([True, True, False, False], dtype=bool)
    roc = roc_points(scores, truth)
    assert (roc["tpr"].diff().dropna() >= 0).all()
    assert (roc["fpr"].diff().dropna() >= 0).all()
    assert roc["tpr"].iloc[-1] == 1.0 and roc["fpr"].iloc[-1] == 1.0


def test_bootstrap_ci_contains_estimate_and_shrinks():
    rng = np.random.default_rng(0)
    # 200 detections in 40 clusters; 60% TP rate.
    det_clusters = np.repeat(np.arange(40), 5)
    tp = (rng.random(200) < 0.6).astype(float)
    lo, hi = bootstrap_precision(det_clusters, tp, draws=800, seed=1)
    point = tp.mean()
    assert lo <= point <= hi
    wide = hi - lo
    big_clusters = np.repeat(np.arange(400), 1)   # many small clusters -> tighter CI
    tp_big = np.tile(tp[:100], 4)                  # same rate
    lo2, hi2 = bootstrap_precision(big_clusters, tp_big.astype(float), draws=800, seed=1)
    assert (hi2 - lo2) < wide


def test_bootstrap_recall_symmetry():
    ref_clusters = np.arange(50) // 2
    matched = (np.random.default_rng(2).random(50) < 0.7).astype(float)
    lo, hi = bootstrap_recall(ref_clusters, matched, draws=500, seed=3)
    assert 0.0 <= lo <= matched.mean() <= hi <= 1.0
