"""Two-step inference cascade: sliding windows -> CNN -> SVC artifact filter.

Stage 1 (CNN, optional): plume-morphology probability on the normalized CH4
window. Stage 2 (SVC, optional): context-aware artifact rejection on the
full channel set. A window becomes a detection when every supplied stage
clears its threshold; with only the SVC supplied it runs standalone (useful
for tests and for re-scoring archived candidates).

Detections carry window-relative blob centroids so callers can convert to
lon/lat via their own grid transform.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from plumechaser.ml.dataset import normalize_scene
from plumechaser.ml.features import scene_features


@dataclass(frozen=True)
class Detection:
    row: float          # blob centroid within the FULL grid
    col: float
    score: float        # combined min(stage probabilities)
    cnn_prob: float | None
    svc_prob: float | None


def slide_windows(
    grid: np.ndarray,
    window: int = 32,
    stride: int = 16,
) -> list[tuple[int, int, np.ndarray]]:
    """Yield (row0, col0, patch) tiles; edge patches are dropped."""
    if grid.shape[0] < window or grid.shape[1] < window:
        return []
    out = []
    for r in range(0, grid.shape[0] - window + 1, stride):
        for c in range(0, grid.shape[1] - window + 1, stride):
            out.append((r, c, grid[r : r + window, c : c + window]))
    return out


def _cnn_forward(model, norm_window: np.ndarray) -> float:
    import torch

    with torch.no_grad():
        x = torch.from_numpy(norm_window[None, None].astype(np.float32))
        logits = model(x).squeeze().item()
    return 1.0 / (1.0 + np.exp(-logits))


def run_detector(
    xch4_grid: np.ndarray,
    *,
    cnn=None,
    svc_scorer: Callable[[np.ndarray], float] | None = None,
    cnn_threshold: float = 0.5,
    svc_threshold: float = 0.5,
    window: int = 32,
    stride: int = 16,
    context_fn: Callable[[int, int, np.ndarray], dict] | None = None,
) -> list[Detection]:
    """Score a TROPOMI-tier XCH4 grid; returns ranked detections.

    ``svc_scorer(feature_row) -> plume-probability`` decouples the cascade
    from sklearn specifics (use :func:`plumechaser.ml.svc.make_svc_scorer`).
    ``context_fn(r0, c0, patch)`` supplies extra channels for SVC features
    keyed as in :func:`scene_features`; it is REQUIRED whenever an SVC stage
    is active, because that model was trained WITH context channels and
    neutral zeros would push inputs out of distribution.
    """
    if svc_scorer is not None and context_fn is None:
        raise ValueError(
            "SVC stage requires context_fn: this model trained on wind/albedo/"
            "QA/chi2 channels; scoring without them is out-of-distribution."
        )
    detections: list[Detection] = []
    for r0, c0, patch in slide_windows(xch4_grid, window, stride):
        finite = np.isfinite(patch)
        if finite.sum() < 64:
            continue
        norm = normalize_scene(patch)

        cnn_prob: float | None = None
        if cnn is not None:
            cnn_prob = _cnn_forward(cnn, norm)
            if cnn_prob < cnn_threshold:
                continue

        svc_prob: float | None = None
        if svc_scorer is not None:
            ctx = context_fn(r0, c0, patch)
            feats = np.asarray([scene_features(patch, **ctx)], dtype=np.float64)
            if not np.isfinite(feats).all():
                feats = np.nan_to_num(feats, nan=0.0)
            svc_prob = float(svc_scorer(feats[0]))
            if svc_prob < svc_threshold:
                continue

        # Blob centroid inside this window (fallback: window center).
        high = (norm > 0.6) & finite
        if high.any():
            ys, xs = np.nonzero(high)
            row, col = float(ys.mean()) + r0, float(xs.mean()) + c0
        else:
            row, col = r0 + window / 2, c0 + window / 2

        stages = [p for p in (cnn_prob, svc_prob) if p is not None]
        detections.append(
            Detection(row=row, col=col, score=float(min(stages)),
                      cnn_prob=cnn_prob, svc_prob=svc_prob)
        )

    detections.sort(key=lambda d: -d.score)
    return detections
