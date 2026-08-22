"""Blob extraction: connected components above an anomaly threshold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from plumechaser.geo import GridTransform


@dataclass(frozen=True)
class Blob:
    label: int
    n_pixels: int
    peak_value: float
    mean_value: float
    row: float  # intensity-weighted centroid
    col: float
    bbox: tuple[slice, slice]


def extract_blobs(
    mask: np.ndarray,
    value_grid: np.ndarray,
    min_pixels: int = 3,
    transform: GridTransform | None = None,
) -> list[Blob]:
    """Label connected True regions in ``mask`` and summarise each blob.

    ``value_grid`` supplies intensities (e.g. robust z-scores) used for the
    reported peak/mean/centroid. Blobs smaller than ``min_pixels`` are dropped.
    If ``transform`` is given, blobs additionally carry lon/lat centroids
    accessible via :func:`blob_centroid_lonlat`.
    """
    if mask.shape != value_grid.shape:
        raise ValueError("mask and value_grid shapes differ")
    labels, n = ndimage.label(mask)
    blobs: list[Blob] = []
    for lab in range(1, n + 1):
        sel = labels == lab
        n_pix = int(sel.sum())
        if n_pix < min_pixels:
            continue
        rows, cols = np.nonzero(sel)
        weights = np.clip(value_grid[sel], 0, None)
        wsum = weights.sum()
        cy = float((rows * weights).sum() / wsum) if wsum > 0 else float(rows.mean())
        cx = float((cols * weights).sum() / wsum) if wsum > 0 else float(cols.mean())
        rs = slice(int(rows.min()), int(rows.max()) + 1)
        cs = slice(int(cols.min()), int(cols.max()) + 1)
        blobs.append(
            Blob(
                label=lab,
                n_pixels=n_pix,
                peak_value=float(value_grid[sel].max()),
                mean_value=float(value_grid[sel].mean()),
                row=cy,
                col=cx,
                bbox=(rs, cs),
            )
        )
    blobs.sort(key=lambda b: -b.peak_value)
    del transform  # kept in signature for API symmetry; see blob_centroid_lonlat
    return blobs


def blob_centroid_lonlat(blob: Blob, transform: GridTransform) -> tuple[float, float]:
    """Lon/lat of a blob centroid under a grid transform."""
    return transform.to_lonlat(blob.row, blob.col)
