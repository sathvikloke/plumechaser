"""Robust background climatology and anomaly scoring for TROPOMI CH4 grids.

Method
------
For each date t and pixel (i, j), the background is the per-pixel rolling
median over a symmetric window of ``window_days`` centred on t, *excluding*
day t itself so a strong event cannot inflate its own baseline. Anomalies are
scored with the modified z-score (Iglewicz & Hoaglin 1993):

    z = 0.6745 * (x - median) / MAD

which is robust to the heavy-tailed retrieval noise that breaks plain
z-scores. Where MAD == 0 (e.g. fully masked windows) the pixel is invalidated.
"""

from __future__ import annotations

import numpy as np

MODIFIED_Z_SCALE = 0.6745  # 0.75th standard normal quantile


def rolling_background(
    stack: np.ndarray,
    window_days: int = 30,
    exclude_center: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel rolling median and MAD along axis 0 of a (T, H, W) stack.

    NaNs are ignored. The output arrays have the same shape as ``stack``;
    entry [t] summarises the neighbourhood of day t. With ``exclude_center``
    the centre day is dropped from its own window.
    """
    if stack.ndim != 3:
        raise ValueError("expected a 3-D (time, rows, cols) stack")
    if window_days < 1:
        raise ValueError("window_days must be >= 1")

    T, H, W = stack.shape
    half = max(1, window_days // 2)
    med = np.full((T, H, W), np.nan, dtype=np.float64)
    mad = np.full((T, H, W), np.nan, dtype=np.float64)

    for t in range(T):
        lo, hi = max(0, t - half), min(T, t + half + 1)
        idx = list(range(lo, hi))
        if exclude_center and len(idx) > 1:
            idx.remove(t)
        sample = stack[idx]  # (n, H, W)
        m = np.nanmedian(sample, axis=0)
        med[t] = m
        mad[t] = np.nanmedian(np.abs(sample - m[None]), axis=0)
    return med, mad


def robust_zscores(
    values: np.ndarray,
    median: np.ndarray,
    mad: np.ndarray,
    floor_mad_ppb: float = 2.0,
) -> np.ndarray:
    """Modified z-scores; pixels with tiny/NaN MAD become NaN.

    ``floor_mad_ppb`` guards against degenerate windows where every sample is
    identical (MAD == 0), which would explode the ratio.
    """
    safe_mad = np.where(np.isfinite(mad) & (mad >= floor_mad_ppb), mad, np.nan)
    with np.errstate(invalid="ignore"):
        return MODIFIED_Z_SCALE * (values - median) / safe_mad


def threshold_mask(z: np.ndarray, z_threshold: float) -> np.ndarray:
    """Boolean mask of pixels whose modified z-score clears ``z_threshold``."""
    return np.isfinite(z) & (z >= z_threshold)
