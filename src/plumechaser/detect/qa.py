"""QA gates applied before any detection logic runs."""

from __future__ import annotations

import numpy as np


def qa_mask(
    qa_grid: np.ndarray,
    min_qa: float = 0.5,
    value_grid: np.ndarray | None = None,
    cloud_fraction: float | None = None,
    max_cloud_fraction: float = 0.3,
    extra_masks: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Combine quality gates into one boolean valid-pixel mask.

    Gates:
      * ``qa_grid >= min_qa``            -- TROPOMI QA value filter
      * finite values in ``value_grid``  -- kills fill values
      * ``cloud_fraction <= max_cloud_fraction`` when provided
      * any number of pre-computed boolean masks (albedo bounds, coastlines...)
    """
    mask = np.isfinite(qa_grid) & (qa_grid >= min_qa)
    if value_grid is not None:
        mask &= np.isfinite(value_grid)
    if cloud_fraction is not None:
        mask &= cloud_fraction <= max_cloud_fraction
    for extra in extra_masks or []:
        if extra.shape != mask.shape:
            raise ValueError("extra mask shape mismatch")
        mask &= extra.astype(bool)
    return mask
