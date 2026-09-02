"""Input-scale validation for satellite reflectance arrays.

Why this exists
---------------
On 2026-08-25 a campaign ran for a full day producing confident, publishable-
looking results from a model that had been handed 0-1 reflectance where it
required DN (reflectance x 10000). Nothing crashed and nothing looked wrong:
the model's band-ratio input channel is scale-invariant, so it kept producing
plausible output while every radiance channel was effectively zero. The error
surfaced only when the resulting fluxes were audited against catalog rates.

A scale error is uniquely dangerous in this pipeline because the two
conventions differ by exactly 10^4 and several downstream operations
(band ratios, normalised differences, log-ratios) are invariant to it. The
invariance is what suppresses the symptom, not what makes the result correct.

So: check the convention explicitly at every boundary where an array crosses
from our fetch code into someone else's model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ScaleVerdict", "infer_reflectance_scale", "assert_dn_scale"]

# Sentinel-2 L1C TOA reflectance in DN sits in the low thousands over land
# (0.02-0.6 reflectance -> 200-6000 DN). Reflectance-scaled data is <= ~1.5
# even for bright cloud. The gap between the conventions is 10^4, so a
# decisive threshold anywhere in between is safe.
REFLECTANCE_MAX_PLAUSIBLE = 1.5
DN_MIN_PLAUSIBLE = 20.0


@dataclass(frozen=True)
class ScaleVerdict:
    """What convention an array appears to be in."""

    scale: str  # "dn" | "reflectance" | "indeterminate"
    p99: float
    finite_fraction: float

    @property
    def is_dn(self) -> bool:
        return self.scale == "dn"


def infer_reflectance_scale(values: np.ndarray) -> ScaleVerdict:
    """Infer whether ``values`` are DN (reflectance x 10000) or 0-1 reflectance.

    Uses the 99th percentile rather than the max so a handful of saturated or
    corrupt pixels cannot flip the verdict.
    """
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    finite_fraction = finite.size / arr.size if arr.size else 0.0
    if finite.size == 0:
        return ScaleVerdict("indeterminate", float("nan"), 0.0)

    p99 = float(np.percentile(finite, 99))
    if p99 <= REFLECTANCE_MAX_PLAUSIBLE:
        scale = "reflectance"
    elif p99 >= DN_MIN_PLAUSIBLE:
        scale = "dn"
    else:
        # Between the two conventions: an all-dark window, or something wrong.
        scale = "indeterminate"
    return ScaleVerdict(scale, p99, finite_fraction)


def assert_dn_scale(values: np.ndarray, name: str = "image") -> ScaleVerdict:
    """Raise unless ``values`` are on the DN scale a model expects.

    Args:
        values: reflectance array of any shape.
        name: label used in the error message.

    Returns:
        The verdict, so callers can log the p99 they were handed.

    Raises:
        ValueError: if the array is 0-1 reflectance, or too dark to tell.
    """
    verdict = infer_reflectance_scale(values)
    if verdict.is_dn:
        return verdict
    if verdict.scale == "reflectance":
        raise ValueError(
            f"{name}: looks like 0-1 reflectance (p99={verdict.p99:.4f}) but DN "
            f"(reflectance x 10000) is required. Band ratios are invariant to "
            f"this error, so it will NOT surface downstream — multiply by 1e4."
        )
    raise ValueError(
        f"{name}: cannot determine reflectance scale (p99={verdict.p99:.4f}, "
        f"{verdict.finite_fraction:.1%} finite). Expected DN in the hundreds "
        f"to thousands; check nodata handling before proceeding."
    )
