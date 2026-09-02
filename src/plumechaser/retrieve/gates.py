"""Honesty gates — the withhold-quantification rule, in one tested place.

The frozen analysis plan (section 7) makes these gates load-bearing: a
retrieval whose background column noise or plume-mask footprint is too large
is declared ARTIFACT-DOMINATED and NO flux is claimed from it.

The gates live here, not in a campaign script, for three reasons:

1. Every retrieval path must be gated identically. That includes production
   third-party models (``marss2l``): a model being operational is not a
   reason to skip the gate, and the first production campaign showed exactly
   why (see docs/S2_REAL_DATA_FINDINGS.md).
2. The thresholds are science constants, so they belong in the config, not
   in a script constant.
3. Load-bearing logic must be unit-tested.

Both statistics are computed on the **background** of the enhancement field
(pixels outside the plume mask). Including plume pixels in the noise estimate
would let a strong plume inflate sigma and mask its own artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .flux_audit import BACKGROUND_CH4_PPB
from .mbmp import robust_scene_sigma

__all__ = ["GateVerdict", "evaluate_gates", "sigma_ppb_limit_for_scale"]


def sigma_ppb_limit_for_scale(
    c1_ppb_per_log_ratio: float,
    sigma_log_ratio_limit: float,
) -> float:
    """Convert the calibration-independent gate anchor into a ppb limit.

    The pre-registered gate is written as "sigma_col > 80 ppb", but ppb is not
    a calibration-independent unit: the 2026-08-25 audit measured our
    simplified coefficients as understating columns by 2.5-6.3x versus the
    production RTM, so the *same physical scene noise* produced two different
    verdicts depending on which chain measured it.

    What the gate actually encodes is a limit on fractional band-ratio noise.
    Anchoring there and converting per-retrieval keeps the pre-registered
    intent exactly while making the threshold comparable across chains:

        sigma_log_ratio_limit = 80 ppb * (alpha_b12 - alpha_b11)
                              = 80 * 9.0e-5 = 7.2e-3

    Evaluated on the simplified chain this returns 80 ppb by construction, so
    nothing about the frozen plan's operating point changes.
    """
    if c1_ppb_per_log_ratio <= 0 or sigma_log_ratio_limit <= 0:
        raise ValueError("calibration slope and limit must both be positive")
    return float(sigma_log_ratio_limit * c1_ppb_per_log_ratio)

ARTIFACT_VERDICT = "ARTIFACT-DOMINATED — quantification withheld"
CLEAN_VERDICT = "gates passed — quantification permitted"


@dataclass(frozen=True)
class GateVerdict:
    """Outcome of the honesty gates for a single retrieval."""

    sigma_col_ppb: float
    mask_fraction: float
    n_mask_px: int
    n_window_px: int  # valid pixels only, when a valid mask is supplied
    artifact_dominated: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
    mean_enhancement_ppb: float = float("nan")

    @property
    def verdict(self) -> str:
        return ARTIFACT_VERDICT if self.artifact_dominated else CLEAN_VERDICT

    def as_dict(self) -> dict[str, object]:
        """Flat form for provenance JSON."""
        return {
            "sigma_col_ppb": round(self.sigma_col_ppb, 2),
            "mask_fraction": round(self.mask_fraction, 4),
            "n_mask_px": self.n_mask_px,
            "n_window_px": self.n_window_px,
            "mean_enhancement_ppb": round(self.mean_enhancement_ppb, 1),
            "artifact_dominated": self.artifact_dominated,
            "gate_reasons": list(self.reasons),
            "verdict": self.verdict,
        }


def evaluate_gates(
    enhancement_ppb: np.ndarray,
    plume_mask: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    sigma_col_ppb_limit: float = 80.0,
    mask_fraction_limit: float = 0.15,
    max_mean_enhancement_ratio: float = 1.0,
) -> GateVerdict:
    """Apply the honesty gates to one retrieval.

    Args:
        enhancement_ppb: (H, W) column enhancement field in ppb.
        plume_mask: (H, W) boolean/0-1 plume mask over the same window.
        valid: optional (H, W) mask of usable pixels. Nodata pixels are
            typically filled with a constant, which would deflate a MAD-based
            sigma and let a noisy scene slip the gate; pass the valid mask so
            both statistics are computed over real observations only.
        sigma_col_ppb_limit: max tolerated background column sigma.
        mask_fraction_limit: max tolerated plume-mask fraction of the window.
        max_mean_enhancement_ratio: max mean in-mask enhancement as a multiple
            of the ambient methane column. See the physical-plausibility note
            below; 1.0 means "the mask may not claim the column is doubled
            across its whole footprint".

    Returns:
        GateVerdict; ``artifact_dominated`` True means withhold the flux.

    Physical plausibility
    ---------------------
    The third gate came out of the 2026-08-25 controlled-release runs. On
    scenes with a **metered zero release** the mask still claimed mean
    enhancements of 1,361 and 8,667 ppb, and on real 5-7 t/h releases it
    claimed 9,341-11,718 ppb -- four to six times the entire ambient column,
    sustained across the whole mask. No point source at these rates can do
    that; a mask-wide mean above the ambient column means the mask has locked
    onto a surface or infrastructure artifact.

    This bound is deliberately permissive. A real plume's *core* can approach
    the ambient column near the source; the constraint is on the mask-wide
    *mean*. For reference, a catalog-consistent 26 t/h plume over the masks we
    were producing needs a mean near 170 ppb, roughly a tenth of the limit.
    """
    field_arr = np.asarray(enhancement_ppb, dtype=float)
    mask = np.asarray(plume_mask).astype(bool)
    if field_arr.shape != mask.shape:
        raise ValueError(
            f"enhancement and mask shapes differ: {field_arr.shape} != {mask.shape}"
        )
    if valid is None:
        valid_arr = np.ones(field_arr.shape, dtype=bool)
    else:
        valid_arr = np.asarray(valid).astype(bool)
        if valid_arr.shape != field_arr.shape:
            raise ValueError(
                f"enhancement and valid shapes differ: "
                f"{field_arr.shape} != {valid_arr.shape}"
            )

    n_window = int(valid_arr.sum())
    n_mask = int((mask & valid_arr).sum())
    mask_fraction = n_mask / n_window if n_window else 0.0

    background = field_arr[~mask & valid_arr]
    background = background[np.isfinite(background)]
    # A mask that swallows the whole window leaves no background to measure;
    # the mask gate below is what catches that case.
    sigma = robust_scene_sigma(background) if background.size else float("inf")

    in_mask = field_arr[mask & valid_arr]
    in_mask = in_mask[np.isfinite(in_mask)]
    mean_enh = float(in_mask.mean()) if in_mask.size else float("nan")
    enh_limit = max_mean_enhancement_ratio * BACKGROUND_CH4_PPB

    reasons: list[str] = []
    if np.isfinite(mean_enh) and mean_enh > enh_limit:
        reasons.append(
            f"mean in-mask enhancement {mean_enh:.0f} ppb is "
            f"{mean_enh / BACKGROUND_CH4_PPB:.1f}x the ambient column — "
            f"unphysical for a point source (limit {enh_limit:.0f} ppb)"
        )
    if sigma > sigma_col_ppb_limit:
        reasons.append(
            f"sigma_col {sigma:.1f} ppb > {sigma_col_ppb_limit:.0f} ppb limit"
        )
    if mask_fraction > mask_fraction_limit:
        reasons.append(
            f"plume mask {mask_fraction:.1%} of window > "
            f"{mask_fraction_limit:.0%} limit"
        )

    return GateVerdict(
        sigma_col_ppb=float(sigma),
        mask_fraction=float(mask_fraction),
        n_mask_px=n_mask,
        n_window_px=n_window,
        artifact_dominated=bool(reasons),
        reasons=tuple(reasons),
        mean_enhancement_ppb=mean_enh,
    )
