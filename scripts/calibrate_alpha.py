#!/usr/bin/env python
"""Measure our simplified MBMP calibration against the production RTM LUT.

Runs in `.venv-mars` (needs marss2l). Writes `config/rtm_calibration.json`.

Why
---
`config/default.yaml` carries single effective absorption coefficients
(`mbmp.alpha_*_per_ppb`) seeded from literature magnitudes. The 2026-08-25
flux audit found they imply Sentinel-2 is several times more sensitive to
methane than UNEP's production radiative-transfer LUT says, which makes every
ppb our own chain reports too small by that factor -- and, because the
sigma_col honesty gate is expressed in ppb, makes the gate threshold
incomparable between the two chains.

This script measures the discrepancy instead of assuming it, by sampling
marss2l's `TransmittanceCH4InterpolationFromDict` over the geometry grid we
actually observe at and fitting a curve our own chain can evaluate without
marss2l installed.

What is stored
--------------
NOT marss2l's lookup table. We store the coefficients of OUR OWN polynomial
fit to the curve we measured, plus the provenance needed to reproduce it.
The underlying physics is Varon et al. (2021, AMT 14, 2771) and Gorrono et
al. (2023, AMT 16, 89); the LUT we sampled ships with `marss2l` (LGPL-3.0,
UNEP/IMEO) and must be cited wherever these numbers are used.

Parameterisation
----------------
Our retrieval works in log space, x = ln(B11/B12)_target - ln(B11/B12)_ref,
which relates to marss2l's ratio as x = -ln(ratio_IL). We fit

    dXCH4(x) = c1*x + c2*x^2 + c3*x^3          (forced through the origin)

per (satellite, SZA, VZA). c1 is the linear-regime sensitivity: the simplified
chain's equivalent is 1/(alpha_b12 - alpha_b11).

    python scripts/calibrate_alpha.py            # writes config/rtm_calibration.json
    python scripts/calibrate_alpha.py --dry-run  # print only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SATELLITES = ("S2A", "S2B")
SZA_GRID = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0)
VZA_GRID = (0.0, 5.0, 10.0)
# Ratio range spanning no-plume to a strong plume; below ~0.80 the LUT is
# extrapolating well past anything we would quantify anyway.
RATIO_SAMPLES = np.concatenate([
    np.linspace(1.0, 0.98, 21),
    np.linspace(0.978, 0.90, 40),
    np.linspace(0.895, 0.82, 16),
])


def fit_curve(x: np.ndarray, ppb: np.ndarray) -> tuple[float, float, float, float]:
    """Least-squares cubic through the origin; returns (c1, c2, c3, max_resid_frac)."""
    design = np.stack([x, x**2, x**3], axis=1)
    coeffs, *_ = np.linalg.lstsq(design, ppb, rcond=None)
    pred = design @ coeffs
    nonzero = ppb > 1.0
    resid = (np.abs(pred[nonzero] - ppb[nonzero]) / ppb[nonzero]).max() if nonzero.any() else 0.0
    return float(coeffs[0]), float(coeffs[1]), float(coeffs[2]), float(resid)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(REPO / "config" / "rtm_calibration.json"))
    args = ap.parse_args(argv)

    from marss2l.mars_sentinel2 import transmittance_to_ch4 as ttc

    try:
        import marss2l
        version = getattr(marss2l, "__version__", "unknown")
    except Exception:  # pragma: no cover - provenance only
        version = "unknown"

    tr = ttc.TransmittanceCH4InterpolationFromDict()
    x = -np.log(RATIO_SAMPLES)  # our log-space coordinate, >= 0

    entries = []
    worst_resid = 0.0
    for sat in SATELLITES:
        for sza in SZA_GRID:
            for vza in VZA_GRID:
                ch4 = tr.deltach4_from_ratio_transmittance(
                    satellite=sat, sza=sza, vza=vza, ratio_il=RATIO_SAMPLES,
                )
                ppb = np.asarray(
                    ch4.values if hasattr(ch4, "values") else ch4, dtype=float
                ).ravel()
                if not np.all(np.isfinite(ppb)):
                    print(f"  skip {sat} SZA={sza} VZA={vza}: non-finite LUT output")
                    continue
                c1, c2, c3, resid = fit_curve(x, ppb)
                worst_resid = max(worst_resid, resid)
                entries.append({
                    "satellite": sat, "sza": sza, "vza": vza,
                    "c1": c1, "c2": c2, "c3": c3,
                    "max_relative_residual": resid,
                })

    if not entries:
        print("no usable LUT samples — aborting")
        return 1

    c1s = np.array([e["c1"] for e in entries])
    # Our simplified chain: dX = x / (alpha_b12 - alpha_b11), i.e. c1_ours = 1/dalpha
    d_alpha_simplified = 1.2e-4 - 3.0e-5
    c1_simplified = 1.0 / d_alpha_simplified
    # c1 is ppb per unit log-ratio, so a LARGER c1 means more methane is
    # inferred from the same band-ratio change. The RTM's c1 exceeds ours,
    # so our chain UNDERSTATES ppb by this factor (equivalently, our alpha
    # assumes S2 is this many times more sensitive than the RTM says).
    understatement = c1s / c1_simplified

    payload = {
        "_comment": (
            "Our own cubic fits to the CH4 transmittance curve measured from "
            "marss2l's RTM LUT. Physics: Varon 2021 (AMT 14, 2771), Gorrono 2023 "
            "(AMT 16, 89). LUT sampled from marss2l (LGPL-3.0, UNEP/IMEO) — cite "
            "it wherever these numbers appear. Regenerate with "
            "scripts/calibrate_alpha.py in .venv-mars."
        ),
        "parameterisation": "dXCH4_ppb = c1*x + c2*x^2 + c3*x^3, x = -ln(ratio_IL)",
        "marss2l_version": version,
        "ratio_range": [float(RATIO_SAMPLES.min()), float(RATIO_SAMPLES.max())],
        "max_relative_residual": worst_resid,
        "simplified_chain_c1": c1_simplified,
        "ppb_understatement_factor": {
            "_comment": (
                "How many times too SMALL the simplified chain's ppb values are, "
                "i.e. how many times too sensitive its alpha assumes S2 to be. "
                "Geometry-dependent, hence the spread."
            ),
            "min": float(understatement.min()),
            "max": float(understatement.max()),
            "median": float(np.median(understatement)),
        },
        "entries": entries,
    }

    print(f"sampled {len(entries)} (satellite, SZA, VZA) combinations")
    print(f"cubic fit worst relative residual: {worst_resid:.2%}")
    print(f"simplified chain c1 = {c1_simplified:,.0f} ppb per unit log-ratio")
    print(f"RTM c1 range        = {c1s.min():,.0f} .. {c1s.max():,.0f}")
    print(f"simplified chain understates ppb by {understatement.min():.1f}x .. "
          f"{understatement.max():.1f}x (median {np.median(understatement):.1f}x)")
    print("  -> a single scalar correction is NOT adequate; the spread is geometry")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
