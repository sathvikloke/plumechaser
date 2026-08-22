#!/usr/bin/env python
"""Synthetic end-to-end smoke run: detect -> verify -> cue -> quantify.

No network, no satellite data: injects a Gaussian CH4 plume into a synthetic
scene pair, runs the full detection stack, and quantifies the emission rate.
Useful as a 10-second sanity check after any refactor.

    python scripts/smoke_check.py
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plumechaser.cue.policy import decide_cue  # noqa: E402
from plumechaser.detect.verify import Candidate, persist_candidates  # noqa: E402
from plumechaser.retrieve.ime import quantitate  # noqa: E402
from plumechaser.retrieve.mbmp import mbmp_enhancement_ppb, plume_mask  # noqa: E402

ALPHA_11, ALPHA_12 = 3.0e-5, 1.2e-4


def main() -> int:
    rng = np.random.default_rng(0)
    h = w = 64
    texture = 0.9 + 0.2 * rng.random((h, w))
    b11_ref = 0.25 * texture
    b12_ref = 0.15 * texture

    truth = np.zeros((h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    truth += 1500.0 * np.exp(-(((yy - 30) ** 2) / 24 + ((xx - 40) ** 2) / 24))

    b11_t, b12_t = b11_ref.copy(), b12_ref.copy()
    b12_t *= np.exp(-ALPHA_12 * truth)
    b11_t *= np.exp(-ALPHA_11 * truth)

    d_xch4 = mbmp_enhancement_ppb(b11_t, b12_t, b11_ref, b12_ref, ALPHA_11, ALPHA_12)
    mask = plume_mask(d_xch4)
    assert mask.sum() > 20, "plume mask too small"

    res = quantitate(d_xch4, mask, u10_ms=4.0, pixel_area_m2=400.0,
                     mc_samples=300, seed=20270307)

    dets = [
        Candidate(f"d{k}", date(2025, 6, 1) + timedelta(days=k), 59.0, 39.0, 6.0,
                  source_pass_ids=[f"p{k}"])
        for k in range(5)
    ]
    confirmed, _ = persist_candidates(dets, min_passes=4, gap_days=3)
    action, reason = decide_cue(persistence_passes=len(confirmed), z_peak=float(mask.sum()))

    print("=== PlumeChaser smoke check ===")
    print(f"mask pixels        : {int(mask.sum())}")
    print(f"injected peak ppb  : {truth.max():.0f}")
    print(f"recovered peak ppb : {np.nanmax(d_xch4):.0f}")
    print(f"rate Q             : {res.q_kg_h:,.0f} kg/h "
          f"(CI {res.ci_low:,.0f}-{res.ci_high:,.0f})")
    print(f"persistence        : {len(confirmed)} confirmed -> cue={action}")
    assert action == "cue_sentinel2"
    assert res.q_kg_h > 0 and res.ci_low < res.q_kg_h < res.ci_high
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
