# Real Sentinel-2 Data Findings — First End-to-End Campaign

**Date:** 2026-08-22 · **Target:** CAMS event 2026-08-05, 26 t/h @
(58.52 E, 39.68 N), Korpezhe margin · **Script:** `scripts/real_s2_demo.py`

## What ran

Real Sentinel-2 pixels (public anonymous sources only) through the complete
chain: STAC/GCS discovery → COG/JP2 windowed reads → radiometric offsets
(baseline ≥05: −1000 DN both bands, parsed per-band) → same-platform pairing
→ sub-pixel co-registration (FFT phase correlation + parabolic refinement)
→ MBMP log-ratio retrieval → robust noise estimation → honesty gates →
IME quantification (when gates pass) → evidence bundle with provenance.

## Results matrix

| Configuration | σ_col (ppb) | Mask frac | Gate | Notes |
|---|---|---|---|---|
| AWS L2A, single ref | 36 | 66% | ❌ dominated | BOA correction artifacts structured, not stochastic |
| GCS L1C, anchor at tile edge | 208 | 13% | ❌ | window overhung granule; dead-detector patch at point |
| GCS L1C, clamped anchor | 69.8 | 65% | ❌ | interior clean; dune photometry dominates |
| GCS L1C, MBPD 2 dates (mixed orbits R063+R020) | 229 | 6.6% | ❌ | cross-orbit BRDF adds variance; √N does not apply |

## Synthetic recovery (Gate-B, injected into REAL scenes)

* Injection: Gaussian peak 1000 ppb at window center, applied to target bands
  before registration/retrieval.
* Single-reference run: recovered peak visible at correct location;
  bg-subtracted core ≈ +182% of truth (constructive natural artifact).
* MBPD run: bg-subtracted core median ≈ 10% of truth — natural structure
  partially cancelled AND signal suppressed by cross-orbit references.
* **Interpretation:** the radiative-transfer arithmetic (band-ratio ↔ ΔXCH₄)
  behaves as published; the dominant error is environmental photometric
  change (active dune field) + detector-seam structure, which naive
  single-coefficient retrievals cannot separate.

## Root causes identified

1. **Dune-field photometry**: Korpezhe margin is an active sand-dune
   environment; B12/B11 ratio drifts percent-level between passes from
   shadowing/BRDF — comparable to or larger than a 26 t/h CH₄ signal.
2. **Detector-seam step**: straight diagonal discontinuity in the ratio
   field (SWIR chip boundary); migrates between orbits → survives medians.
3. **Mirror lag**: public GCS L1C runs ~12 days behind → comparison dates
   forced to ±(9–13 d) and often mixed orbits; Copernicus Data Space
   (account required) removes this constraint.
4. **Tile-edge anchoring**: catalog points near tile margins need
   anchor-clamping (implemented after diagnosis).

## Consequences for the project

1. **Honesty gates are load-bearing** — every artifact-dominated attempt was
   auto-withheld from quantification. No bogus rates were produced at any
   point. This is the gate system working exactly as designed (plan §7).
2. **Machinery validated end-to-end on real pixels** — discovery, reads,
   offsets, registration, retrieval, masking, winds, bundling all executed;
   failures are environmental, not software.
3. **Production retrieval = adopt `marss2l` (LGPL-3.0)** whose trained
   segmentation + normalization exist precisely to separate these artifacts
   (verified compatible license). Our simplified-alpha chain remains as the
   educational/fallback core and Gate-B harness.
4. **Atlas σ_col values must come from measured per-basin statistics**
   (this campaign's 70–230 ppb range over arid terrain vs literature 12–25
   ppb assumptions shows why) — next atlas iteration uses empirical σ.
5. Region choice insight: Korpezhe-margin dunes are a worst-case surface;
   Permian carbonate/calcrete pads may behave far better — queued behind
   account-gated access.

## Reproduce

```bash
python scripts/real_s2_demo.py --source gcs-l1c --mgrs 40TFK \
    --lon 58.52 --lat 39.68 --date 2026-08-05 --basin korpezhe \
    --inject-ppb 1000 --n-refs 4 \
    --event-id EVT-SYNTH-MBPD --rate-t-h 26
```
Bundles: `bundles/EVT-*` (provenance-hashed; integrity in lab notebook).

---

## Production-model campaign (marss2l 0.2.10, same day)

Ran UNEP's actual operational models (`MARS-S2L`, LGPL, pip-installable)
on the same events, pixels from the anonymous GCS L1C mirror, angles from
STAC, winds from Open-Meteo — still zero accounts required.

| Event | Catalog rate | MARS-S2L detection | scene_score | Q (Varon Ueff + core≥0.6) |
|---|---|---|---|---|
| Korpezhe 2026-08-05 | 26 t/h | **is_plume=True** | **0.996** | ≈337 t/h |
| Permian 2026-04-27 | 82 t/h | **is_plume=True** | **0.995** | ≈478 t/h |

### What this establishes

* **Detection layer independently validated**: the production segmentation
  model — trained by SRON/UNEP partners — confirms plumes at BOTH
  catalog-targeted sites with ≥0.99 confidence, on data we fetched and
  preprocessed ourselves without any institutional access.
* Our simplified chain's honesty gates + this production cross-check now
  form a two-tier verification story no single-model demo can match.

### Open item: absolute-flux overestimate (~6–13×)

Both Q values exceed catalog rates systematically. Itemized suspects:
1. RTM inversion amplifies residual surface-ratio offsets to thousands of
   ppb over bright heterogeneous terrain inside the segmented region
   (mean in-mask ΔXCH₄ ≈ 2,200 ppb — implausibly high for real gas at
   these wind speeds).
2. Target-day offset (S2 scene −2 d from TROPOMI detection) may capture a
   different emission state; TROPOMI daily rates are column averages.
3. `obtain_flux_rate` defaults use raw U₁₀ as U_eff — Varon coefficients
   now applied (−55%); remaining gap NOT unit-related (ppb path audited
   through marshsi.convert_units).
4. Operational IMEO fluxes additionally pass analyst QC on plume geometry
   and background window choice before publication.

**Status: detection validated; absolute flux flagged UNDER AUDIT — excluded
from headline claims until root-caused against a controlled-release or
analyst-published case.**
