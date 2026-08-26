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

**Status: RESOLVED 2026-08-25 — see the audit below. Suspect 1 was correct
in symptom but wrong in cause: the enhancement field was not the problem,
the plume mask was, and the mask was wrong because the model was fed inputs
at the wrong scale.**

---

## Flux audit — 2026-08-25

**Reproduce:** `python scripts/flux_audit.py` (offline, reads bundles) ·
`scripts/mars2l_demo.py` for the re-runs · new tests in
`tests/test_flux_audit.py`, `tests/test_gates.py`.

### Suspect cleared first: units

`marshsi.quantification` converts ppb → ppm·m with a factor `H/1000 = 8`
(H = 8,000 m) and then `ppm·m × 1e-6 × 1000/22.4 × 0.01604` kg/m².
That path yields **5.7286e-6 kg m⁻² ppb⁻¹**. Our own independent hydrostatic
path (`retrieve.mbmp.column_mass_kg_m2`, P/g dry-air column at 1013 hPa)
yields **5.7214e-6**. Agreement 0.13% — units are not the error, confirmed
numerically rather than by inspection.

### The identity that localises the error

Substituting `IME = mean_ppb · C · N · A_pix` and `L = sqrt(N · A_pix)`
into `Q = 3600 · U_eff · IME / L` collapses it to

```
Q = 3600 · U_eff · C · mean_ppb · sqrt(plume area)
```

So Q is **linear in the mean in-mask enhancement** and grows as the
**square root of the mask area**. A mask 4× too large doubles Q on its own.
Decomposing the recorded `q_output` dicts this way showed the production
runs claiming **2,177–2,720 ppb mean enhancement sustained over 14–24 km²** —
i.e. the total methane column more than doubled across an area the size of a
small city, from a 26–82 t/h point source. That is the implausible term.

### Root cause: model input scale (10⁴)

`plume_detection_model.predict` documents its inputs as "TOA reflectances
multiplied by 10000", and the loaded MARS-S2L checkpoint has
`norm_data=False`, i.e. it normalises by `tensor /= 5000`. `mars2l_demo.py`
was passing **0–1 reflectance**, so every radiance channel arrived ~10⁴×
too small — effectively a constant image.

The model survived this because it is `cat_mbmp=True`: the band-ratio
channel it prepends is scale-invariant and carried real signal. So the
network ran **on the MBMP ratio alone, with all radiance context zeroed** —
and radiance context is exactly what lets it separate a plume from a bright
surface artifact. Result: hugely oversized, highly confident masks.

This also explains why the ΔXCH₄ field itself looked reasonable: `ratio_IL`
normalises by the scene mean, so the retrieval is scale-invariant too. Only
the segmentation was corrupted — and via `sqrt(area)`, the segmentation is
what set the flux.

### Three compounding preprocessing defects found alongside it

1. **Partial-granule backgrounds.** One MGRS tile/day can hold several
   SAFEs, and a swath-edge granule may cover only part of the tile. Taking
   `safes[0]` blindly gave a background missing **43% of the window**;
   those pixels were then fed in as −1000 DN (nodata + baseline offset).
   Fixed: scene selection now probes actual window coverage.
2. **Cross-platform / cross-orbit pairing.** The background was picked at
   `target − 5 d`, which for one tile is the *other* platform on a
   *different* relative orbit — S2A/S2B SWIR calibration differences and
   BRDF, neither of which cancels in a band ratio. Fixed: same platform,
   exact 10-day multiples (same relative orbit) preferred.
3. **No cloud screening at all.** `validmask` was hardcoded all-ones.
   Fixed: CloudSEN12 (`UNetMobV2_V2`, bundled with marss2l, still no
   accounts) now screens both scenes, and background selection rejects
   cloudy candidates against the frozen `sentinel2.max_cloud_fraction`.

### Corrected results

| Run | scale | pairing | coverage | cloud screen | mask px | mean ppb | σ_col ppb | Q / catalog |
|---|---|---|---|---|---|---|---|---|
| Korpezhe v2 (original) | reflectance ❌ | cross-orbit 5 d | 57% | none | 48,772 | 2,177 | not computed | **12.9×** |
| Korpezhe v5 | DN ✓ | same orbit 50 d | 100% | none | 20,820 | 1,496 | 1,074 | 5.8× |
| Korpezhe v6 | DN ✓ | same orbit 50 d | 100% | ✓ | 0 | — | — | **no detection** |
| Permian (original) | reflectance ❌ | cross-orbit 5 d | 57% | none | 35,980 | 2,720 | not computed | **5.8×** |
| Permian v4 | DN ✓ | same orbit 30 d | 100% | ✓ | 19,614 | 900 | 1,812 | **1.4×** |

The 6–13× overestimate is accounted for. With correct inputs the Permian
flux lands **within 1.4× of the published rate** — the remaining gap is well
inside the combined wind, mask-definition and 30-day-baseline uncertainty,
and is no longer a systematic order-of-magnitude error.

### Retraction

The claim "production model DETECTED plumes at BOTH catalog targets
(scene_score .996/.995)" **does not survive the fix and is withdrawn.**
Those scores were produced from inputs at the wrong scale. Corrected:

* **Permian 2026-04-27: detection holds** (`is_plume=True`, score 0.994)
  with a clean same-orbit, cloud-free, fully-covering background.
* **Korpezhe 2026-08-05: no detection** (`is_plume=False`, score 0.213).
  The only S2 scene the anonymous mirror offers near the event
  (S2B 2026-08-03, −2 d) is **53% cloud + shadow** by CloudSEN12. Verified
  as real cloud, not bright-sand false positive: the flagged class is much
  brighter than clear ground in B02 (0.319 vs 0.189) but *not* in B11
  (0.431 vs 0.417), with a matched 27% shadow population — sand brightens
  toward SWIR, cloud does not. The STAC "0.00% cloud" that this campaign
  had been quoting refers to the **S2C scene of 2026-08-05**, which the GCS
  mirror does not carry; it was never a property of the pixels we used.

Korpezhe v6 is therefore **not a clean null for the event** — it is a
scene-unavailability result, and belongs in the observability atlas as such.

### Second finding: our simplified alpha is ~5× off

Sampling marss2l's RTM LUT (`TransmittanceCH4InterpolationFromDict`) near
ratio ≈ 1 gives an effective absorption coefficient of **1.4e-5 – 2.0e-5
per ppb** across SZA 24–45°. Our `config/default.yaml` simplified pair
implies `alpha_b12 − alpha_b11 = 9.0e-5 per ppb`, i.e. we assume Sentinel-2
is **4.4–6.3× more sensitive to methane than the production RTM says**.

Consequence: every ΔXCH₄ (and therefore σ_col) our own simplified chain has
produced is roughly 5× too small in ppb. The 70–208 ppb σ_col we measured
over the Korpezhe dune margin corresponds to ~400–1,200 ppb on the RTM
scale. The config comment already said "replace alpha with an RTM-derived
LUT before quoting absolute rates"; this quantifies by how much.

### Gate status and one honest caveat

Honesty gates now apply to the production path too — they previously did
not, which is how a 337 t/h number reached a table at all. They are promoted
out of the demo script into `src/plumechaser/retrieve/gates.py`, driven by
`config.gates`, computed over **valid pixels only** (a constant nodata fill
would otherwise deflate a MAD-based sigma), and unit-tested.

Every corrected run trips the σ_col gate (494–2,280 ppb vs an 80 ppb limit)
and its quantification is withheld. **Caveat that must not be lost:** the
80 ppb limit was calibrated on our simplified-alpha scale, which the finding
above shows is ~5× more sensitive than the RTM scale it is now being applied
to. On the RTM scale the equivalent limit is roughly 350–500 ppb. The
verdicts happen to be unchanged (all corrected runs exceed even 500 ppb),
but **the threshold needs re-deriving per retrieval scale before the
Nov 1 freeze** — it is currently being compared against the wrong yardstick.

### Also fixed in passing

`gcs_band_href` used the **L2A** `RADIO_ADD_OFFSET` band_id ordering
(B11=10, B12=11). L1C carries **13** entries including the B10 cirrus band,
so the correct L1C mapping is B10=10, **B11=11, B12=12** — verified against
a real `MTD_DS.xml`. Numerically harmless today because all 13 offsets are
currently −1000, but it silently reads the wrong band's offset the moment
ESA issues per-band values. The "known gotcha" note stating B11=10/B12=11
was describing L2A and should be corrected wherever it is recorded.

### Still open

* Absolute flux is **still not quotable**: gates withhold every run, and the
  one surviving number (Permian, 1.4×) comes from a 30-day baseline chosen
  only because the 10- and 20-day candidates were 84% and 25% cloudy.
* The 26 t/h Korpezhe case remains unvalidated for lack of a clear scene.
* A controlled-release cross-check (Ehrenberg/Casa Grande AZ, Sherwin et al.
  2023 published rates) is still the right way to close this properly.
* **Copernicus Data Space access would directly attack the dominant
  remaining error term.** The binding constraint is now scene availability:
  the anonymous mirror lags ~12 d, carries no S2C, and forced both a 30-day
  and a 50-day background baseline. More candidate dates means shorter
  baselines, less surface change, lower σ_col.
