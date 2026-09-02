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

### Second finding: our simplified alpha is ~5× off — now measured and fixed

**Update 2026-08-25 (later same day):** the estimate below came from three
spot samples. `scripts/calibrate_alpha.py` now measures the full curve over
SZA 10–70°, VZA 0–10°, both platforms (42 geometries, cubic fit, worst
residual 1.4%; stored in `config/rtm_calibration.json`).

Measured result: our simplified coefficients **understate methane columns by
2.5× to 6.3×, median 4.4×**, driven almost entirely by solar zenith angle —
viewing zenith moves it under 5%. The spread matters: a single scalar
correction would be wrong by a factor of 2.5 across the observing envelope,
which is why `retrieve/calibration.py` stores a geometry-dependent curve
rather than a replacement constant. The RTM is also markedly non-linear, so
a purely linear coefficient understates strong plumes by tens of percent on
top of the scale error.

The gate threshold problem this created is resolved by anchoring the gate on
fractional band-ratio noise instead of ppb — see
`docs/ANALYSIS_PLAN.md` §9, addendum 2026-08-25. The pre-registered operating
point is unchanged.

*Original spot-check estimate, retained for the record:*

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

---

## Controlled-release validation — 2026-08-25, and it closes the audit

Metered ground truth from the Stanford/EDF single-blind tests
(`config/controlled_release_truth.json`, Sherwin et al. 2023/2024). This is
the only dataset where the true rate is **known** rather than inferred from
another satellite's catalog, and it settles the absolute-flux question.

`python scripts/controlled_release.py --limit 6`

| Overpass | Metered truth | is_plume | score | Our flux | vs truth | mean in-mask ΔXCH₄ |
|---|---|---|---|---|---|---|
| Ehrenberg 2021-11-01 | **0 (zero control)** | **True** | 0.998 | **157 t/h** | — | 8,667 ppb (**4.8× column**) |
| Casa Grande 2022-11-25 | **0 (zero control)** | **True** | 0.996 | **146 t/h** | — | 1,361 ppb |
| Ehrenberg 2021-10-19 | 7.18 t/h | True | 0.993 | 326 t/h | **45×** | 11,718 ppb (**6.5× column**) |
| Ehrenberg 2021-10-29 | 5.02 t/h | True | 0.9995 | 223 t/h | **44×** | 9,341 ppb (**5.2× column**) |
| Ehrenberg 2021-10-24 | 3.99 t/h | False | 0.120 | — | — | (98% cloud, correct null) |

All quantifications were **withheld by the honesty gates**. Every scene used
a same-platform, same-relative-orbit, cloud-screened, fully-covering
background, with the campaign window excluded so no reference carried its own
release.

### The finding

**Our artifact floor is ~150 t/h.** On scenes with a metered zero the chain
reports 146 and 157 t/h and the production model calls a plume at ≥0.996
confidence. The releases we were trying to measure are 1.4–7.2 t/h — twenty
to forty times *below* the floor. The 223–326 t/h returned on real release
days is only ~1.5–2× the zero-control value, i.e. **the "signal" on release
days is the same artifact as on the zero day**, not the gas.

The mechanism is explicit in the mean in-mask enhancement: 8,667–11,718 ppb
at Ehrenberg, **4.8–6.5× the entire ambient methane column, sustained across
the mask** — and 8,667 ppb of it on a day when the meter read zero. The mask
is locking onto persistent structure at the site (the release rig, hard
standing, vehicles) whose B12/B11 ratio differs from the background pass by
about 10%, which the RTM inverts into ~10,000 ppb.

Two distinct artifact modes are visible: Ehrenberg is compact and
high-amplitude (~10,000 ppb over 0.4–1.0 km²), Casa Grande is diffuse and
moderate (1,361 ppb over 16.7 km²).

### This corrects an earlier reading in this document

The Permian result above — retrieved flux 1.4× the catalog rate — was
reported as encouraging. **It is not evidence of quantification skill.** With
an artifact floor near 150 t/h and a catalog rate of 82 t/h, a 117 t/h
retrieval is fully consistent with pure artifact. The chain cannot presently
distinguish an 82 t/h source from zero, so the agreement is coincidence. The
1.4× figure must not appear as a validation result anywhere.

### What this does and does not indict

It indicts **our chain**, not Sentinel-2 and not MARS-S2L as UNEP operates
it. On these same overpasses the published teams retrieved rates within
roughly a factor of two of truth (2021-10-19, truth 7,176 kg/h: Harvard
4,069 · Kayrros 6,790 · LARS 5,468 · SRON 3,760). We are 45× off on the same
pixels. The gap is plume delineation and analyst QC — a mask of 2,626 px
(1.05 km²) for a 7 t/h release is one to two orders of magnitude too large.

### Consequences

1. **The flux audit is closed with a negative result, and that is a real
   result.** Absolute flux from this chain is not usable at any rate we have
   targeted, and we can now say so with a measured number rather than a
   suspicion.
2. **The honesty gates are validated against known truth.** They withheld
   100% of quantified runs, including both false detections. This is the
   strongest evidence the project has that the gate system works.
3. **`scene_score` is not a trustworthy detection statistic in our
   pipeline.** It read ≥0.996 on two metered zeros. Detection claims need
   the gates and the physical-plausibility check, not the score.
4. **New gate added** (`gates.max_mean_enhancement_ratio`): a mask-wide mean
   enhancement above the ambient column is unphysical for a point source.
   Deliberately permissive — a 26 t/h plume needs ~170 ppb, a tenth of the
   limit — but it catches every bad run in this document, including the ones
   that passed both existing gates.
5. **The observability atlas gains its sharpest entry**: at 20 m resolution
   over bright desert, with free-sensor scene availability, this chain's
   quantification floor is ~150 t/h, against a demonstrated community
   detection floor near 1.0–1.4 t/h.

---

## Zero controls at n=8, and a negative result for delineation

### The artifact floor is a distribution, not a point

Eight metered-zero Sentinel-2 scenes have now been run (two zero controls
plus six no-release Casa Grande overpasses; two more timed out on the GCS
mirror).

| Scene | Detected? | False flux |
|---|---|---|
| Ehrenberg 2021-11-01 | yes (0.998) | 157 t/h |
| Casa Grande 2022-11-25 | yes (0.996) | 146 t/h |
| Casa Grande 2022-10-24 | yes | 136 t/h |
| Casa Grande 2022-10-21 | yes | 111 t/h |
| Casa Grande 2022-10-19 | yes (0.999) | 34 t/h |
| Casa Grande 2022-10-11 | yes (0.989) | 18 t/h |
| Casa Grande 2022-10-29 | yes (0.862) | 14.8 t/h |
| Casa Grande 2022-10-14 | **no** (0.352) | — |
| Casa Grande 2022-10-16 | **no** (0.008) | — |

**False-detection rate 7/9 on scenes with no emission**, and false flux
spanning **14.8–157 t/h**.

### The two sites fail differently, and that is why two gates are needed

| Site | Mean in-mask ΔXCH₄ on metered zeros | Caught by |
|---|---|---|
| Ehrenberg | 8,667 ppb (4.8× ambient column) | both gates |
| Casa Grande | 883–1,361 ppb (0.49–0.76× ambient) | **σ_col only** |

The Casa Grande artifacts are moderate-amplitude and entirely plume-plausible
in magnitude — they sit *below* the physical-plausibility ceiling and would be
waved through by it. Only the σ_col noise gate rejects them. The Ehrenberg
artifacts are the opposite: extreme amplitude, caught by either gate alone.

Neither gate is sufficient by itself, and that is not redundancy — the two
sites fail by different physical mechanisms (site infrastructure with an
extreme band-ratio offset, versus broad low-amplitude surface change), so
each gate is load-bearing at one of them. Any future simplification that
drops one would silently reopen half the failure space. This supersedes the earlier "~150 t/h" point
estimate, which came from the two largest of eight and was therefore the
optimistic end of the wrong summary statistic. The honest statement is a
distribution with a median near 120 t/h, and the two correct nulls matter:
the failure is frequent but not universal, so it is a property of particular
scenes rather than an unconditional artefact.

Every one of the six was withheld by the gates.

### Delineation removes 100% of the mask on real pixels

`retrieve/delineate.py` was applied to the real Ehrenberg releases:

| Overpass | Metered | Model mask | After delineation | Dominant rule |
|---|---|---|---|---|
| 2021-10-19 | 7.18 t/h | 3,285 px | **0 px** | amplitude |
| 2021-10-29 | 5.02 t/h | 2,676 px | **0 px** | amplitude |

Every pixel in the production model's mask exceeds twice the ambient methane
column, so the physical-plausibility rule removes all of them. **The mask
contains no physically possible plume pixel at all** — it is artifact
throughout, which is consistent with the mean enhancements of
8,667–11,718 ppb measured earlier.

**What this does and does not establish.** It refutes the hypothesis that a
real plume sits inside the model's mask alongside artifact and can be
separated geometrically. It does **not** establish that the scene contains no
recoverable plume: a 7 t/h release produces roughly 600 ppb of enhancement,
and the segmentation model — selecting for large anomalies — may never have
included those pixels in its mask. Delineation can only ever constrain a
candidate mask; it cannot recover signal the candidate never contained.

The synthetic test reporting 45.3× → 1.00× therefore validated the machinery
only. Its true plume was constructed to carry exactly the metered rate, so
recovering the true mask returned truth by construction. It should not be
cited as evidence that the gap is closable.

### The one experiment left that could close the gap

Derive the candidate mask from the ΔXCH₄ field itself at plume-plausible
amplitude — a few σ above background, within the downwind sector, near the
known source — instead of from the segmentation model, then delineate and
quantify. That answers the remaining question directly: *is there any
recoverable plume signal in these pixels at all?* Nothing else in the current
toolchain can answer it, and a null there would close the question for good.

Supporting evidence that the attempt is worth making, from the atlas: our
artifact floor sits **27–253× above our own noise-limited floor**, while the
floor other teams demonstrate on the same pixels sits near it. The headroom
is real; what is unproven is our ability to reach into it.

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
