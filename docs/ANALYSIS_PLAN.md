# PlumeChaser — Frozen Analysis Plan (v1.0)

> **Status:** PRE-REGISTRATION DRAFT. This document must be deposited
> (Zenodo, restricted until publication) and tagged in git (`freeze-2026-11-01`)
> **before** any post-freeze catalog file is opened for scoring.
> After freezing, changes are made only via dated addenda in section 9.

## 1. Design principle

Config-frozen, pre-registered agreement study — not a blind trial. We have
read public catalogs during development; blindness is therefore not claimed.
Integrity comes from: immutable config hash, environment lock, render-pack
hashing, and scoring scripts that touch labels exactly once.

## 2. Endpoints

* **Primary:** transient-tier precision/recall of our detections against the
  SRON weekly operational list (strict mode), champion basins only.
* **Secondary:** lenient mode vs Schuit et al. (2021) full-year set;
  persistent-tier coverage statistics; FP rate via structured self-labeling.
* **Exploratory:** Region-C coverage tier; incremental value vs archived
  catalogs (stretch).

**Dual-mode interpretation (mandatory in every report):**
"Strict mode benchmarks against operational human curation; lenient mode
estimates the algorithmic ceiling; the strict–lenient gap is itself our
quantification of curation effect."

## 3. Match hierarchy

| Tier | Rule |
|---|---|
| Transient | event-level match: ≤25 km AND ≤±3 days |
| Persistent | source-level coverage: detection within 25 km of a known persistent source |

Tolerance parameters may be varied ONLY inside the pre-registered sensitivity
grid (section 5).

## 4. Statistics

* Cluster bootstrap (block = source cluster) for all CIs; percentile method,
  2000 draws, seed `20270307` recorded per run manifest.
* Point estimates always from unperturbed inputs.
* Rate comparison: Bland–Altman restricted to comparable-footprint pairs
  (S2 vs reference at Varon sites / Carbon Mapper spot checks); pairing
  caveat stated on figure. TROPOMI-vs-S2 reported as ratio table only.
* ERA5 primary winds; GEOS-FP delta term computed for Korpezhe and reported
  as wind-product uncertainty.

## 5. Power branches & sensitivity

Effective sample size n_eff = number of REFERENCE clusters after greedy
clustering (25 km / 14 d central parameters).

| n_eff | Branch | Consequence |
|---|---|---|
| ≥ 30 | full | P/R headline with bootstrap CIs |
| 15–29 | intermediate | agreement rate + calibration curve; P/R exploratory |
| < 15 | descriptive | case-study reproduction + coverage stats |

Sensitivity sweep: distance ∈ {12.5, 25, 50} km × window ∈ {7, 14, 21} d.
Branch chosen on central combo BY RULE; all nine combos reported. If ≥3/9
combos change branch, no hierarchy is claimed (robustness flag).

## 6. False-positive protocol

Two-pass labeling on an immutable render pack (pack SHA-256 recorded):
pass 1 = CH4 morphology only; pass 2 = full context. Independent raters,
adjudication with written rationale, Cohen's κ reported alongside
single-rater AND adjudicated FP rates.

Controls: ≥30 known-positive catalog scenes from regions never reviewed
during development, geography stripped; annotator hit-rate <85% triggers
recalibration before further labeling.

Throughput: random subsample n=150 when candidates exceed 300.

**Non-discrimination caveat (verbatim in outputs):** "Annotators are the
system developers; κ quantifies inter-rater consistency, not expert
accuracy."

## 7. Null-result routing (deposited verbatim)

"If observed transient detections fall below climatologically expected counts
(N_expected from cloud statistics × historical rates), the deficit is
reported as a first-order observability finding feeding the atlas (D3), not
a failed validation."

## 8. Success ladder (pre-committed)

| Tier | Criteria |
|---|---|
| Full | plan executed; ≥15 clusters; rates reproduce Varon envelope; atlas assembled |
| Solid | autonomous pipeline; descriptive agreement; FP protocol complete |
| Floor | end-to-end demo on Korpezhe + naive-vs-CNN delta + cost/latency analysis |

Kill dates: Dec 15 autonomy · Feb 1 results tabulated · Mar 1 content freeze.

## 9. Addenda

* **2026-08-22 (pre-freeze):** License stack verified — Zenodo 13903869
  training data **CC-BY-4.0**; `marss2l` **LGPL-3.0** (usable as dependency;
  clean-room MBMP contingency retired); Orbio Eucalyptus custom
  non-commercial (fair-compatible). CAMS weekly CSV adopted as the strict-mode
  reference source (`sites.ecmwf.int/.../data_methane_explorer.csv`, schema
  fingerprint `b8db5cd2`). Power assessment on real data: champion-set
  n_eff = 57 → branch FULL (see docs/PRE_FREEZE_POWER_ASSESSMENT.md).

* **2026-08-25 (pre-freeze) — flux audit, gate reparameterisation, RTM
  calibration.** Full technical record in `docs/S2_REAL_DATA_FINDINGS.md`.

  1. *Retraction.* The claim that the MARS-S2L production model detected
     plumes at both catalog targets (scene_score .996/.995) is **withdrawn**.
     Those runs fed the model 0–1 reflectance where it requires DN
     (reflectance × 10⁴), which zeroed every radiance channel and left it
     running on the scale-invariant band-ratio channel alone. Corrected:
     Permian 2026-04-27 holds (0.994); Korpezhe 2026-08-05 is a
     **no-detection** (0.213) because the only scene the anonymous mirror
     carries near the event is 53% cloud + shadow. Korpezhe is therefore a
     scene-unavailability result and routes to the observability atlas under
     §7, not to the agreement study.

  2. *Honesty-gate reparameterisation (§7 operating point UNCHANGED).* The
     gate was written as `sigma_col > 80 ppb`. ppb is not a
     calibration-independent unit: our simplified absorption coefficients
     were measured to understate columns by 2.5–6.3× versus the production
     RTM, so the same physical scene noise produced different verdicts
     depending on which chain measured it. The gate is now anchored on the
     quantity it actually constrains — fractional band-ratio noise —
     `sigma_log_ratio_limit = 0.0072 = 80 ppb × (alpha_b12 − alpha_b11)`,
     with each retrieval converting to ppb using its own calibration slope.
     On the simplified chain this returns exactly 80 ppb by construction, so
     **no pre-registered threshold changes value**; only its expression
     becomes chain-independent. On the RTM scale the equivalent limit is
     ≈500 ppb. Unit-tested in `tests/test_calibration.py`.

  3. *RTM-derived calibration.* `retrieve/calibration.py` replaces the fixed
     `mbmp.alpha_*_per_ppb` with a geometry-dependent cubic measured against
     the production RTM LUT over SZA 10–70°, VZA 0–10°, both platforms
     (`scripts/calibrate_alpha.py`, stored in `config/rtm_calibration.json`,
     worst fit residual 1.4%). The correction is 2.5–6.3× and varies with
     solar zenith, so it could not be absorbed into a single new constant.
     This resolves the §10 accepted risk "simplified MBMP absorption
     coefficients pending RTM-LUT". Fixed-alpha retrieval is retained as the
     educational/fallback path.

  4. *Scope.* No endpoint, match rule, power branch, bootstrap parameter, or
     success-ladder criterion is altered by this addendum. Absolute fluxes
     remain UNDER AUDIT and are withheld by the gates in every run to date.

*(entries appear here only with date + reason)*

## 10. Accepted risks (permanent disclosure)

Developer-annotator circularity (mitigated, not eliminated) · catalogs-as-
truth limitation · TROPOMI physics floor (~8 t/h) · ~~simplified MBMP
absorption coefficients pending RTM-LUT~~ **resolved 2026-08-25, addendum 3**
· backbone layer partially self-referential (labelled
'instrument-and-pipeline') · **free-sensor scene availability**: the
anonymous mirror lags ~12 d and carries no S2C, which forced 30- and 50-day
background baselines and cost us the Korpezhe event entirely (added
2026-08-25; this is an atlas finding as much as a limitation).
