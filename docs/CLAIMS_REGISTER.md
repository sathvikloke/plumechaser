# Claims Register

What this project may and may not assert, with the evidence for each.

**Status:** live until the 2027-03-01 content freeze. Every claim in the
paper, poster, abstract and video must trace to a row here. If a sentence
cannot be traced to a row, either the evidence is missing or the sentence is
overreaching — both are fixable, but only before judging.

**This register is not draft text.** It is a technical record of what the
evidence supports. The students write every sentence of the manuscript
themselves; this exists so they know what the sentences are allowed to say.

---

## 1. Claims the evidence supports

### C1 — A fully open, account-free monitoring pipeline
**Claim:** an end-to-end methane super-emitter pipeline — TROPOMI screening →
automated cueing → Sentinel-2 MBMP retrieval → IME quantification →
infrastructure context — built and operated entirely on anonymous public
sources, requiring no institutional account of any kind.
**Evidence:** the repository; `docs/RUNBOOK.md`; the campaign scripts. Data
paths are Google's public GCS Sentinel-2 mirror, element84 STAC, Open-Meteo
ERA5, the CAMS weekly CSV, and Zenodo record 13903869.
**Strength:** strong; demonstrable by re-running.
**Must accompany:** the account-free property costs scene availability — see
C7, which is a direct consequence.

### C2 — Detector trained and evaluated on real expert-labelled data
**Claim:** CNN precision 0.904 ± 0.045, recall 0.898 ± 0.041 across three
seeds; calibrated RBF-SVC second stage, cross-validated F1 0.848 ± 0.041.
**Evidence:** `docs/TRAINING_REPORT.md`; SRON Zenodo training data
(CC-BY-4.0); three-seed protocol, variance reported rather than a best run.
**Strength:** strong.
**Must accompany:** these are held-out scores on the SRON distribution, not
end-to-end field performance.

### C3 — Honesty gates validated against metered ground truth
**Claim:** the pre-registered withholding rule was tested against controlled
releases with known metered rates and withheld 100% of quantifications,
including on two scenes where the retrieval falsely reported a plume at a
metered release of zero.
**Evidence:** `docs/S2_REAL_DATA_FINDINGS.md`, controlled-release section;
`config/controlled_release_truth.json`; `outputs/controlled_release.json`;
`src/plumechaser/retrieve/gates.py` and its tests.
**Strength:** strong, and unusual — most student and many professional
pipelines never test their safeguards against a known negative.
**Must accompany:** the gates were pre-registered before this test
(`docs/ANALYSIS_PLAN.md` §7), which is what makes the result meaningful
rather than post-hoc.

### C4 — A measured quantification floor
**Claim:** on metered-zero releases this chain reported 146 and 157 t/h with
the production segmentation model calling a plume at ≥0.996 confidence; on
metered 5.0 and 7.2 t/h releases it reported 223 and 326 t/h, only ~1.5–2×
the zero-control value. The absolute-flux artifact floor is therefore
approximately 150 t/h, above every emission rate this study targets.
**Evidence:** as C3. Mean in-mask enhancements of 8,667–11,718 ppb — 4.8–6.5×
the ambient methane column — identify the mechanism as mask over-extension
onto persistent site structure.
**Strength:** strong for the direction and order of magnitude; **n = 2 zero
controls and 2 releases** at the time of writing. State n.
**Must accompany:** this is a limit of *our chain*, not of Sentinel-2 — see
C5.

### C5 — The limitation is ours, not the sensor's
**Claim:** on the same overpasses and the same pixels, the published teams
retrieved within roughly a factor of two of truth (2021-10-19, metered
7,176 kg/h: Harvard 4,069 · Kayrros 6,790 · LARS 5,468 · SRON 3,760 kg/h).
The gap is plume delineation and analyst quality control, not sensor
capability.
**Evidence:** Sherwin et al. 2023 (Sci Rep 13:3836) and its data repository;
our masks were 1–2 orders of magnitude too large for the release rate.
**Strength:** strong.
**Must accompany:** cite the teams; do not imply we reproduced their result.

### C6 — Measured calibration discrepancy in the simplified retrieval
**Claim:** the single-coefficient absorption model commonly used in
educational implementations understates methane columns by 2.5–6.3×
(median 4.4×) relative to a production radiative-transfer LUT, with the
spread driven by solar zenith angle.
**Evidence:** `scripts/calibrate_alpha.py`, 42 geometries, cubic fit, worst
residual 1.4%; `config/rtm_calibration.json`.
**Strength:** strong, and useful to others building simplified chains.
**Must accompany:** the LUT sampled ships with `marss2l` (LGPL-3.0,
UNEP/IMEO) and must be cited wherever these numbers appear; underlying
physics is Varon et al. 2021 and Gorroño et al. 2023.

### C7 — Free-sensor observability is itself a finding
**Claim:** account-free access imposes measurable observability limits: the
public mirror lags ~12 days and carries no Sentinel-2C, which forced 30- and
50-day background baselines and left the Korpezhe target event observable
only on a scene that was 53% cloud and shadow.
**Evidence:** `docs/S2_REAL_DATA_FINDINGS.md`; CloudSEN12 screening;
per-scene coverage and cloud fractions recorded in every bundle.
**Strength:** strong; routes to the atlas under `ANALYSIS_PLAN` §7.

### C8 — Error found, corrected, and documented in the open
**Claim:** a 10⁴ input-scale error invalidated an earlier set of detections;
it was found by auditing fluxes against catalog rates, root-caused,
corrected, and the affected results formally withdrawn with a register and
machine-enforced quarantine.
**Evidence:** `docs/SUPERSEDED_RESULTS.md`;
`src/plumechaser/report/status.py`; `src/plumechaser/data/scale.py` and its
regression test.
**Strength:** strong. Present it as method, not apology — the audit trail is
the point, and the guard exists because band ratios are invariant to the
error, so no downstream signal could have revealed it.

---

## 2. Claims the evidence does NOT support

| # | Do not claim | Why |
|---|---|---|
| X1 | Any absolute emission rate in t/h as a measurement | Artifact floor ~150 t/h exceeds every targeted rate (C4). Rule 4 stands. |
| X2 | Agreement between our rates and catalog rates | Withdrawn as an endpoint, `ANALYSIS_PLAN` §2, amended 2026-08-25. |
| X3 | "Permian retrieval within 1.4× of catalog" | Retracted. Against a ~150 t/h floor and an 82 t/h catalog rate this is coincidence, not skill. |
| X4 | MARS-S2L detections at 0.996 / 0.995 at both targets | Withdrawn — produced under the input-scale error. Corrected: Permian holds at 0.994; Korpezhe is a no-detection. |
| X5 | `scene_score` as a reliable detection statistic | It read ≥0.996 on two metered zeros. |
| X6 | Any "first ever" | Standing project rule. The claim is "first fully open, account-free, validated replica plus observability atlas". |
| X7 | Basin-average retrieval noise | The measured σ values are single scenes, n=1 each. State n or do not state it. |
| X8 | "Sentinel-2 detection limit is 1–3 t/h" as our finding | That is a vendor/literature figure quoted in Sherwin 2023, not a measurement. Demonstrated floors are ~1.0–1.4 t/h, by other teams. |
| X9 | That our honesty gates prove our detections are correct | They prove withheld quantifications were correctly withheld. Detection accuracy is a separate claim (C2). |

---

## 3. Caveats required verbatim in outputs

* **Non-discrimination (FP protocol):** "Annotators are the system
  developers; κ quantifies inter-rater consistency, not expert accuracy."
* **Dual-mode interpretation:** "Strict mode benchmarks against operational
  human curation; lenient mode estimates the algorithmic ceiling; the
  strict–lenient gap is itself our quantification of curation effect."
* **Multi-source attribution sentence** when facility density exceeds the
  configured rule (`attribute/context.py`).
* **Accepted risks**, `ANALYSIS_PLAN` §10, disclosed permanently.

## 4. Prior art to cite, always

Schuit et al. 2023 (ACP) · Varon et al. 2021 (AMT 14, 2771) · Gorroño et al.
2023 (AMT 16, 89) · Lauvaux et al. 2022 (Science) · `marss2l` (LGPL-3.0,
UNEP/IMEO) · Sherwin et al. 2023 (Sci Rep 13:3836) and Sherwin et al. 2024
(AMT 17, 765) for the controlled releases · CloudSEN12 (`UNetMobV2_V2`) for
cloud screening · SRON Zenodo 13903869 (CC-BY-4.0) for training data.

## 5. What the students must produce themselves

Everything narrative. Specifically: the abstract, introduction, methods
prose, results prose, discussion, conclusions, poster text, and video script.
This register, the findings document and the figures are inputs to that
writing, not drafts of it. Every AI-assisted code contribution is logged
separately in the AI-use prompt log.
