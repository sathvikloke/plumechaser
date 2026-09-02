# Superseded results register

Technical record of results this project has withdrawn or demoted, the reason,
and what replaced them. One row per affected bundle. Nothing listed here may be
quoted in a paper, poster, abstract, talk, or any other external artifact.

Root-cause analysis: `docs/S2_REAL_DATA_FINDINGS.md`, section
"Flux audit — 2026-08-25". Register opened 2026-09-01.

Status vocabulary:

| status | meaning |
|---|---|
| `superseded-withdrawn` | Result was published internally, is wrong, and is retracted. |
| `diagnostic` | Intermediate run kept to isolate one correction. Never a final result. |

---

## W-1 — MARS-S2L model-input scale error (10⁴)

**Withdrawn:** 2026-08-25 (audit closed, commit `3f75fe3`) ·
**Bundles marked:** 2026-09-01

### The withdrawn claim

> "UNEP's MARS-S2L production model DETECTED plumes at BOTH catalog targets
> with scene_score .996/.995."

Recorded in `docs/S2_REAL_DATA_FINDINGS.md` under "Production-model campaign
(marss2l 0.2.10, same day)":

| Event | Catalog rate | Withdrawn detection | Withdrawn scene_score | Withdrawn Q |
|---|---|---|---|---|
| Korpezhe 2026-08-05 | 26 t/h | is_plume=True | 0.996 | ≈337 t/h |
| Permian 2026-04-27 | 82 t/h | is_plume=True | 0.995 | ≈478 t/h |

### Cause

`plume_detection_model.predict` documents its inputs as TOA reflectance
multiplied by 10000 (DN), and the loaded MARS-S2L checkpoint has
`norm_data=False`, i.e. it normalises internally by `tensor /= 5000`.
`scripts/mars2l_demo.py` passed 0–1 reflectance, so every radiance channel
arrived ~10⁴× too small — effectively a constant image.

The model still produced output because it is `cat_mbmp=True`: the band-ratio
channel it prepends is scale-invariant and carried real signal. The network
therefore ran on the MBMP ratio alone with all radiance context zeroed, and
radiance context is what separates a plume from a bright surface artifact.
The result was hugely oversized, highly confident masks. Because
`Q = 3600 · U_eff · C · mean_ppb · √(plume area)`, the segmentation is what
set the flux, so the mask error propagated directly into the reported rate.

Three preprocessing defects compounded it in the same runs: partial-granule
backgrounds (43% of one window missing, fed in as −1000 DN), cross-platform /
cross-orbit background pairing, and no cloud screening at all (`validmask`
hardcoded all-ones).

### Corrected result

| Target | Corrected verdict | Bundle |
|---|---|---|
| Permian 2026-04-27 | Detection **holds** — `is_plume=True`, scene_score 0.994 | `EVT-20260427-P82-MARSS2L-v4` |
| Korpezhe 2026-08-05 | **No detection** — `is_plume=False`, scene_score 0.213 | `EVT-20260805-K26-MARSS2L-v6` |

Korpezhe v6 is **not a clean null for the event**. The only S2 scene the
anonymous GCS mirror offers near the event (S2B 2026-08-03, −2 d) is 53%
cloud + shadow by CloudSEN12. It is a scene-availability result and belongs in
the observability atlas as such. The "0.00% cloud" figure this campaign had
been quoting from STAC refers to the S2C scene of 2026-08-05, which the mirror
does not carry; it was never a property of the pixels used.

### Affected bundles

`bundles/` is gitignored, so these directories are local-only.

| Bundle | Status | Withdrawn values | Superseded by |
|---|---|---|---|
| `EVT-20260805-K26-MARSS2L` | `superseded-withdrawn` | is_plume=true, scene_score 0.9959, plume_px 60162, `q_output.Q` 888524.8 kg/h | `EVT-20260805-K26-MARSS2L-v6` |
| `EVT-20260805-K26-MARSS2L-v2` | `superseded-withdrawn` | is_plume=true, scene_score 0.9959, plume_px 60162, `q_output.Q` 336620.76 kg/h | `EVT-20260805-K26-MARSS2L-v6` |
| `EVT-20260427-P82-MARSS2L` | `superseded-withdrawn` | is_plume=true, scene_score 0.9946, plume_px 46668, `q_output.Q` 478172.03 kg/h | `EVT-20260427-P82-MARSS2L-v4` |
| `EVT-20260805-K26-MARSS2L-v3` | `diagnostic` | scene_score 0.9876 — DN scale fixed, background still 57% coverage, no cloud screen | `EVT-20260805-K26-MARSS2L-v6` |
| `EVT-20260805-K26-MARSS2L-v4` | `diagnostic` | scene_score 0.9911 — gates on valid pixels only, background still 57% coverage | `EVT-20260805-K26-MARSS2L-v6` |
| `EVT-20260805-K26-MARSS2L-v5` | `diagnostic` | scene_score 0.9985 — same-orbit 50 d background at 100% coverage, still no cloud screen | `EVT-20260805-K26-MARSS2L-v6` |
| `EVT-20260427-P82-MARSS2L-v2` | `diagnostic` | scene_score 0.8461 — DN scale fixed, background not coverage- or cloud-screened | `EVT-20260427-P82-MARSS2L-v4` |
| `EVT-20260427-P82-MARSS2L-v3` | `diagnostic` | scene_score 0.1045 — cloud screen added but background 83.9% cloud, valid_fraction 0.16 | `EVT-20260427-P82-MARSS2L-v4` |

Bundles were **not deleted**. They are the audit trail for a withdrawn result
and must survive to the freeze.

### How each bundle is marked

1. `SUPERSEDED.md` in the bundle directory — status, reason, superseding
   event id, and the specific withdrawn values.
2. `provenance.json` gains `superseded: true`, `result_status`,
   `superseded_by`, `superseded_reason`, `superseded_on`, `superseded_notes`,
   `do_not_quote: true`.
3. `dossier.html` gains a red DO NOT QUOTE banner at the top of `<body>`, and
   `[SUPERSEDED]` / `[DIAGNOSTIC ONLY]` is prefixed to both `<title>` and
   `<h1>` so the marking survives a screenshot or a print-to-PDF.
4. `scripts/make_figures.py` carries a committed denylist of the event ids
   (`WITHDRAWN_EVENT_IDS`, `DIAGNOSTIC_EVENT_IDS`) that wins over the on-disk
   flag, because `bundles/` is gitignored and the flag can be lost by a
   re-clone or by a re-run that overwrites the bundle.

### Integrity hashes

`bundle_integrity()` hashes every file in a bundle directory, so adding the
markers changed the hash of each marked bundle. Both values are recorded so
the lab notebook can be reconciled.

| Bundle | Integrity before marking | After marking |
|---|---|---|
| `EVT-20260805-K26-MARSS2L` | `91213a60a975ac20` | `787989bed6a2bb03` |
| `EVT-20260805-K26-MARSS2L-v2` | `2613bbc7dbf1672e` | `e593a09e535a65c6` |
| `EVT-20260805-K26-MARSS2L-v3` | `2b8a0c74f4b1625f` | `8f7fd26c38a024a0` |
| `EVT-20260805-K26-MARSS2L-v4` | `ba9d24cbed48fef6` | `efee930cfc4ceb1c` |
| `EVT-20260805-K26-MARSS2L-v5` | `86ad9236dc5f0d54` | `e47c5f2b0859bb5b` |
| `EVT-20260805-K26-MARSS2L-v6` | `68cf8f17b642a093` | unchanged |
| `EVT-20260427-P82-MARSS2L` | `86fff4b4952fdb5e` | `050295768add90b8` |
| `EVT-20260427-P82-MARSS2L-v2` | `13467c6cc7bb7c53` | `2216251c70eeb1a3` |
| `EVT-20260427-P82-MARSS2L-v3` | `fdbdf05566a1f610` | `a8cdeaf416612d35` |
| `EVT-20260427-P82-MARSS2L-v4` | `7ee456309e257523` | unchanged |

(First 16 hex characters of the SHA-256.)

---

## W-2 — Simplified-alpha ΔXCH₄ scale, ~5× low

**Demoted:** 2026-08-25 · **Status:** not withdrawn, but not quotable as an
absolute value

Sampling marss2l's RTM LUT (`TransmittanceCH4InterpolationFromDict`) near
ratio ≈ 1 gives an effective absorption coefficient of 1.4e-5 – 2.0e-5 per ppb
across SZA 24–45°. `config/default.yaml`'s simplified pair implies
`alpha_b12 − alpha_b11 = 9.0e-5 per ppb`, i.e. 4.4–6.3× more sensitive than
the production RTM.

Consequence: every ΔXCH₄ and σ_col the simplified chain has produced is
roughly 5× too small in ppb. The 70–208 ppb σ_col reported over the Korpezhe
dune margin in the earlier campaign corresponds to ~400–1,200 ppb on the RTM
scale. Affected bundles are the non-MARS-S2L ones
(`EVT-20260805-K26`, `-L1C`, `-L1C-v2`, `EVT-SYNTH-*`), whose provenance
already carries `"calibration": "simplified-alpha demo grade"`. They are not
marked superseded because the verdicts they record (all artifact-dominated,
quantification withheld) are unchanged by the rescale; only the ppb magnitudes
move.

Open consequence: the 80 ppb σ_col gate limit was calibrated on the
simplified-alpha scale and is now being applied to the RTM scale, where the
equivalent limit is roughly 350–500 ppb. Verdicts happen to be unchanged (every
corrected run exceeds even 500 ppb), but the threshold must be re-derived per
retrieval scale before the Nov 1 freeze.

---

## Standing rule

Absolute flux is unquotable project-wide while the flux audit is open
(project rule 4). This is not limited to the withdrawn bundles: the honesty
gates withhold quantification on **every** corrected run, including
`EVT-20260427-P82-MARSS2L-v4`, whose Q lands within 1.4× of the published
catalog rate. A `q_output` present in any bundle predates the application of
honesty gates to the production path.

`scripts/make_figures.py` enforces this for figures: bundles reach a figure
only through `quotable_bundles()`, and no figure renders an absolute flux as a
headline value. Run `python scripts/make_figures.py --bundle-status` to see the
current classification.
