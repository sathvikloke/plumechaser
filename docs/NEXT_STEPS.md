# NEXT STEPS — execution checklist

Design phase is closed (8 audit rounds, ~90 findings). Everything below is
execution. Total blocking time: ≈3 hours.

## Session A — unblock everything (<60 min)
- [x] ~~Download SRON weekly CSV~~ → mirrored 2026-08-22, 4,751 events, fp `b8db5cd2`
- [x] ~~Zenodo license verdict~~ → **CC-BY-4.0**, open
- [x] ~~marss2l license~~ → **LGPL-3.0** — clean-room path RETIRED
- [x] ~~Power calculation~~ → champion n_eff=57, branch **FULL**
      (docs/PRE_FREEZE_POWER_ASSESSMENT.md)
- [x] ~~Detector training~~ → 3 seeds on real Zenodo data, P .904±.045 /
      R .898±.041 (docs/TRAINING_REPORT.md)
- [ ] Register: Carbon Mapper portal; confirm GEE/Data Space/CDS approvals
- [ ] Look up regional fair date → declare Branch X or Y
- [ ] Email 3 external-reviewer candidates

## Session B — compliance (45 min)
- [ ] File research plan + sponsor forms BEFORE any project data runs
- [ ] Log prototyping to date as tutorial exercises on public datasets
- [ ] Start AI-use prompt log (this project's code = AI-assisted portions cited)

## Weekend C — skills baseline (≤3 hrs each, timed)
- [ ] Student A: reproduce Varon 2021 Hassi Messaoud panel via marss2l Colab
      or GEE band math. Pass = plume visible, orientation matches paper.
- [ ] Student B: GEE/pandas TROPOMI monthly composite over Korpezhe + QA +
      PNG/CSV export. Pass ≤2 hrs.

## Week D — ownership & budget
- [ ] Fill owner map (docs/RUNBOOK section table) with names tonight
- [ ] Family decision: $60 application fee + $600/participant contingency
- [ ] Passport validity ≥ Dec 2027 check

## Already done (do not redo)
- [x] Repo live: github.com/sathvikloke/plumechaser (CI green, 70 tests)
- [x] CAMS catalog mirrored (4,751 events); licenses verified
- [x] Power branch = FULL (n_eff 57); CNN + SVC trained on real SRON data
- [x] Real S2 campaigns: our chain (gates held) + marss2l production run
      — see docs/S2_REAL_DATA_FINDINGS.md
- [x] Flux audit CLOSED 2026-08-25: root cause was a 10^4 model-input scale
      error (reflectance vs DN) that stripped the segmentation model of all
      radiance context, plus partial-granule backgrounds, cross-orbit
      pairing and no cloud screening. Corrected Permian lands at 1.4x
      catalog; the ".996/.995 detections at both targets" claim is
      WITHDRAWN (Permian holds at .994; Korpezhe is no-detection because
      the only mirrored scene is 53% cloud).

## Now open (from the flux audit)
- [ ] Re-derive the σ_col gate per retrieval scale BEFORE Nov 1 freeze —
      80 ppb was calibrated on our simplified alpha, which is 4.4–6.3x more
      sensitive than the production RTM LUT it is now applied to
- [ ] Replace `mbmp.alpha_*` with an RTM-derived LUT (measured gap ~5x)
- [ ] Controlled-release cross-check (Ehrenberg/Casa Grande AZ, Sherwin
      et al. 2023 rates) to close absolute flux properly
- [ ] Still do not quote absolute t/h anywhere — gates withhold every run

## Then: phase gates
| Gate | Date | Deliverable |
|---|---|---|
| Pipeline MVP | Oct 20 | screening→candidates automated on Korpezhe |
| Freeze | Nov 1 | git tag + Zenodo DOI + environment.lock |
| Autonomy | Dec 15 | week of unattended runs |
| Results | Feb 1 | agreement tables + atlas draft |
| Content freeze | Mar 1 | paper/poster/video locked |

Rule that outranks all others: when blocked >30 minutes, post the blocker,
don't redesign around it.
