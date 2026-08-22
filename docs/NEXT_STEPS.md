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
