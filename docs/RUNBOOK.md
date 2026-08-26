# Runbook — operating PlumeChaser end to end

## 0. One-time setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,data]"          # core + netCDF/xarray
pip install -e ".[gee]"               # after GEE noncommercial approval
earthengine authenticate              # browser flow
pip install -e ".[cds]"               # ERA5 winds; put key in ~/.cdsapirc
```

Accounts needed: Google Earth Engine (noncommercial), Copernicus Data Space,
NASA Earthdata, Copernicus CDS, Carbon Mapper portal. Approvals take days —
register first.

## 1. Mirror catalogs (T0)

```bash
plumechaser fetch-catalogs --url <SRON/CAMS weekly CSV url>
```

Every download lands in `data/mirrors/<source>/<stamp>_<hash8>/` and is
logged to `data/manifests/<source>.manifest.jsonl` with SHA-256. Schema
drift raises `SchemaDriftError`; acknowledge only after updating the parser.

## 2. Train the detector

Download the three Zenodo files (record 13903869) into `data/zenodo/`, then:

```bash
plumechaser train --pos data/zenodo/CNN_pos_trainingdata.nc \
                  --neg data/zenodo/CNN_neg_trainingdata.nc \
                  --out runs/detector --seeds 0 1 2
```

Requires the `ml` extra. `seed_summary.json` reports three-seed variance —
this is what reviewers see, not a single lucky run.

## 3. Screening loop (daily GEE export → local scoring)

```bash
plumechaser screen --basin korpezhe --date $(date +%F)
```

The GEE task exports the day frame; client-side scoring uses
`detect.background.rolling_background` on the synced stack (identical math,
unit-tested). Candidates then pass `qa_mask`, blob extraction, and
`persist_candidates` before any cue decision.

## 4. Cue decisions

Verified candidates go through `cue.policy.decide_cue`. EVERY decision —
including counterfactuals and drops — appends to the cue manifest JSONL.
That manifest is the autonomy evidence shown at judging.

## 5. MBMP + IME quantification (per cued event)

Reference selection via `cue.reference.select_reference_pass` (frozen rules:
±12 d window, ±2 d exclusion, cloud ≤5%, surface corr ≥0.95, no own-plume).
Then:

* `retrieve.mbmp.mpmb_enhancement_ppb`
* `retrieve.mbmp.plume_mask`
* `retrieve.ime.quantitate` (Monte Carlo CI; seed from config)

Events without a clean reference stay **DETECTION-ONLY** by rule.

## 6. Attribution context

`attribute.context.infrastructure_context` joins FLIGHT/GEM/OSM layers.
Density >5 facilities per 5 km radius ⇒ the dossier prints the multi-source
sentence verbatim. We never name operators on the public tier.

## 7. Agreement study

```bash
plumechaser score --detections outputs/detections.csv \
                  --strict data/mirrors/sron_weekly/latest.csv \
                  [--lenient data/zenodo/schuit2021_set.csv]
```

Implements frozen-plan sections 2–5 including sensitivity grid and branch
assignment. Labels are opened exactly once, here.

## 8. Replay bundles (fair-safe demo)

```
bundles/EVT-2025-001/
    provenance.json     # config sha256, code commit, data hashes, quant dict
    tropomi.png         # screening panel
    mbmp.png            # S2 ΔXCH4 + mask overlay
    dossier.html        # rendered evidence dossier
```

Dashboard reads bundles ONLY (`plumechaser dashboard`) — zero live APIs on
the fair floor, offline laptop + USB copies + printed fallback pack.

## 9. Freeze ceremony (Nov 1)

1. Final commit; tag `freeze-2026-11-01`.
2. Export `.venv` package versions to `runs/environment.lock`.
3. Deposit `docs/ANALYSIS_PLAN.md` + `config/default.yaml` + lock file to
   Zenodo (restricted access), record DOI in lab notebook.
4. From this moment: catalog files are read only inside step 7.

## 10. Real-data campaign scripts (no accounts needed)

Two Python environments exist:
* `.venv/` — the plumechaser package itself (`pip install -e ".[dev,data]"`)
* `.venv-mars/` — UNEP's production package: `pip install marss2l`
  (LGPL; pulls torch/segmentation-models; ~1 GB)

| Script | Purpose | Env |
|---|---|---|
| `scripts/power_assessment.py` | branch assignment from mirrored catalog | .venv |
| `scripts/real_s2_demo.py` | our simplified MBMP chain on real S2 pixels (GCS L1C / AWS L2A), honesty gates, Gate-B injection | .venv |
| `scripts/mars2l_demo.py` | UNEP MARS-S2L production inference on same events (detection + flux; honesty-gated) | .venv-mars |
| `scripts/flux_audit.py` | offline decomposition of every recorded flux into mean-enhancement × sqrt(area) | .venv |
| `scripts/make_figures.py` | paper figures from real data | .venv |
| `scripts/freeze.py` | freeze ceremony (dry-run first!) | .venv |
| `scripts/smoke_check.py` | 10-second synthetic sanity run | .venv |

Campaign findings + open flux-audit items:
`docs/S2_REAL_DATA_FINDINGS.md` (read before quoting any rate).

### Input conventions that have already cost us a campaign

* MARS-S2L wants **DN = TOA reflectance × 10000**, not 0–1 reflectance.
  Its MBMP channel is scale-invariant, so a wrong scale does not crash and
  does not even look wrong — it silently strips the model's radiance
  context and inflates masks.
* Background scenes must be **same platform, same relative orbit (10-day
  multiples), fully covering the window, and cloud-screened**. One MGRS
  tile/day can hold several SAFEs; `safes[0]` is not safe.
* L1C `RADIO_ADD_OFFSET` has **13** band_ids including B10:
  B10=10, **B11=11, B12=12**. The L2A ordering (B11=10, B12=11) is wrong
  for the L1C products the GCS mirror serves.
