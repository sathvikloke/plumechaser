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
