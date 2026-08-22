# 🛰 PlumeChaser

**Open autonomous methane super-emitter monitoring.** TROPOMI screening →
automated cueing → Sentinel-2/Landsat MBMP retrieval → IME quantification →
infrastructure context — evaluated by a pre-registered agreement study
against public catalogs, with honest detection-limit mapping.

> Post-MethaneSAT (lost July 2025), global free monitoring capacity dropped.
> This project asks and answers quantitatively: *what can $0 satellites still
> see, where, and when?*

[![CI](https://github.com/sathvikloke/plumechaser/actions/workflows/ci.yml/badge.svg)](./actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-56%20passing-brightgreen.svg)]()

## Architecture

```
T0 ingest ──► T1 detect ──► T2 verify ──► T3 cue ──► T4 MBMP+IME ──► T5 context ──► T6 report
GEE/mirrors   climatology    persistence    policy     S2 B11/B12     FLIGHT/GEM      dossiers,
z-scores      + blobs        + QA gates     logged     retrieval      density rule    bundles
              (naive+CNN)                   w/ CFs     Monte Carlo CI
```

Every cue decision — including counterfactuals not taken — is appended to a
JSONL manifest. That log is the autonomy evidence.

## Science basis

| Component | Method | Reference |
|---|---|---|
| Detection training data | SRON labeled TROPOMI scenes (828+/2242−) | [Zenodo 13903869](https://zenodo.org/records/13903869); Schuit et al. 2023, ACP 23, 9071 |
| MBMP retrieval | multi-band multi-pass log-ratio differencing | Varon et al. 2021, AMT 14, 2771 |
| Quantification | IME, `Ueff = 0.33·U10 + 0.45` | Frankenberg 2016; Varon 2018/2021 |
| Detection limits | benchmarked S2 thresholds ~1–2 t/h (homogeneous), >5 t/h (heterogeneous) | Gorroño et al. 2023, AMT 16, 89 |
| Operational context | UN MARS tip-and-cue workflow | [UNEP IMEO](https://methanedata.unep.org) |

**Honest calibration note:** the MBMP absorption coefficients are
literature-seeded simplifications pending the RTM-LUT step; absolute rates
carry the IME method's documented 30–90% envelope until then. Structure,
artifact cancellation, masking, and uncertainty propagation are faithful.

## Quickstart

```bash
pip install -e ".[dev,data]"       # core science + I/O
plumechaser fetch-catalogs --url <sron_weekly_csv_url>
plumechaser train --pos CNN_pos_trainingdata.nc --neg CNN_neg_trainingdata.nc
plumechaser score --detections outputs/detections.csv --strict sron.csv
```

Optional extras: `gee` (Earth Engine screening), `cds` (ERA5 winds),
`ml` (PyTorch detector), `viz` (Streamlit replay dashboard).

Full operating procedure: **[docs/RUNBOOK.md](docs/RUNBOOK.md)**.
Pre-registered evaluation protocol: **[docs/ANALYSIS_PLAN.md](docs/ANALYSIS_PLAN.md)**.

## Repository map

```
src/plumechaser/
├── detect/      background climatology, QA gates, blobs, persistence
├── retrieve/    MBMP enhancement maps, plume masks, IME + Monte Carlo CI
├── ml/          Schuit-replica CNN dataset/model/training (3-seed protocol)
├── evaluate/    matching, cluster-bootstrap metrics, agreement orchestrator
├── cue/         reference-pass selector, persistent-first tasking policy
├── attribute/   infrastructure CONTEXT joins with honesty density rule
├── atlas/       analytic detection-limit surfaces (observability atlas)
├── labeling/    two-pass FP render packs, κ, control calibration
├── data/        mirrored fetches w/ SHA-256 manifests + schema drift guards
└── report/      evidence dossiers, offline replay dashboard
```

56 unit tests cover every scientific kernel against hand-computed values and
synthetic-plume recovery. CI runs lint + tests on Python 3.10 & 3.13.

## Ethics & data licensing

* **Code:** MIT.
* **Data:** providers keep their own terms — UNEP/Eye-on-Methane and SRON/CAMS
  catalogs are CC BY-NC-SA (credit required, non-commercial);
  Copernicus/NASA data under their open policies.
* **Publication model:** public tier coarsens coordinates to 0.01° with a
  ≥30-day delay and never names operators; full dossiers are shared on
  request; methodology is fully open. Attribution output is *infrastructure
  context*, never an accusation — above the density rule (>5 facilities /
  5 km) dossiers state individual attribution is unsupported.

## Roadmap

- [ ] RTM-derived LUT replacing simplified α coefficients
- [ ] Carbon Mapper API spot-check tier (EMIT/Tanager)
- [ ] Landsat fallback tier for cloud-blocked S2 windows
- [ ] Sentinel-5 (MetOp-SG) ingestion when CH4 products mature
