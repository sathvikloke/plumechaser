# Detector Training Report — REAL DATA

**Date:** 2026-08-22 · **Data:** Zenodo 13903869 (828 pos / 2,242 neg scenes,
2018–2020 SRON v18_17) · **Split:** seeded 85/15 · **Protocol:** flip/rot
augmentation, pos_weight 2:1, early stopping on val loss, three seeds.

| Seed | Best epoch | Val loss | Precision | Recall |
|---|---|---|---|---|
| 0 | 59 | 0.172 | 0.966 | 0.918 |
| 1 | 47 | 0.254 | 0.888 | 0.841 |
| 2 | 58 | 0.223 | 0.858 | 0.934 |

**Seed variance:** precision 0.904 ± 0.045, recall 0.898 ± 0.041

## Reading (honest framing)

* These are HELD-OUT SCENES FROM THE TRAINING ERA (2018–2020) — they
  demonstrate the architecture learns the task; they are NOT project
  validation. Hindcast scoring happens post-freeze against 2025–26
  catalogs only (plan §1).
* Era-shift caveat (plan §R7): training product is SRON v18_17;
  inference will run on current operational L2. Expect degradation vs
  these numbers; the delta is itself a reported finding.
* No single lucky seed carries the result — spread is ±0.04–0.05.

## Reproduce

```bash
plumechaser train --pos data/zenodo/CNN_pos_trainingdata.nc \
                  --neg data/zenodo/CNN_neg_trainingdata.nc \
                  --out runs/detector --seeds 0 1 2
```

## Stage 2 — SVC artifact filter (real expert labels)

**Data:** `SVC_trainingdata.nc` — 843 CNN-flagged scenes manually labeled by
SRON experts (444 plume / 341 artefact / 58 empty), with 15 context channels.

**Features:** 15-dimension documented approximation of Schuit et al. Tables
A1/C1 across four groups — morphology (blob fraction/aspect, frac>0.6),
retrieval quality (χ², QA min, AOT-SWIR, albedo, cloud fraction),
meteorology (wind speed + directional coherence), geography (|lat|).
Approximation is disclosed; exact published feature list is incomplete.

**Model:** StandardScaler → CalibratedClassifierCV(RBF-SVC, sigmoid, cv=3);
C selected over {1, 10, 100} by stratified 5-fold CV F1 on the training split.

| C | CV F1 |
|---|---|
| 1 | 0.835 ± 0.029 |
| **10** | **0.848 ± 0.041** |
| 100 | 0.826 ± 0.030 |

Holdout (seed 0): precision **0.831**, recall **0.881**.

**Cascade verification:** on a mosaic of one real plume scene vs one real
artefact scene (with genuine context channels routed to the SVC), the
two-step detector fired only inside the plume zone and never on the artefact
scene (`tests/test_inference.py::test_real_cascade_with_trained_svc`).

Caveats: approximate features may underperform Schuit's exact set; era-shift
applies as in stage 1.
