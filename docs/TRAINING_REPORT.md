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
