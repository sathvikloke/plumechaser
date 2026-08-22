# Labeling Rubric — Two-Pass False-Positive Protocol

## Session rules
1. Raters work independently; no discussion until both finish a batch.
2. Batch = 25 scenes (~45 min). Shuffle order fixed by pack manifest.
3. Pass 1 is completed for the WHOLE batch before any pass-2 view opens.

## Pass 1 — morphology only (CH4 panel, no metadata)
Judge the normalized enhancement panel alone:

| Label | Meaning |
|---|---|
| `plume_like` | elongated/structured enhancement emanating from a compact origin |
| `not_plume` | diffuse blob, orbit stripe, edge artifact, or noise |
| `uncertain` | cannot tell |

## Pass 2 — full context → final label
Reveal albedo, aerosol, QA and wind arrow. Final labels: `real_plume`,
`artifact`, `uncertain`. Apply ALL four criteria:

a. **Connectivity** — enhancement is spatially connected to a plausible source pixel.
b. **Significance** — exceeds 3σ of local scene background (robust MAD).
c. **Adjacency exclusion** — not hugging coastlines, cloud edges, or known striping seams.
d. **Wind consistency** — orientation compatible with the wind vector shown.

Any criterion failing ⇒ `artifact`. Criteria conflict ⇒ `uncertain`
(adjudicated jointly later with written rationale logged).

## Controls
* ≥30 known-positive catalog scenes shuffled in, from regions never reviewed
  during development, geography stripped.
* Per-rater control hit-rate <85% ⇒ recalibration session + rubric revision
  before labeling continues.

## Metrics recorded per batch
single-rater FP rate · adjudicated FP rate · Cohen's κ between raters ·
control hit-rate. All four are reported in the paper — no exceptions.

> Standing caveat quoted in every output: annotators are the system
> developers; κ quantifies inter-rater consistency, not expert accuracy.
