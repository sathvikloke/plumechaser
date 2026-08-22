"""Labeling support for the two-pass FP protocol (frozen plan section A2).

Pass 1: annotator sees ONLY the normalized CH4 panel -> morphology judgment.
Pass 2: full context (albedo, aerosol, wind arrow, QA) revealed; final label.

The render pack is generated ONCE at freeze and hashed; labels bind to the
pack hash so any later code change cannot silently alter what was judged.
A calibration control set of known-positive catalog scenes is shuffled into
the queue with geography stripped, to measure annotator sensitivity against
expert truth (<85% hit-rate triggers recalibration per plan).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_render_pack(
    scenes: dict[str, np.ndarray],
    out_dir: str | Path,
    controls: dict[str, np.ndarray] | None = None,
) -> Path:
    """Write standardized .npy panels + manifest for one labeling round.

    ``scenes`` maps candidate_id -> CH4 enhancement grid (pass-1 view).
    ``controls`` maps control_id -> KNOWN-plume scene mixed into the queue.
    Returns the pack directory (its SHA256 goes into the frozen manifest).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    all_scenes = {**scenes}
    if controls:
        all_scenes |= controls
    order = sorted(all_scenes)  # deterministic queue order
    rng = np.random.default_rng(20270307)
    rng.shuffle(order)

    for sid in order:
        arr = np.asarray(all_scenes[sid], dtype=np.float32)
        np.save(out / f"{sid}.npy", arr)
        entries.append({"id": sid, "is_control": bool(controls and sid in controls)})
    manifest = {"pack_version": 1, "n": len(entries), "entries": entries}
    blob = json.dumps(manifest, sort_keys=True).encode()
    manifest["pack_sha256"] = hashlib.sha256(blob).hexdigest()[:16]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


def collect_labels(labels_csv: str | Path) -> pd.DataFrame:
    """Load a filled labels CSV -> canonical columns id, pass1, pass2_final."""
    df = pd.read_csv(labels_csv)
    missing = {"id", "pass1_morphology", "pass2_final"} - set(df.columns)
    if missing:
        raise ValueError(f"labels CSV missing columns: {missing}")
    valid_p1 = {"plume_like", "not_plume", "uncertain"}
    valid_p2 = {"real_plume", "artifact", "uncertain"}
    bad1 = set(df["pass1_morphology"].dropna()) - valid_p1
    bad2 = set(df["pass2_final"].dropna()) - valid_p2
    if bad1 or bad2:
        raise ValueError(f"invalid labels found: p1={bad1} p2={bad2}")
    return df


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa between two raters' categorical labels (adjudication input).

    Convention: perfect agreement with a degenerate (single-category)
    expectation returns 1.0, not 0/0.
    """
    la, lb = pd.Series(a), pd.Series(b)
    cats = sorted(set(la.dropna()) | set(lb.dropna()))
    n = len(la)
    if n == 0 or not cats:
        raise ValueError("no labels to compare")
    po = float((la.values == lb.values).mean())
    pe = sum(float((la == c).sum()) / n * float((lb == c).sum()) / n for c in cats)
    if abs(1 - pe) < 1e-12:
        return 1.0 if po > 1 - 1e-12 else 0.0
    return (po - pe) / (1 - pe)


def control_hit_rate(
    labels: pd.DataFrame, manifest_path: str | Path, positive_label: str = "real_plume"
) -> float:
    """Fraction of known-positive controls labelled positive by an annotator."""
    manifest = json.loads(Path(manifest_path).read_text())
    controls = {e["id"] for e in manifest["entries"] if e.get("is_control")}
    sub = labels[labels["id"].isin(controls)]
    if sub.empty:
        raise ValueError("no control labels present")
    return float((sub["pass2_final"] == positive_label).mean())
