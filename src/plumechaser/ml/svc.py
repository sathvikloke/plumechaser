"""Two-step detector stage 2: RBF-SVC artifact filter on engineered features.

Trains a StandardScaler+SVC pipeline to separate expert-labeled plumes from
artifacts/empty scenes (Zenodo SVC_trainingdata.nc, 843 scenes). Reports
stratified holdout + 5-fold CV so reviewers see stability, and persists the
fitted pipeline with joblib for the inference cascade.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np


def train_svc(
    nc_path: str | Path,
    out_dir: str | Path,
    *,
    seed: int = 0,
    c_values: tuple[float, ...] = (1.0, 10.0, 100.0),
) -> dict:
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import precision_score, recall_score
        from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install scikit-learn") from exc

    from plumechaser.ml.features import build_feature_table

    x, y = build_feature_table(nc_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    x_tr, x_val, y_tr, y_val = train_test_split(
        x, y, test_size=0.15, stratify=y, random_state=seed
    )

    best: dict = {}
    for c in c_values:
        # sklearn>=1.9: probability calibration via CalibratedClassifierCV
        base_svc = SVC(kernel="rbf", C=c, gamma="scale", class_weight="balanced",
                       random_state=seed)
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("cal", CalibratedClassifierCV(base_svc, method="sigmoid", cv=3)),
            ]
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        f1s = cross_val_score(clf, x_tr, y_tr, cv=cv, scoring="f1")
        if not best or f1s.mean() > best["cv_f1_mean"]:
            best = {"C": c, "cv_f1_mean": float(f1s.mean()), "pipeline": clf}
        print(f"C={c:>6}: CV F1 = {f1s.mean():.3f} ± {f1s.std():.3f}")

    pipe = best.pop("pipeline")
    pipe.fit(x_tr, y_tr)
    pred = pipe.predict(x_val)
    metrics = {
        **best,
        "holdout_precision": float(precision_score(y_val, pred, zero_division=0)),
        "holdout_recall": float(recall_score(y_val, pred, zero_division=0)),
        "n_train": int(len(y_tr)),
        "n_holdout": int(len(y_val)),
        "class_balance_train": {
            "plume": int(y_tr.sum()),
            "not_plume": int((y_tr == 0).sum()),
        },
        "seed": seed,
    }

    import joblib

    joblib.dump(pipe, out / "svc.joblib")
    (out / "svc_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def load_svc(path: str | Path):
    """Load a fitted SVC pipeline."""
    import joblib

    return joblib.load(path)


def make_svc_scorer(pipeline) -> Callable[[np.ndarray], float]:
    """Adapt a fitted sklearn pipeline to the detector's scorer interface.

    The cascade calls ``scorer(feature_vector_row) -> float`` probability of
    the plume class; this adapter handles class-order lookup explicitly so a
    re-fitted model can never silently flip columns.
    """
    classes = list(pipeline.classes_)

    def scorer(feats: np.ndarray) -> float:
        proba = pipeline.predict_proba(np.atleast_2d(feats))[0]
        return float(proba[classes.index(1)] if 1 in classes else proba[-1])

    return scorer
