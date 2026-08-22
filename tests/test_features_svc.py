"""Feature engineering + SVC adapter tests (hermetic unless data present)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plumechaser.ml.features import FEATURE_NAMES, scene_features

REPO_ROOT = Path(__file__).resolve().parents[1]
SVC_NC = REPO_ROOT / "data" / "zenodo" / "SVC_trainingdata.nc"


def _scene(seed=0, plume=True):
    rng = np.random.default_rng(seed)
    x = 1800.0 + rng.normal(0.0, 8.0, (32, 32))
    if plume:
        yy, xx = np.mgrid[0:32, 0:32]
        x += 120.0 * np.exp(-(((yy - 14) ** 2) / 20 + ((xx - 18) ** 2) / 20))
    return x


def test_feature_vector_shape_and_finiteness():
    feats = scene_features(_scene())
    assert len(feats) == len(FEATURE_NAMES)
    assert all(np.isfinite(feats))
    rng = np.random.default_rng(1)
    feats_ctx = scene_features(
        _scene(1),
        albedo_swir=np.full((32, 32), 0.2),
        chi2=np.full((32, 32), 1.0),
        qa_value=np.full((32, 32), 0.7),
        cloud_frac=rng.random((32, 32)) * 0.1,
        u10=rng.normal(5.0, 1.0, (32, 32)),
        v10=rng.normal(-1.0, 0.5, (32, 32)),
        lat_center=39.0,
    )
    assert all(np.isfinite(feats_ctx))


def test_plume_scene_scores_higher_on_morphology():
    f_plume = dict(zip(FEATURE_NAMES, scene_features(_scene(plume=True)), strict=False))
    f_flat = dict(zip(FEATURE_NAMES, scene_features(_scene(plume=False)), strict=False))
    assert f_plume["norm_max"] > f_flat["norm_max"]
    assert f_plume["frac_high"] > f_flat["frac_high"]


def test_all_nan_scene_yields_zeros_not_nan():
    feats = dict(zip(FEATURE_NAMES, scene_features(np.full((32, 32), np.nan)), strict=False))
    assert feats["valid_frac"] == 0.0
    assert all(np.isfinite(list(feats.values())))


def test_real_feature_table_shapes():
    if not SVC_NC.exists():
        pytest.skip("Zenodo SVC file not downloaded")
    from plumechaser.ml.features import build_feature_table

    x, y = build_feature_table(SVC_NC)
    assert x.shape == (843, len(FEATURE_NAMES))
    assert int(y.sum()) == 444
    assert np.isfinite(x).all()
