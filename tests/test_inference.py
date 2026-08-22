"""Inference cascade tests — hermetic (no sklearn/torch/data required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plumechaser.detect.inference import run_detector, slide_windows
from plumechaser.ml.features import FEATURE_NAMES

REPO_ROOT = Path(__file__).resolve().parents[1]


def _mosaic(seed=3):
    """64x64: bright plume-like patch left half, flat background right half."""
    rng = np.random.default_rng(seed)
    grid = 1800.0 + rng.normal(0.0, 6.0, (64, 64))
    yy, xx = np.mgrid[0:64, 0:64]
    grid += 200.0 * np.exp(-(((yy - 20) ** 2) / 30 + ((xx - 16) ** 2) / 30))
    return grid


def _frac_high_scorer(threshold: float = 0.05):
    idx = FEATURE_NAMES.index("frac_high")

    def scorer(feats_row: np.ndarray) -> float:
        return 0.9 if feats_row[idx] > threshold else 0.1

    return scorer


def test_slide_windows_shapes():
    g = np.zeros((70, 80))
    wins = slide_windows(g, window=32, stride=16)
    assert len(wins) == ((70 - 32) // 16 + 1) * ((80 - 32) // 16 + 1)
    r0, c0, patch = wins[0]
    assert patch.shape == (32, 32)


def test_svc_requires_context_fn():
    with pytest.raises(ValueError, match="context_fn"):
        run_detector(_mosaic(), svc_scorer=lambda f: 0.9)


def test_cascade_fires_only_on_plume_zone():
    grid = _mosaic()
    dets = run_detector(
        grid,
        svc_scorer=_frac_high_scorer(),
        context_fn=lambda r0, c0, patch: {},
        svc_threshold=0.5,
        window=32,
        stride=16,
    )
    assert dets, "expected detections on a strong synthetic plume"
    for d in dets:
        # blob centroid sits near the injected Gaussian center
        assert abs(d.row - 20) < 8 and abs(d.col - 16) < 10


def test_flat_scene_yields_nothing():
    grid = np.full((64, 64), 1800.0)
    dets = run_detector(
        grid,
        svc_scorer=_frac_high_scorer(),
        context_fn=lambda *a: {},
    )
    assert dets == []


def test_real_cascade_with_trained_svc():
    """Integration: real SVC + real SRON scenes (skips without artifacts)."""
    pytest.importorskip("sklearn")
    joblib_path = REPO_ROOT / "runs" / "svc" / "svc.joblib"
    nc_path = REPO_ROOT / "data" / "zenodo" / "SVC_trainingdata.nc"
    if not (joblib_path.exists() and nc_path.exists()):
        pytest.skip("trained SVC or Zenodo data not present")

    import xarray as xr

    from plumechaser.ml.svc import load_svc, make_svc_scorer

    scorer = make_svc_scorer(load_svc(joblib_path))
    ds = xr.open_dataset(nc_path)
    pos_i = int(np.where(ds.manual_label.values == "plume")[0][0])
    neg_i = int(np.where(ds.manual_label.values == "artefact")[0][0])

    grid = np.full((64, 96), np.nan)
    grid[:32, :32] = ds.xch4.values[pos_i]
    grid[:32, 64:] = ds.xch4.values[neg_i]

    ctx = {}
    for key, var in [
        ("albedo_swir", "albedo_SWIR"),
        ("aot_swir", "aerosol_optical_thickness_SWIR"),
        ("chi2", "chi2"),
        ("qa_value", "qa_value"),
        ("cloud_frac", "pseudo_cloud_fraction"),
        ("u10", "windspeed_east_u10"),
        ("v10", "windspeed_north_v10"),
    ]:
        g = np.full((64, 96), np.nan)
        g[:32, :32] = ds[var].values[pos_i]
        g[:32, 64:] = ds[var].values[neg_i]
        ctx[key] = g

    def context_fn(r0, c0, patch):
        sl = (slice(r0, r0 + 32), slice(c0, c0 + 32))
        d = {k: v[sl] for k, v in ctx.items()}
        d["lat_center"] = float(np.nanmean(np.abs(ds.latitude.values[pos_i])))
        return d

    dets = run_detector(grid, svc_scorer=scorer, context_fn=context_fn)
    ds.close()
    assert all(d.col < 45 for d in dets), "no detection may fire in NEG/GAP zone"
