"""Training-data loading + preprocessing replicating Schuit et al. (2023).

Source data: SRON's manually labelled TROPOMI scenes (Zenodo record
13903869): CNN_pos_trainingdata.nc (828 plume scenes),
CNN_neg_trainingdata.nc (2242 clean scenes), SVC_trainingdata.nc (843
CNN-flagged scenes with expert plume/artefact/empty labels). All originate
from 2018-2020, which keeps validation against post-2024 catalogs free of
label circularity.

Scene normalisation (Schuit et al. 2023, section 2.1):
    * values below mean - 1 sd          -> 0
    * values above mean + 100 ppb - 1 sd -> 1
    * linear between those anchors
    * NaN/filtered pixels               -> 0
This preserves plume-shaped enhancements above local background while
removing absolute latitude/surface-altitude offsets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SCENE_SHAPE = (32, 32)


def load_scenes(nc_path: str | Path, channel: str | None = None) -> np.ndarray:
    """Load (N, 32, 32) CH4 scenes from a Zenodo/SRON NetCDF file.

    ``channel`` defaults to auto-detect ('xch4' in SRON releases, 'ch4'
    elsewhere). The files also carry 15 context channels (albedo_SWIR,
    aerosol_optical_thickness_SWIR, windspeed_*, qa_value, ...) used by the
    SVC features and pass-2 labeling panels.
    """
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - env guard
        raise ImportError("install the 'data' extra: pip install plumechaser[data]") from exc
    ds = xr.open_dataset(nc_path)
    if channel is None:
        channel = next((c for c in ("xch4", "ch4") if c in ds), None)
        if channel is None:
            raise KeyError(f"no xch4/ch4 variable in {Path(nc_path).name}: has {list(ds.data_vars)}")
    if channel not in ds:
        raise KeyError(f"channel '{channel}' not in {Path(nc_path).name}: has {list(ds.data_vars)}")
    arr = ds[channel].values.astype(np.float64)
    ds.close()
    if arr.ndim == 4:  # (N, 1, H, W) layout
        arr = arr[:, 0]
    if arr.shape[1:] != SCENE_SHAPE:
        raise ValueError(f"unexpected scene shape {arr.shape[1:]}")
    return arr


def normalize_scene(scene: np.ndarray) -> np.ndarray:
    """Schuit-style contrast normalisation of one (32, 32) XCH4 scene."""
    finite = scene[np.isfinite(scene)]
    if finite.size < 10:
        return np.zeros_like(scene, dtype=np.float64)
    mu = float(finite.mean())
    sd = float(finite.std())
    lo = mu - sd
    hi = mu + 100.0 - sd
    out = np.zeros_like(scene, dtype=np.float64)
    valid = np.isfinite(scene)
    clipped = np.clip(scene[valid], lo, hi)
    denom = hi - lo
    out[valid] = (clipped - lo) / denom if denom > 0 else 0.0
    return out


def normalize_stack(stack: np.ndarray) -> np.ndarray:
    """Vectorised :func:`normalize_scene` over axis 0."""
    return np.stack([normalize_scene(s) for s in stack], axis=0)


def build_training_arrays(
    pos_nc: str | Path,
    neg_nc: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) float32 arrays ready for CNN training; y=1 means plume."""
    pos = normalize_stack(load_scenes(pos_nc))
    neg = normalize_stack(load_scenes(neg_nc))
    x = np.concatenate([pos, neg], axis=0)[:, None, :, :].astype(np.float32)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.float32)
    return x, y
