"""Feature engineering for the SVC artifact filter (Schuit et al. 2023 step 2).

The exact feature list of Schuit et al. (Tables A1/C1) is only partially
published; we implement a documented approximation covering the four groups
they describe: scene morphology, retrieval quality, meteorology, geography.
Features are computed per 32x32 scene from the Zenodo channel set and feed a
binary RBF-SVC: plume vs {artefact, empty}.

Deterministic, dependency-light (numpy + scipy.ndimage).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from plumechaser.ml.dataset import normalize_scene

FEATURE_NAMES = [
    "valid_frac",
    "xch4_std",
    "xch4_ptp",
    "norm_max",
    "frac_high",
    "blob_frac_largest",
    "blob_aspect",
    "wind_speed_mean",
    "wind_coherence",
    "chi2_mean",
    "qa_min",
    "aot_swir_mean",
    "albedo_swir_mean",
    "cloud_frac_mean",
    "abs_lat_center",
]


def _blob_stats(norm: np.ndarray, valid: np.ndarray, high_mask: np.ndarray) -> tuple[float, float]:
    """Fraction of valid pixels inside the largest blob + its elongation."""
    if high_mask.sum() < 3:
        return float(high_mask.sum() / max(valid.sum(), 1)), 1.0
    labels, n = ndimage.label(high_mask)
    sizes = ndimage.sum_labels(np.ones_like(labels), labels, index=range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    frac = float(sizes.max() / max(valid.sum(), 1))
    ys, xs = np.nonzero(labels == largest)
    if ys.size < 3:
        return min(frac, 1.0), 1.0
    pts = np.stack([ys - ys.mean(), xs - xs.mean()])
    cov = np.nan_to_num(np.cov(pts), nan=0.0)
    eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eig = np.clip(eig, 1e-6, None)
    aspect = float(np.sqrt(eig[0] / eig[1]))
    return min(frac, 1.0), min(aspect, 10.0)


def _safe(x: float) -> float:
    """Feature sanitizer: any non-finite value becomes 0.0 (documented)."""
    return float(x) if np.isfinite(x) else 0.0


def scene_features(
    xch4: np.ndarray,
    *,
    albedo_swir: np.ndarray | None = None,
    aot_swir: np.ndarray | None = None,
    chi2: np.ndarray | None = None,
    qa_value: np.ndarray | None = None,
    cloud_frac: np.ndarray | None = None,
    u10: np.ndarray | None = None,
    v10: np.ndarray | None = None,
    lat_center: float = 0.0,
) -> list[float]:
    """Compute FEATURE_NAMES for one scene; missing context -> neutral zeros."""
    norm = normalize_scene(xch4)
    finite = np.isfinite(xch4)
    valid_frac = float(finite.mean())
    vals = xch4[finite]
    x_std = float(vals.std()) if vals.size else 0.0
    x_ptp = float(vals.max() - vals.min()) if vals.size else 0.0
    norm_v = norm[finite] if finite.any() else np.array([0.0])
    norm_max = float(norm_v.max())
    frac_high = float((norm > 0.6).sum() / max(finite.sum(), 1))

    blob_frac, blob_aspect = _blob_stats(
        norm, finite, (norm > 0.6) & finite
    )

    def _mean(x: np.ndarray | None) -> float:
        if x is None:
            return 0.0
        v = x[finite & np.isfinite(x)]
        return float(v.mean()) if v.size else 0.0

    if u10 is not None and v10 is not None:
        us, vs = _safe(np.nanmean(u10)), _safe(np.nanmean(v10))
        speed = _safe(np.nanmean(np.hypot(u10, v10)))
        coherence = float(np.hypot(us, vs) / speed) if speed > 1e-3 else 0.0
    else:
        speed, coherence = 0.0, 0.0

    qa_min = (
        _safe(np.nanmin(qa_value[finite & np.isfinite(qa_value)]))
        if qa_value is not None
        else 0.0
    )

    return [
        _safe(valid_frac),
        _safe(x_std),
        _safe(x_ptp),
        _safe(norm_max),
        _safe(frac_high),
        _safe(blob_frac),
        _safe(blob_aspect),
        _safe(speed),
        _safe(coherence),
        _mean(chi2),
        qa_min,
        _mean(aot_swir),
        _mean(albedo_swir),
        _mean(cloud_frac),
        _safe(abs(float(lat_center))),
    ]


def build_feature_table(
    nc_path,
    label_field: str = "manual_label",
) -> tuple[np.ndarray, np.ndarray]:
    """Load an SRON NetCDF and compute (X, y); y=1 iff manual_label == 'plume'."""
    import xarray as xr

    ds = xr.open_dataset(nc_path)
    y = (ds[label_field].values == "plume").astype(np.int32)

    lats = ds["latitude"].values
    rows = []
    for i in range(ds.sizes[ds.dims and list(ds.dims)[0]]):
        rows.append(
            scene_features(
                ds["xch4"].values[i],
                albedo_swir=ds["albedo_SWIR"].values[i],
                aot_swir=ds["aerosol_optical_thickness_SWIR"].values[i],
                chi2=ds["chi2"].values[i],
                qa_value=ds["qa_value"].values[i],
                cloud_frac=ds["pseudo_cloud_fraction"].values[i],
                u10=ds["windspeed_east_u10"].values[i],
                v10=ds["windspeed_north_v10"].values[i],
                lat_center=float(np.nanmean(lats[i])),
            )
        )
    ds.close()
    return np.asarray(rows, dtype=np.float64), y
