#!/usr/bin/env python
"""Generate paper-grade figures from REAL data already in the repo.

1. docs/figures/labeling_panel.png   - pass-1/pass-2 labeling demo panels
   built from a real SRON plume scene + its context channels.
2. docs/figures/atlas_limits.png     - analytic detection-limit curves vs
   REAL emission rates from the mirrored CAMS weekly catalog per basin.

Both are reproducible: python scripts/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plumechaser.atlas.limits import min_detectable_rate  # noqa: E402
from plumechaser.config import load_config  # noqa: E402
from plumechaser.data.sron_catalog import filter_events, load_weekly_csv  # noqa: E402
from plumechaser.ml.dataset import normalize_scene  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def newest_mirror() -> Path:
    files = sorted((REPO / "data" / "mirrors" / "sron_weekly").rglob("*.csv"))
    if not files:
        raise SystemExit("no mirrored catalog; run fetch-catalogs first")
    return files[-1]


def fig_labeling_panel() -> None:
    """Pass-1 (CH4 morphology) / Pass-2 (context) demo from a REAL scene."""
    import xarray as xr

    ds = xr.open_dataset(REPO / "data" / "zenodo" / "SVC_trainingdata.nc")
    i = int(np.where(ds.manual_label.values == "plume")[0][3])
    xch4 = ds.xch4.values[i]
    meta = {
        "albedo_SWIR": ("SWIR albedo", "viridis"),
        "aerosol_optical_thickness_SWIR": ("AOT SWIR", "magma"),
        "chi2": ("fit chi2", "cividis"),
        "pseudo_cloud_fraction": ("cloud fraction", "Blues"),
    }
    u10 = ds.windspeed_east_u10.values[i]
    v10 = ds.windspeed_north_v10.values[i]
    speed = float(np.nanmean(np.hypot(u10, v10)))
    wdir = float((270.0 - np.degrees(np.arctan2(-v10.mean(), -u10.mean()))) % 360.0)

    fig, axes = plt.subplots(1, 6, figsize=(17, 3.1))
    norm = normalize_scene(xch4)
    im0 = axes[0].imshow(norm, cmap="inferno")
    axes[0].set_title("PASS 1\nCH4 normalized (only view)", fontsize=9)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    qa = np.nan_to_num(ds.qa_value.values[i], nan=np.nanmin(ds.qa_value.values[i]))
    im1 = axes[1].imshow(qa, cmap="RdYlGn", vmin=0, vmax=1)
    axes[1].set_title(f"PASS 2 · QA (min {np.nanmin(qa):.2f})", fontsize=9)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    for ax, (var, (title, cmap)) in zip(axes[2:], meta.items(), strict=False):
        arr = ds[var].values[i]
        im = ax.imshow(np.nan_to_num(arr), cmap=cmap)
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)

    axes[5].annotate(
        f"wind\n{speed:.1f} m/s\nfrom {wdir:.0f}deg",
        xy=(0.5, 0.55), xycoords="axes fraction", ha="center", fontsize=11,
    )
    axes[5].arrow(0.5, 0.25, 0.30 * np.cos(np.radians(wdir)),
                  0.30 * np.sin(np.radians(wdir)), width=0.02, color="k")
    axes[5].set_axis_off()
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(
        f"Two-pass labeling protocol demo — SRON scene #{i} "
        f"(expert label: {ds.manual_label.values[i]})",
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(FIG / "labeling_panel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    ds.close()
    print("wrote docs/figures/labeling_panel.png")


def fig_atlas_limits() -> None:
    """Analytic Q_min curves vs REAL catalog rates for the champion basins."""
    cfg = load_config()
    df = load_weekly_csv(newest_mirror())
    hc = filter_events(df, date_range=("2025-01-01", None))

    classes = {"korpezhe": "homogeneous_arid", "permian": "heterogeneous"}
    colors = {"korpezhe": "#d62728", "permian": "#1f77b4"}
    sigma = {"homogeneous_arid": 12.0, "heterogeneous": 25.0}
    lengths = dict(cfg.raw["atlas"]["typical_plume_length_m"])
    px = int(cfg.tropomi.screening_pixel_size_m) ** 2
    k = float(cfg.raw["atlas"]["k_roc"])

    winds = {"korpezhe": np.linspace(2, 12, 60), "permian": np.linspace(2, 12, 60)}

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for basin, cls in classes.items():
        q = [
            min_detectable_rate(
                sigma_col_ppb=sigma[cls], k_sigma=k, min_pixels=cfg.tropomi.min_blob_pixels,
                pixel_area_m2=px, u10_ms=float(u),
                typical_plume_length_m=lengths[cls],
            )
            for u in winds[basin]
        ]
        ax.plot(winds[basin], q, color=colors[basin],
                label=f"{basin}: Q_min ({cls}, sigma={sigma[cls]} ppb)")
        sub = filter_events(hc, bbox=cfg.basins[basin].bbox).dropna(subset=["rate_t_h"])
        med_u = float(np.median(winds[basin]))
        np.asarray(q)
        if len(sub):
            # REAL event rates on the Y axis (t/h -> kg/h), positioned at each
            # basin's median climatological wind so they read against Q_min.
            med_u_basin = float(np.interp(med_u, [2, 12], [winds[basin][0], winds[basin][-1]]))
            rates_kgh = sub["rate_t_h"].to_numpy() * 1000.0
            ax.plot(np.full(len(sub), med_u_basin), rates_kgh,
                    "|", ms=11, mew=2, color=colors[basin], alpha=0.75,
                    label=f"{basin}: real events n={len(sub)} (at U={med_u_basin:.0f} m/s)")
            ax.axvline(med_u_basin, color=colors[basin], linestyle=":", alpha=0.35)

    ax.set_xlim(1, 13)
    ax.set_yscale("log")
    ax.set_xlabel("10 m wind speed U10 [m/s]")
    ax.set_ylabel("minimum detectable rate Q_min [kg/h]")
    ax.set_title(f"Free-sensor observability limits vs real detections\n"
                 f"(TROPOMI-tier screening grid, k={k:g} sigma, "
                 f"{int(cfg.tropomi.screening_pixel_size_m)} m pixels)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "atlas_limits.png", dpi=150)
    plt.close(fig)
    print("wrote docs/figures/atlas_limits.png")


if __name__ == "__main__":
    fig_labeling_panel()
    fig_atlas_limits()
