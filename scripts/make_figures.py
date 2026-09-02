#!/usr/bin/env python
"""Generate paper-grade figures from REAL data already in the repo.

1. docs/figures/labeling_panel.png   - pass-1/pass-2 labeling demo panels
   built from a real SRON plume scene + its context channels.
2. docs/figures/atlas_limits.png     - analytic detection-limit curves vs
   REAL emission rates from the mirrored CAMS weekly catalog per basin.

Both are reproducible: python scripts/make_figures.py

Two rules bind every figure in this file
----------------------------------------
1. **Superseded bundles are never plotted.** Any figure that touches
   ``bundles/`` MUST source its directories from :func:`quotable_bundles`,
   which drops withdrawn and diagnostic-only runs. Reading a bundle by hand
   is a bug: the 2026-08-25 flux audit withdrew scene scores and fluxes that
   are still physically present in those directories, and a withdrawn number
   that resurfaces on a poster is a scientific-integrity failure. If a figure
   legitimately needs the audit trail, pass ``include_superseded=True`` and
   label every affected element WITHDRAWN / SUPERSEDED on the figure itself.
2. **No absolute flux as a headline value** (project rule 4). PlumeChaser's
   own retrieved fluxes stay unquotable while the flux audit is open -- the
   honesty gates withhold quantification on every corrected run. Rates that
   do appear on a figure must be third-party published catalog values, or an
   analytic detection *limit*, and must be labelled as such.

See docs/SUPERSEDED_RESULTS.md and docs/S2_REAL_DATA_FINDINGS.md
("Flux audit - 2026-08-25").
"""

from __future__ import annotations

import argparse
import json
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
from plumechaser.report.status import (  # noqa: E402
    DIAGNOSTIC_EVENT_IDS as _DIAGNOSTIC_IDS,
)
from plumechaser.report.status import (  # noqa: E402
    WITHDRAWN_EVENT_IDS as _WITHDRAWN_IDS,
)

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "docs" / "figures"
BUNDLES = REPO / "bundles"

# Committed fallback denylist. bundles/ is gitignored, so the "superseded"
# flag inside provenance.json can be lost by a re-clone, a re-run that
# overwrites the bundle, or a copy from an old machine. These event ids are
# withdrawn regardless of what any provenance.json on disk says.
#
# The ids themselves come from plumechaser.report.status, which the replay
# dashboard and the flux auditor read too -- one authoritative list, so the
# three consumers cannot drift apart. Only the reason strings live here.
# Keep in sync with docs/SUPERSEDED_RESULTS.md.
WITHDRAWN_EVENT_IDS = {
    eid: "MARS-S2L run at reflectance scale; withdrawn 2026-08-25"
    for eid in _WITHDRAWN_IDS
}
DIAGNOSTIC_EVENT_IDS = {
    eid: "intermediate flux-audit diagnostic" for eid in _DIAGNOSTIC_IDS
}

FLUX_WITHHELD_NOTE = (
    "PlumeChaser retrieved fluxes are not shown: absolute flux is WITHHELD "
    "while the flux audit is open (honesty gates withhold every run).\n"
    "Event markers are third-party published CAMS/SRON catalog rates, not "
    "our measurements."
)


def newest_mirror() -> Path:
    files = sorted((REPO / "data" / "mirrors" / "sron_weekly").rglob("*.csv"))
    if not files:
        raise SystemExit("no mirrored catalog; run fetch-catalogs first")
    return files[-1]


def bundle_status(bdir: Path) -> tuple[str, str]:
    """Classify a bundle directory as ``ok`` / ``superseded`` / ``diagnostic``.

    The committed denylists above win over the on-disk flag, so a bundle whose
    provenance.json lost its marker is still caught.
    """
    if bdir.name in WITHDRAWN_EVENT_IDS:
        return "superseded", WITHDRAWN_EVENT_IDS[bdir.name]
    if bdir.name in DIAGNOSTIC_EVENT_IDS:
        return "diagnostic", DIAGNOSTIC_EVENT_IDS[bdir.name]
    if (bdir / "SUPERSEDED.md").exists():
        return "superseded", "SUPERSEDED.md present"
    prov = bdir / "provenance.json"
    if prov.exists():
        try:
            meta = json.loads(prov.read_text())
        except (OSError, json.JSONDecodeError):
            return "superseded", "unreadable provenance.json - treated as unquotable"
        if meta.get("superseded") or meta.get("do_not_quote"):
            status = meta.get("result_status", "superseded")
            status = "diagnostic" if status == "diagnostic" else "superseded"
            by = meta.get("superseded_by", "?")
            return status, f"provenance flag; superseded by {by}"
    return "ok", ""


def quotable_bundles(root: Path = BUNDLES, *, include_superseded: bool = False) -> list[Path]:
    """Bundle directories a figure is allowed to plot.

    The ONLY sanctioned way for a figure in this file to reach ``bundles/``.
    With ``include_superseded=True`` the caller takes responsibility for
    labelling every superseded element WITHDRAWN on the figure itself.
    """
    if not root.exists():
        return []
    dirs = sorted(d for d in root.iterdir() if d.is_dir() and (d / "provenance.json").exists())
    if include_superseded:
        return dirs
    return [d for d in dirs if bundle_status(d)[0] == "ok"]


def print_bundle_status(root: Path = BUNDLES) -> None:
    """Show which bundles figures may quote and which are barred, with reasons."""
    dirs = quotable_bundles(root, include_superseded=True)
    if not dirs:
        print(f"no bundles under {root}")
        return
    print(f"{'bundle':<34}{'status':<12}reason")
    print("-" * 92)
    for d in dirs:
        status, reason = bundle_status(d)
        print(f"{d.name:<34}{status:<12}{reason}")
    barred = [d for d in dirs if bundle_status(d)[0] != "ok"]
    print(f"\n{len(dirs) - len(barred)} quotable / {len(barred)} barred from figures.")
    print("Absolute flux stays unquotable for ALL of them (project rule 4).")


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
    """Analytic Q_min curves vs published catalog rates for the champion basins.

    Nothing here is a PlumeChaser flux measurement: the curves are an analytic
    detection *limit* and the markers are published CAMS/SRON catalog rates.
    See rule 2 in the module docstring.
    """
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
            # PUBLISHED catalog rates on the Y axis (t/h -> kg/h), positioned at
            # each basin's median climatological wind so they read against Q_min.
            # These are CAMS/SRON's numbers, not a PlumeChaser retrieval — our
            # own absolute fluxes are withheld under the open flux audit.
            med_u_basin = float(np.interp(med_u, [2, 12], [winds[basin][0], winds[basin][-1]]))
            rates_kgh = sub["rate_t_h"].to_numpy() * 1000.0
            ax.plot(np.full(len(sub), med_u_basin), rates_kgh,
                    "|", ms=11, mew=2, color=colors[basin], alpha=0.75,
                    label=f"{basin}: published CAMS/SRON catalog rates n={len(sub)} "
                          f"(at U={med_u_basin:.0f} m/s)")
            ax.axvline(med_u_basin, color=colors[basin], linestyle=":", alpha=0.35)

    ax.set_xlim(1, 13)
    ax.set_yscale("log")
    ax.set_xlabel("10 m wind speed U10 [m/s]")
    ax.set_ylabel("emission rate [kg/h] — detection LIMIT (curves)\n"
                  "/ published catalog rate (markers)")
    ax.set_title(f"Free-sensor observability limits vs published catalog rates\n"
                 f"(TROPOMI-tier screening grid, k={k:g} sigma, "
                 f"{int(cfg.tropomi.screening_pixel_size_m)} m pixels)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.text(0.01, -0.02, FLUX_WITHHELD_NOTE, fontsize=7, color="#7f1d1d", va="top")
    fig.tight_layout()
    fig.savefig(FIG / "atlas_limits.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote docs/figures/atlas_limits.png")


FIGURES = {
    "labeling": fig_labeling_panel,
    "atlas": fig_atlas_limits,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--figure", choices=[*FIGURES, "all"], default="all",
        help="which figure to render (default: all)",
    )
    ap.add_argument(
        "--bundle-status", action="store_true",
        help="list which bundles figures may quote, and why the rest are barred, "
             "then exit without rendering",
    )
    args = ap.parse_args(argv)

    if args.bundle_status:
        print_bundle_status()
        return 0

    FIG.mkdir(parents=True, exist_ok=True)
    names = list(FIGURES) if args.figure == "all" else [args.figure]
    for name in names:
        FIGURES[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
