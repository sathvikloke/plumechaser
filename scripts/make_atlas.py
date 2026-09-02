#!/usr/bin/env python
"""Build the free-sensor observability atlas from MEASURED anchors.

    python scripts/make_atlas.py            # figures + JSON into docs/figures
    python scripts/make_atlas.py --help

Outputs
-------
docs/figures/atlas_observability.png   the atlas: analytic noise-limited Q_min
                                       against the two empirical floors
docs/figures/atlas_noise_anchors.png   what the surface is anchored on --
                                       measured sigma vs the assumed sigma it
                                       replaces, and the Q_min each implies
docs/figures/atlas_summary.json        every number on both figures, machine
                                       readable, with its provenance

What this asserts, and who measured what
----------------------------------------
MEASURED BY US
  * band-ratio noise sigma_log_ratio -- config atlas.measured_sigma_log_ratio,
    three single scenes, calibration-independent
  * the RTM ppb-per-log-ratio curve -- config/rtm_calibration.json
  * the ARTIFACT FLOOR: the spurious flux this chain reports on scenes metered
    at zero emission. This is a LIMITATION, never an emission (project rule 4)

MEASURED BY OTHERS, PUBLISHED
  * the community detection floor, 1.0-1.4 t/h, demonstrated in the
    Stanford/EDF single-blind tests (Sherwin et al. 2023, 2024)
  * catalog emission rates for the champion basins (CAMS/SRON weekly)

ASSUMED (plotted only for contrast, always labelled)
  * the 12/25 ppb literature sigma_col the atlas used before measurements existed

DERIVED
  * Q_min, from measured sigma at each scene's real solar geometry

Re-runnability
--------------
Nothing about a specific scene is hardcoded. Scenes come from the config keys,
their geometry from whichever bundle matches that key, the floors from the
controlled-release truth file plus the bundles it points at. Add scenes to
``atlas.measured_sigma_log_ratio`` and re-run this file unchanged.

Two rules bind this script, as they bind scripts/make_figures.py
----------------------------------------------------------------
1. No PlumeChaser absolute flux as a headline number. The ~150 t/h floor here is
   plotted as OUR ARTIFACT FLOOR -- a limitation -- and the label carrying that
   wording is built in ``atlas.limits.our_artifact_floor``, not here, so it
   cannot be dropped at the drawing stage.
2. Provisional n is stated ON the figure, not only in a caption. Every sigma is
   one scene.

Figures are readable in grayscale print: meaning is carried by line style,
marker, hatch and direct text, never by colour alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plumechaser.atlas.limits import (  # noqa: E402
    PROVISIONAL_N_NOTE,
    S2_MIN_BLOB_PIXELS,
    community_detection_floor,
    load_measured_scenes,
    our_artifact_floor,
    qmin_curve,
)
from plumechaser.config import load_config  # noqa: E402
from plumechaser.retrieve.calibration import load_calibration  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

#: The sigma_col the atlas assumed before anything was measured. Literature
#: values for favourable surfaces; kept only as the contrast that shows how far
#: off an assumption can be. Never used to compute a claimed limit.
ASSUMED_SIGMA_COL_PPB = {"homogeneous_arid": 12.0, "heterogeneous": 25.0}

#: Sensitivity fan for the S2 minimum-blob choice (documented default is 25 px;
#: see atlas.limits.S2_MIN_BLOB_PIXELS for why it is not a config key).
MIN_PIXEL_SENSITIVITY = (9, 25, 100)

#: Grayscale-safe styling: distinct dash patterns and markers, mid-grays only.
CURVE_STYLES = (
    {"color": "0.00", "linestyle": "-", "marker": "o"},
    {"color": "0.40", "linestyle": "--", "marker": "s"},
    {"color": "0.25", "linestyle": "-.", "marker": "^"},
    {"color": "0.55", "linestyle": (0, (1, 1)), "marker": "D"},
    {"color": "0.15", "linestyle": (0, (5, 1, 1, 1, 1, 1)), "marker": "v"},
)


def _style(i: int) -> dict:
    return CURVE_STYLES[i % len(CURVE_STYLES)]


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


def build(args) -> dict:
    """Compute every number the atlas shows. No plotting, no I/O side effects."""
    cfg = load_config(args.config)
    cal = load_calibration(args.calibration)
    atlas_cfg = cfg.raw.get("atlas", {})
    k_roc = float(atlas_cfg.get("k_roc", 3.0))
    lengths = dict(atlas_cfg.get("typical_plume_length_m", {}))
    pixel_size = int(cfg.sentinel2.pixel_size_m)

    scenes = load_measured_scenes(cfg, bundles_dir=args.bundles)
    if not scenes:
        raise SystemExit(
            "config atlas.measured_sigma_log_ratio is empty — nothing to anchor on"
        )

    community = community_detection_floor(args.truth)
    artifact = our_artifact_floor(args.bundles, args.truth)

    records = []
    for scene in scenes:
        length = float(lengths.get(scene.surface_class, 1000.0))
        sigma_ppb = scene.sigma_col_ppb(cal)
        assumed = ASSUMED_SIGMA_COL_PPB.get(scene.surface_class)
        q_at_own_wind = float(
            qmin_curve(
                scene,
                [scene.u10_ms],
                cal,
                k_sigma=k_roc,
                min_pixels=args.min_pixels,
                pixel_size_m=pixel_size,
                typical_plume_length_m=length,
                ueff_slope=cfg.ime.ueff_slope,
                ueff_intercept=cfg.ime.ueff_intercept,
            )[0]
        )
        records.append(
            {
                "key": scene.key,
                "label": scene.label(),
                "basin": scene.basin,
                "surface_class": scene.surface_class,
                "date": scene.day.isoformat(),
                "n_scenes": scene.n_scenes,
                "provisional": scene.provisional,
                "source_run": scene.event_id,
                "source_run_status": scene.bundle_result_status,
                "geometry_source": scene.geometry_source,
                "satellite": scene.satellite,
                "sza_deg": round(scene.sza_deg, 2),
                "vza_deg": scene.vza_deg,
                "u10_ms": scene.u10_ms,
                "surface_pressure_hpa": scene.surface_pressure_hpa,
                "typical_plume_length_m": length,
                "sigma_log_ratio_measured": scene.sigma_log_ratio,
                "sigma_col_ppb_measured_at_geometry": round(sigma_ppb, 1),
                "sigma_col_ppb_recorded_by_gates": scene.recorded_sigma_col_ppb,
                "sigma_col_ppb_assumed_literature": assumed,
                "qmin_t_h_at_own_wind": round(q_at_own_wind / 1000.0, 3),
                "qmin_t_h_by_min_pixels": {
                    str(n): round(
                        float(
                            qmin_curve(
                                scene,
                                [scene.u10_ms],
                                cal,
                                k_sigma=k_roc,
                                min_pixels=n,
                                pixel_size_m=pixel_size,
                                typical_plume_length_m=length,
                                ueff_slope=cfg.ime.ueff_slope,
                                ueff_intercept=cfg.ime.ueff_intercept,
                            )[0]
                        )
                        / 1000.0,
                        3,
                    )
                    for n in MIN_PIXEL_SENSITIVITY
                },
                "_scene": scene,
                "_length_m": length,
            }
        )

    targets = _target_rates(cfg, args.catalog, args.truth)
    return {
        "cfg": cfg,
        "cal": cal,
        "k_roc": k_roc,
        "pixel_size_m": pixel_size,
        "min_pixels": args.min_pixels,
        "records": records,
        "community": community,
        "artifact": artifact,
        "targets": targets,
    }


def _target_rates(cfg, catalog_arg: str | None, truth_path: str) -> dict:
    """Emission rates this study targets. All published by other people."""
    truth = json.loads(Path(truth_path).read_text())
    metered = sorted(
        float(o["kg_h"]) / 1000.0
        for c in truth.get("campaigns", [])
        for o in c.get("overpasses", [])
        if o.get("kind") == "release" and float(o.get("kg_h", 0.0)) > 0.0
    )
    out = {
        "metered_controlled_releases_t_h": {
            "low": metered[0],
            "high": metered[-1],
            "n": len(metered),
            "attribution": "Stanford/EDF metered truth (Sherwin et al. 2023/2024)",
        },
        "catalog_by_basin": {},
    }

    path = Path(catalog_arg) if catalog_arg else _newest_catalog()
    if path is None or not path.exists():
        return out
    try:
        from plumechaser.data.sron_catalog import filter_events, load_weekly_csv

        events = load_weekly_csv(path)
    except Exception as exc:  # noqa: BLE001 - a missing catalog must not abort
        print(f"note: catalog layer skipped ({exc})")
        return out

    for name, basin in cfg.basins.items():
        sub = filter_events(events, bbox=basin.bbox).dropna(subset=["rate_t_h"])
        if not len(sub):
            continue
        rates = sub["rate_t_h"].to_numpy(dtype=float)
        out["catalog_by_basin"][name] = {
            "n": int(len(rates)),
            "min_t_h": float(np.min(rates)),
            "median_t_h": float(np.median(rates)),
            "max_t_h": float(np.max(rates)),
            "attribution": f"published CAMS/SRON weekly catalog ({path.name})",
        }
    return out


def _newest_catalog() -> Path | None:
    found = sorted((REPO / "data" / "mirrors" / "sron_weekly").rglob("*.csv"))
    return found[-1] if found else None


# --------------------------------------------------------------------------
# figure 1 -- the atlas
# --------------------------------------------------------------------------


def _band(ax, low, high, *, face, hatch, edge, zorder, rule_lw=0.0, horizontal=True):
    span = ax.axhspan if horizontal else ax.axvspan
    rule = ax.axhline if horizontal else ax.axvline
    span(low, high, facecolor=face, hatch=hatch, edgecolor=edge, linewidth=0.0,
         zorder=zorder)
    if rule_lw:
        for value in (low, high):
            rule(value, color="black", linewidth=rule_lw, zorder=zorder + 1)


def fig_observability(state: dict, out_dir: Path, dpi: int) -> Path:
    cfg, cal = state["cfg"], state["cal"]
    comm, art = state["community"], state["artifact"]
    targets = state["targets"]["metered_controlled_releases_t_h"]

    width, height = 12.8, 8.35
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(left=0.070, right=0.735, bottom=2.55 / height,
                        top=1.0 - 1.05 / height)
    ax = fig.add_subplot(1, 1, 1)

    # Bands first, so the curves sit on top of them.
    _band(ax, targets["low"], targets["high"], face="0.93", hatch="\\\\\\",
          edge="0.45", zorder=1)
    _band(ax, comm.low_t_h, comm.high_t_h, face="0.62", hatch="///", edge="black",
          zorder=3, rule_lw=1.2)
    _band(ax, art.low_t_h, art.high_t_h, face="0.28", hatch="xx", edge="black",
          zorder=3, rule_lw=2.0)

    u10 = np.linspace(1.0, 12.0, 60)
    handles: list = []
    for i, rec in enumerate(state["records"]):
        scene = rec["_scene"]
        q = (
            qmin_curve(
                scene,
                u10,
                cal,
                k_sigma=state["k_roc"],
                min_pixels=state["min_pixels"],
                pixel_size_m=state["pixel_size_m"],
                typical_plume_length_m=rec["_length_m"],
                ueff_slope=cfg.ime.ueff_slope,
                ueff_intercept=cfg.ime.ueff_intercept,
            )
            / 1000.0
        )
        st = _style(i)
        ax.plot(u10, q, linewidth=2.2, markevery=12, markersize=6, zorder=6, **st)
        ax.plot(
            [scene.u10_ms],
            [rec["qmin_t_h_at_own_wind"]],
            marker=st["marker"],
            color=st["color"],
            markersize=12,
            markerfacecolor="white",
            markeredgewidth=2.2,
            linestyle="none",
            zorder=7,
        )
        handles.append(
            Line2D(
                [],
                [],
                linewidth=2.2,
                marker=st["marker"],
                color=st["color"],
                linestyle=st["linestyle"],
                label=(
                    f"Q_min, noise-limited — {rec['label']}\n"
                    f"     MEASURED sigma_ratio {scene.sigma_log_ratio:.4f} -> "
                    f"{rec['sigma_col_ppb_measured_at_geometry']:.0f} ppb at "
                    f"{scene.satellite} SZA {rec['sza_deg']:.0f} deg "
                    f"({scene.geometry_source}); open marker = scene's own wind"
                ),
            )
        )

    handles += [
        Patch(facecolor="0.28", hatch="xx", edgecolor="black", label=art.label),
        Patch(facecolor="0.62", hatch="///", edgecolor="black", label=comm.label),
        Patch(
            facecolor="0.93",
            hatch="\\\\\\",
            edgecolor="0.45",
            label=(
                f"metered controlled-release rates this study targets, "
                f"{targets['low']:.1f}-{targets['high']:.1f} t/h "
                f"(n={targets['n']}, published truth)"
            ),
        ),
    ]

    # Published catalog rates. Other people's numbers, never ours.
    catalog = sorted(state["targets"]["catalog_by_basin"].items())
    if catalog:
        slots = np.linspace(10.7, 11.9, len(catalog))
        for j, (slot, (basin, stats)) in enumerate(zip(slots, catalog, strict=False)):
            ax.vlines(slot, stats["min_t_h"], stats["max_t_h"], color="0.15",
                      linewidth=1.3, zorder=5)
            ax.plot([slot], [stats["median_t_h"]], marker="_", color="black",
                    markersize=15, markeredgewidth=2.5, linestyle="none", zorder=6)
            ax.annotate(
                f"{basin}\nn={stats['n']}",
                xy=(slot, stats["max_t_h"]),
                xytext=(0, 7 + 16 * (j % 2)),
                textcoords="offset points",
                ha="center",
                fontsize=6.5,
                zorder=8,
            )
        handles.append(
            Line2D([], [], color="0.15", marker="_", markersize=12, linewidth=1.3,
                   label="published CAMS/SRON catalog rates per basin (right edge): "
                         "min-max bar, median tick")
        )

    ax.set_yscale("log")
    ax.set_xlim(1.0, 12.5)
    ax.set_ylim(0.1, 3000.0)
    ax.set_xlabel("10 m wind speed U10 [m/s]")
    ax.set_ylabel("emission rate [t/h]  (log scale)")
    ax.set_title(
        "Free-sensor observability atlas — what Sentinel-2 can and cannot see\n"
        "analytic noise floor from MEASURED band-ratio noise at real scene "
        "geometry, against two empirical floors\n"
        f"(k={state['k_roc']:g} sigma, {state['pixel_size_m']} m pixels, "
        f"N_min={state['min_pixels']} px, L from config "
        f"atlas.typical_plume_length_m)",
        fontsize=10.5,
    )
    ax.grid(alpha=0.3, which="both")

    # Right-margin band labels: outside the data area, so they never collide.
    for low, high, text in (
        (art.low_t_h, art.high_t_h, "OUR ARTIFACT FLOOR\n~150 t/h — a LIMITATION,\n"
                                    "not an emission (n=2)"),
        (comm.low_t_h, comm.high_t_h, "community detection floor\n"
                                      "1.0-1.4 t/h — demonstrated\nby others (n=2)"),
        (targets["low"], targets["high"], "rates this study targets\n"
                                          "(metered, published)"),
    ):
        ax.annotate(
            text,
            xy=(1.015, float(np.sqrt(low * high))),
            xycoords=("axes fraction", "data"),
            va="center",
            ha="left",
            fontsize=7.5,
            annotation_clip=False,
        )

    ax.text(
        0.015,
        0.975,
        PROVISIONAL_N_NOTE,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
        zorder=9,
        bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "round,pad=0.4"},
    )

    qmins = [r["qmin_t_h_at_own_wind"] for r in state["records"]]
    ax.text(
        0.985,
        0.02,
        (
            "READING: our artifact floor sits "
            f"{art.low_t_h / max(qmins):.0f}-{art.high_t_h / min(qmins):.0f}x ABOVE "
            "our own noise-limited floor, while the\nfloor other teams demonstrate on "
            "the same pixels sits close to it. What limits this chain is\nplume "
            "delineation and surface structure, NOT radiometric noise. No value on "
            "this figure is a\nPlumeChaser measurement of an emission."
        ),
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="right",
        zorder=9,
        bbox={"facecolor": "white", "edgecolor": "0.4", "boxstyle": "round,pad=0.35"},
    )

    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.045, 2.12 / height),
        fontsize=7.4,
        frameon=True,
        borderaxespad=0.0,
    )
    path = out_dir / "atlas_observability.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# figure 2 -- what the surface is anchored on
# --------------------------------------------------------------------------

_ANCHOR_NOTES = (
    "Rows are SINGLE SCENES, n=1 each. There is no per-basin or per-season "
    "distribution here and none is implied.",
    "The two MEASURED bars are different estimators of the same run: the dark bar "
    "takes the MAD of the band-ratio field and pushes",
    "it through the RTM curve at the scene's geometry; the hatched bar is the MAD "
    "the honesty gates took of the out-of-mask ppb",
    "field. They differ by 2.5-3.9x on the two runs that report both, which is more "
    "than MAD mechanics alone explain — treat the",
    "pair as the current uncertainty on sigma_col until one estimator is retired. "
    "A run marked DIAGNOSTIC is an audit step, not a result.",
)

_PX_FACES = ("white", "0.35", "0.80")
_PX_HATCH = ("", "", "\\\\")


def fig_noise_anchors(state: dict, out_dir: Path, dpi: int) -> Path:
    records = state["records"]
    comm, art = state["community"], state["artifact"]
    n = len(records)

    width, height = 13.4, 3.30 + 1.25 * n
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(left=0.315, right=0.962, bottom=1.60 / height,
                        top=1.0 - 1.15 / height, wspace=0.05)
    ax_s = fig.add_subplot(1, 2, 1)
    ax_q = fig.add_subplot(1, 2, 2)

    rows = list(range(n))[::-1]
    bar_h = 0.24
    for row, rec in zip(rows, records, strict=False):
        assumed = rec["sigma_col_ppb_assumed_literature"]
        if assumed:
            ax_s.barh(row + bar_h, assumed, height=bar_h, facecolor="white",
                      edgecolor="black", hatch="..", zorder=3)
        ax_s.barh(row, rec["sigma_col_ppb_measured_at_geometry"], height=bar_h,
                  facecolor="0.35", edgecolor="black", zorder=3)
        recorded = rec["sigma_col_ppb_recorded_by_gates"]
        if recorded:
            ax_s.barh(row - bar_h, recorded, height=bar_h, facecolor="0.80",
                      edgecolor="black", hatch="//", zorder=3)

    ax_s.set_yticks(rows)
    ax_s.set_yticklabels([r["label"] for r in records], fontsize=7.5)
    ax_s.set_ylim(-0.55, n - 0.45)
    ax_s.set_xscale("log")
    ax_s.set_xlim(5, 2e4)
    ax_s.set_xlabel("sigma_col [ppb], log scale")
    gate_limit = float(state["cfg"].gates.sigma_col_ppb_limit)
    ax_s.axvline(gate_limit, color="black", linestyle=(0, (4, 2)), linewidth=1.6,
                 zorder=4)
    ax_s.annotate(
        f"honesty gate, {gate_limit:.0f} ppb",
        xy=(gate_limit, n - 0.52), xytext=(4, 0), textcoords="offset points",
        rotation=90, fontsize=7, va="top", ha="left", zorder=5,
    )
    ax_s.set_title(
        "What the atlas is anchored on\n"
        "ASSUMED literature sigma vs MEASURED noise",
        fontsize=10,
    )
    ax_s.grid(alpha=0.3, axis="x", which="both")

    for row, rec in zip(rows, records, strict=False):
        for j, n_px in enumerate(MIN_PIXEL_SENSITIVITY):
            ax_q.barh(
                row + (1 - j) * bar_h,
                rec["qmin_t_h_by_min_pixels"][str(n_px)],
                height=bar_h,
                facecolor=_PX_FACES[j],
                edgecolor="black",
                hatch=_PX_HATCH[j],
                zorder=3,
            )
    ax_q.set_yticks(rows)
    ax_q.set_yticklabels([])
    ax_q.set_ylim(-0.55, n - 0.45)
    ax_q.set_xscale("log")
    ax_q.set_xlim(0.05, 2000)
    ax_q.set_xlabel("noise-limited Q_min [t/h] at the scene's own wind, log scale")
    _band(ax_q, comm.low_t_h, comm.high_t_h, face="0.62", hatch="///", edge="black",
          zorder=1, rule_lw=1.2, horizontal=False)
    _band(ax_q, art.low_t_h, art.high_t_h, face="0.28", hatch="xx", edge="black",
          zorder=1, rule_lw=2.0, horizontal=False)
    ax_q.annotate(
        "community floor\n(others, measured)",
        xy=(comm.low_t_h, n - 0.52), xytext=(-5, 0), textcoords="offset points",
        fontsize=7, va="top", ha="right", zorder=6,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
    )
    ax_q.annotate(
        "OUR ARTIFACT FLOOR\n(a limitation, not an emission)",
        xy=(art.low_t_h, n - 0.52), xytext=(-5, 0), textcoords="offset points",
        fontsize=7, va="top", ha="right", zorder=6,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
    )
    ax_q.set_title(
        "What that noise implies\n"
        f"and how much the N_min choice moves it (default "
        f"{S2_MIN_BLOB_PIXELS} px)",
        fontsize=10,
    )
    ax_q.grid(alpha=0.3, axis="x", which="both")

    fig.suptitle(PROVISIONAL_N_NOTE, fontsize=10.5, fontweight="bold",
                 y=1.0 - 0.30 / height)
    fig.legend(
        handles=[
            Patch(facecolor="white", edgecolor="black", hatch="..",
                  label="ASSUMED — literature sigma_col the atlas used before "
                        "measurement (12 / 25 ppb)"),
            Patch(facecolor="0.35", edgecolor="black",
                  label="MEASURED — config sigma_log_ratio through the RTM curve "
                        "at this scene's SZA/VZA"),
            Patch(facecolor="0.80", edgecolor="black", hatch="//",
                  label="MEASURED — sigma_col the honesty gates recorded on the "
                        "same run (different estimator)"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.315, 1.12 / height),
        fontsize=7,
        frameon=True,
        borderaxespad=0.0,
    )
    fig.legend(
        handles=[
            Patch(facecolor=_PX_FACES[j], edgecolor="black", hatch=_PX_HATCH[j],
                  label=f"N_min = {px} px")
            for j, px in enumerate(MIN_PIXEL_SENSITIVITY)
        ],
        loc="upper left",
        bbox_to_anchor=(0.690, 1.12 / height),
        fontsize=7,
        ncol=3,
        frameon=True,
        borderaxespad=0.0,
    )
    fig.text(0.012, 0.62 / height, "\n".join(_ANCHOR_NOTES), va="top", fontsize=7.2)

    path = out_dir / "atlas_noise_anchors.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# JSON sidecar + stdout summary
# --------------------------------------------------------------------------


def _rel(path: str | Path) -> str:
    """Repo-relative where possible: an absolute local path is not provenance."""
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def write_summary(state: dict, out_dir: Path, args) -> Path:
    def floor_json(floor) -> dict:
        return {
            "label": floor.label,
            "low_t_h": round(floor.low_t_h, 3),
            "high_t_h": round(floor.high_t_h, 3),
            "n": floor.n,
            "attribution": floor.attribution,
            "caveats": list(floor.caveats),
            "members": [{"source": s, "t_h": round(v, 3)} for s, v in floor.members],
        }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/make_atlas.py",
        "inputs": {
            "config": _rel(args.config),
            "rtm_calibration": _rel(args.calibration),
            "controlled_release_truth": _rel(args.truth),
            "bundles": _rel(args.bundles),
        },
        "settings": {
            "k_roc": state["k_roc"],
            "pixel_size_m": state["pixel_size_m"],
            "min_blob_pixels": state["min_pixels"],
            "min_blob_pixels_is_a_function_default": True,
            "min_blob_pixels_rationale": (
                "tropomi.min_blob_pixels=3 is calibrated for 1113 m pixels; at "
                "20 m it is single-pixel noise. See atlas.limits."
                "S2_MIN_BLOB_PIXELS."
            ),
            "assumed_sigma_col_ppb_for_contrast": ASSUMED_SIGMA_COL_PPB,
        },
        "provisional_note": PROVISIONAL_N_NOTE,
        "measured_scenes": [
            {k: v for k, v in rec.items() if not k.startswith("_")}
            for rec in state["records"]
        ],
        "floors": {
            "community_detection": floor_json(state["community"]),
            "our_quantification_artifact": floor_json(state["artifact"]),
        },
        "targets": state["targets"],
    }
    path = out_dir / "atlas_summary.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def print_summary(state: dict) -> None:
    print("\nMEASURED NOISE ANCHORS (each n=1 scene, PROVISIONAL)")
    print(
        f"{'scene':<46} {'sigma_ratio':>11} {'sigma_ppb':>10} "
        f"{'SZA':>6} {'Q_min t/h':>10}  source run"
    )
    for rec in state["records"]:
        print(
            f"{rec['key']:<46} {rec['sigma_log_ratio_measured']:>11.4f} "
            f"{rec['sigma_col_ppb_measured_at_geometry']:>10.0f} "
            f"{rec['sza_deg']:>6.1f} {rec['qmin_t_h_at_own_wind']:>10.2f}  "
            f"{rec['source_run']} [{rec['source_run_status']}]"
        )
    for floor in (state["community"], state["artifact"]):
        print(f"\n{floor.label}")
        print(f"  attribution: {floor.attribution}   n={floor.n}")
        for source, value in floor.members:
            print(f"    {source:<34} {value:8.2f} t/h")
        for caveat in floor.caveats:
            print(f"  caveat: {caveat}")


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="make_atlas.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", default=str(REPO / "config" / "default.yaml"))
    ap.add_argument(
        "--calibration", default=str(REPO / "config" / "rtm_calibration.json")
    )
    ap.add_argument(
        "--truth", default=str(REPO / "config" / "controlled_release_truth.json")
    )
    ap.add_argument("--bundles", default=str(REPO / "bundles"))
    ap.add_argument(
        "--catalog",
        default=None,
        help="CAMS/SRON weekly CSV; default is the newest mirror, skipped if absent",
    )
    ap.add_argument("--out-dir", default=str(REPO / "docs" / "figures"))
    ap.add_argument(
        "--min-pixels",
        type=int,
        default=S2_MIN_BLOB_PIXELS,
        help=f"minimum coherent blob at {20} m, in pixels "
             f"(default {S2_MIN_BLOB_PIXELS}; a documented function default, "
             f"not a config key)",
    )
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument(
        "--json-only", action="store_true", help="skip the figures, write the JSON"
    )
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = build(args)
    print_summary(state)

    written = [write_summary(state, out_dir, args)]
    if not args.json_only:
        written.append(fig_observability(state, out_dir, args.dpi))
        written.append(fig_noise_anchors(state, out_dir, args.dpi))
    print()
    for path in written:
        print(f"wrote {path.relative_to(REPO) if REPO in path.parents else path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
