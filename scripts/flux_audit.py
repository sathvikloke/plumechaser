#!/usr/bin/env python
"""Offline audit of every absolute flux this project has ever produced.

Decomposes each recorded ``obtain_flux_rate`` output into the two terms that
actually set the number -- mean in-mask enhancement and plume area -- and
compares against the published catalog rate recovered from the mirrored CAMS
weekly CSV. No network, no re-run: it reads bundles/*/provenance.json.

    python scripts/flux_audit.py                    # audit every bundle
    python scripts/flux_audit.py --json outputs/flux_audit.json

Reading the output
------------------
``Q = 3600 * U_eff * C * mean_ppb * sqrt(area)``. So ``ratio_to_catalog``
splits cleanly: whatever is not explained by the mean enhancement being too
high must be explained by the mask being too large, and vice versa.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from plumechaser.config import config_sha256, load_config  # noqa: E402
from plumechaser.report.status import STATUS_QUOTABLE, bundle_status  # noqa: E402
from plumechaser.retrieve.flux_audit import (  # noqa: E402
    audit_q_output,
    first_principles_kg_m2_per_ppb,
    marss2l_kg_m2_per_ppb,
)

# mars2l_demo default retrieval window: half_km=8.0 at 20 m -> 800 x 800.
DEFAULT_WINDOW_PX = 800 * 800
MATCH_RADIUS_DEG = 0.25
MATCH_WINDOW_DAYS = 5


def catalog_rate(lon: float, lat: float, det_date: str) -> tuple[float | None, str]:
    """Recover the published rate for an event from the mirrored catalog."""
    import datetime as _dt

    from plumechaser.data.sron_catalog import load_weekly_csv

    mirrors = sorted((REPO / "data" / "mirrors" / "sron_weekly").glob("*/*.csv"))
    if not mirrors:
        return None, "no mirrored catalog"
    df = load_weekly_csv(mirrors[-1])
    try:
        d0 = _dt.date.fromisoformat(det_date)
    except ValueError:
        return None, f"unparseable date {det_date!r}"

    near = df[
        (df["lon"] - lon).abs().le(MATCH_RADIUS_DEG)
        & (df["lat"] - lat).abs().le(MATCH_RADIUS_DEG)
    ].copy()
    if near.empty:
        return None, "no catalog event within match radius"
    near["dt"] = near["date"].map(lambda d: abs((d - d0).days))
    near = near[near["dt"] <= MATCH_WINDOW_DAYS].sort_values("dt")
    near = near[near["rate_t_h"].notna()]
    if near.empty:
        return None, "no dated catalog rate in window"
    row = near.iloc[0]
    return float(row["rate_t_h"]), f"{row['id']} (+/-{int(row['dt'])} d)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default=str(REPO / "bundles"))
    ap.add_argument("--window-px", type=int, default=DEFAULT_WINDOW_PX)
    ap.add_argument("--json", default=None, help="write the full report here")
    args = ap.parse_args(argv)

    cfg = load_config(REPO / "config" / "default.yaml")

    c_marss = marss2l_kg_m2_per_ppb()
    c_ours = first_principles_kg_m2_per_ppb()
    print("UNIT PATH")
    print(f"  marss2l  ppb -> kg/m^2 : {c_marss:.4e}")
    print(f"  P/g      ppb -> kg/m^2 : {c_ours:.4e}")
    print(f"  disagreement           : {abs(c_marss / c_ours - 1) * 100:.2f}%  "
          f"-> units {'CLEARED' if abs(c_marss / c_ours - 1) < 0.01 else 'SUSPECT'}")
    print()

    rows = []
    for prov in sorted(Path(args.bundles).glob("*/provenance.json")):
        meta = json.loads(prov.read_text())
        # Withheld outputs are still audited — that is the whole point of
        # keeping them; they are only barred from headlines.
        q = meta.get("q_output")
        withheld = q is None
        if withheld:
            q = meta.get("q_output_withheld_artifact_dominated")
        if not q:
            continue
        rate, prov_note = catalog_rate(meta["lon"], meta["lat"], meta["det_date"])
        a = audit_q_output(
            q,
            event_id=meta["event_id"],
            catalog_rate_t_h=rate,
            window_px=args.window_px,
        )
        row = a.as_dict()
        row["catalog_match"] = prov_note
        row["code_commit"] = meta.get("code_commit")
        row["scene_score"] = meta.get("scene_score")
        row["quantification_withheld"] = withheld
        row["gates"] = meta.get("gates")
        # This is the one tool that deliberately reads retracted bundles, so
        # it is also the one place a withdrawn flux could print unlabelled.
        row["result_status"] = bundle_status(meta, prov.parent.name)
        rows.append(row)

    if not rows:
        print("no bundle carries a q_output — nothing to audit")
        return 0

    print("FLUX DECOMPOSITION")
    hdr = (f"{'event':<34}{'Q t/h':>9}{'cat':>7}{'x':>7}"
           f"{'mean ppb':>10}{'area km2':>10}{'mask%':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        cat = "n/a" if r["catalog_t_h"] is None else f"{r['catalog_t_h']:.0f}"
        rat = "n/a" if r["ratio_to_catalog"] is None else f"{r['ratio_to_catalog']:.1f}"
        marks = []
        if r["result_status"] != STATUS_QUOTABLE:
            marks.append(r["result_status"].upper())
        if r["quantification_withheld"]:
            marks.append("WITHHELD")
        flag = ("  [" + " / ".join(marks) + "]") if marks else ""
        print(f"{r['event_id']:<34}{r['q_t_h']:>9.1f}{cat:>7}{rat:>7}"
              f"{r['mean_enhancement_ppb']:>10.0f}{r['plume_area_km2']:>10.2f}"
              f"{r['mask_fraction'] * 100:>7.1f}%{flag}")

    print()
    print("PLAUSIBILITY")
    for r in rows:
        print(f"  {r['event_id']}")
        print(f"    column enhancement factor : {r['column_enhancement_factor']:.2f}x "
              f"the 1800 ppb background, sustained over "
              f"{r['plume_area_km2']:.1f} km^2")
        if r["mean_ppb_for_catalog_rate"] is not None:
            print(f"    catalog-consistent mean   : "
                  f"{r['mean_ppb_for_catalog_rate']:.0f} ppb over the same mask")
            print(f"    or the same mean over a mask "
                  f"{r['mask_shrink_for_catalog_rate']:.0f}x smaller in area")
        for note in r["notes"]:
            print(f"    NOTE: {note}")

    print()
    print("GATES THAT WOULD HAVE APPLIED (config sha256 "
          f"{config_sha256(REPO / 'config' / 'default.yaml')[:12]})")
    print(f"  mask fraction limit : {cfg.gates.mask_fraction_limit:.0%}")
    for r in rows:
        tripped = r["mask_fraction"] > cfg.gates.mask_fraction_limit
        print(f"  {r['event_id']:<34} mask {r['mask_fraction']:.1%} -> "
              f"{'TRIPPED' if tripped else 'passes'}")
    print("  sigma_col gate needs the enhancement field, not just q_output;")
    print("  it is applied live in scripts/mars2l_demo.py.")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "unit_path": {
                "marss2l_kg_m2_per_ppb": c_marss,
                "first_principles_kg_m2_per_ppb": c_ours,
                "relative_disagreement": abs(c_marss / c_ours - 1),
            },
            "window_px": args.window_px,
            "config_sha256": config_sha256(REPO / "config" / "default.yaml"),
            "events": rows,
        }, indent=2))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
