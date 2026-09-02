#!/usr/bin/env python
"""Run the retrieval against METERED controlled releases. Runs in .venv-mars.

This is the only dataset where the true emission rate is known rather than
inferred from another satellite's catalog, so it is the correct instrument
for closing the absolute-flux audit. Ground truth and its provenance live in
``config/controlled_release_truth.json``.

Two things make it more informative than the catalog comparisons:

* **Zero controls.** Overpasses with a metered release of zero (and, in 2022,
  fourteen more where Stanford ran no release at all). Any flux we report on
  those is false by construction, which measures our artifact floor directly
  instead of bounding it.
* **Known rates near the detection limit.** Most releases sit at 1-7 t/h,
  right at the ~1-1.4 t/h floor these campaigns demonstrated, so the runs
  probe exactly the regime the observability atlas cares about.

Two correctness rules are enforced and must not be relaxed:

1. ``--exact-date``: a neighbouring overpass carries a different release, so
   the nearest-scene fallback would silently compare against the wrong truth.
2. ``--exclude-background``: the campaign window is barred from supplying the
   background scene. A background taken mid-campaign carries its own plume
   and would partially cancel the target's.

    python scripts/controlled_release.py --list
    python scripts/controlled_release.py --campaign ehrenberg_2021 --kind zero_control
    python scripts/controlled_release.py --limit 4 --out outputs/controlled_release.json
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

TRUTH = REPO / "config" / "controlled_release_truth.json"
# Largest releases and zero controls first: the former are most likely to be
# detectable at all, the latter measure the false-flux floor. Both are worth
# more than a marginal mid-range release if the run is cut short.
KIND_PRIORITY = {"zero_control": 0, "release": 1, "no_release": 2}


def load_truth() -> dict:
    return json.loads(TRUTH.read_text())


def iter_overpasses(truth: dict, campaign=None, kind=None):
    for camp in truth["campaigns"]:
        if campaign and camp["name"] != campaign:
            continue
        for op in camp["overpasses"]:
            if kind and op.get("kind") != kind:
                continue
            yield camp, op


def sort_key(item):
    _camp, op = item
    return (KIND_PRIORITY.get(op.get("kind"), 9), -float(op.get("kg_h", 0.0)))


def run_one(camp: dict, op: dict, *, half_km: float, core_threshold: float,
            no_cloud_mask: bool) -> dict:
    """Run one overpass; never raises, so one failure cannot end a campaign."""
    from mars2l_demo import main as demo_main

    eid = f"CR-{camp['name']}-{op['date'].replace('-', '')}"
    argv = [
        "--lon", str(camp["lon"]), "--lat", str(camp["lat"]),
        "--date", op["date"],
        "--basin", camp["name"],
        "--event-id", eid,
        "--half-km", str(half_km),
        "--core-threshold", str(core_threshold),
        "--exact-date",
        "--exclude-background",
        f"{camp['window_start']}:{camp['window_end']}",
    ]
    if op.get("kg_h"):
        argv += ["--rate-t-h", str(float(op["kg_h"]) / 1000.0)]
    if no_cloud_mask:
        argv += ["--no-cloud-mask"]

    record = {
        "event_id": eid,
        "campaign": camp["name"],
        "date": op["date"],
        "kind": op.get("kind"),
        "truth_kg_h": op.get("kg_h"),
        "truth_note": op.get("note"),
    }
    print(f"\n{'=' * 72}\n{eid}  truth={op.get('kg_h')} kg/h  "
          f"kind={op.get('kind')}\n{'=' * 72}")
    try:
        rc = demo_main(argv)
        record["status"] = "ok" if rc == 0 else f"exit {rc}"
    except SystemExit as exc:  # the demo's own guard rails
        record["status"] = f"skipped: {exc}"
        print(f"SKIPPED: {exc}")
    except Exception as exc:  # noqa: BLE001 - keep the campaign alive
        record["status"] = f"error: {exc}"
        traceback.print_exc()

    prov = REPO / "bundles" / eid / "provenance.json"
    if prov.exists():
        try:
            meta = json.loads(prov.read_text())
        except json.JSONDecodeError:
            meta = {}
        q = meta.get("q_output") or meta.get(
            "q_output_withheld_artifact_dominated") or {}
        record |= {
            "is_plume": meta.get("is_plume"),
            "scene_score": meta.get("scene_score"),
            "gates": meta.get("gates"),
            "retrieved_kg_h": q.get("Q"),
            "quantification_withheld": meta.get("q_output") is None and bool(q),
            "background_safe": meta.get("pixels_background_safe"),
            "same_relative_orbit": meta.get("same_relative_orbit"),
            "background_cloud_fraction": meta.get("background_cloud_fraction"),
            "valid_fraction": meta.get("valid_fraction"),
        }
    return record


def summarise(records: list[dict]) -> None:
    print(f"\n\n{'=' * 78}\nCONTROLLED-RELEASE SUMMARY\n{'=' * 78}")
    hdr = (f"{'event':<34}{'kind':<14}{'truth kg/h':>11}"
           f"{'plume':>7}{'retrieved':>11}{'ratio':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in records:
        truth = r.get("truth_kg_h")
        got = r.get("retrieved_kg_h")
        ratio = ("n/a" if not truth or not got else f"{got / truth:.2f}")
        flag = "" if not r.get("quantification_withheld") else " [withheld]"
        print(f"{r['event_id']:<34}{str(r.get('kind')):<14}"
              f"{('n/a' if truth is None else f'{truth:.0f}'):>11}"
              f"{str(r.get('is_plume')):>7}"
              f"{('n/a' if got is None else f'{got:.0f}'):>11}"
              f"{ratio:>8}{flag}")
        if r.get("status") != "ok":
            print(f"    {r.get('status')}")

    # The headline the zero controls exist to produce.
    zeros = [r for r in records
             if r.get("kind") in ("zero_control", "no_release")
             and r.get("retrieved_kg_h") is not None]
    false_pos = [r for r in zeros if r.get("is_plume")]
    if zeros:
        print(f"\nZERO CONTROLS: {len(zeros)} run, "
              f"{len(false_pos)} produced a plume detection")
        for r in false_pos:
            print(f"  FALSE FLUX {r['event_id']}: "
                  f"{r['retrieved_kg_h']:.0f} kg/h against a metered zero")
        if not false_pos:
            print("  no false detections — the artifact floor is below "
                  "the detection threshold on these scenes")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default=None,
                    help="ehrenberg_2021 | casa_grande_2022")
    ap.add_argument("--kind", default=None,
                    help="release | zero_control | no_release")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--half-km", type=float, default=6.0,
                    help="release plumes are small; a tight window keeps the "
                         "mask honest and the download cheap")
    ap.add_argument("--core-threshold", type=float, default=0.6)
    ap.add_argument("--no-cloud-mask", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--list", action="store_true",
                    help="show the plan without running anything")
    args = ap.parse_args(argv)

    truth = load_truth()
    items = sorted(iter_overpasses(truth, args.campaign, args.kind), key=sort_key)
    if args.limit:
        items = items[:args.limit]

    if args.list:
        print(f"{len(items)} overpass(es) selected, in run order:")
        for camp, op in items:
            print(f"  {camp['name']:<20}{op['date']}  "
                  f"{str(op.get('kind')):<14}{op.get('kg_h')} kg/h")
        return 0

    if not items:
        print("no overpasses match that selection")
        return 1

    records = []
    for camp, op in items:
        records.append(run_one(
            camp, op,
            half_km=args.half_km,
            core_threshold=args.core_threshold,
            no_cloud_mask=args.no_cloud_mask,
        ))

    summarise(records)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "truth_source": str(TRUTH.relative_to(REPO)),
            "citation": truth["citation"],
            "records": records,
        }, indent=2, default=str))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
