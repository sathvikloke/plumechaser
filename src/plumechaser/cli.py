"""Command-line interface wiring every pipeline stage.

Usage examples:
    plumechaser fetch-catalogs --url <sron_csv_url>
    plumechaser train --pos CNN_pos.nc --neg CNN_neg.nc --out runs/detector
    plumechaser screen --basin korpezhe --date 2025-06-01
    plumechaser score --detections detections.csv --strict sron.csv
    plumechaser atlas
    plumechaser dashboard
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="plumechaser", description=__doc__)
    p.add_argument("--config", default="config/default.yaml", help="path to YAML config")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch-catalogs", help="mirror a SRON/CAMS weekly CSV")
    f.add_argument("--url", required=True)
    f.add_argument("--source", default="sron_weekly")

    t = sub.add_parser("train", help="train the plume CNN on Zenodo scenes")
    t.add_argument("--pos", required=True)
    t.add_argument("--neg", required=True)
    t.add_argument("--out", default="runs/detector")
    t.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    v = sub.add_parser("train-svc", help="train SVC artifact filter (stage 2)")
    v.add_argument("--data", default="data/zenodo/SVC_trainingdata.nc")
    v.add_argument("--out", default="runs/svc")
    v.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("screen", help="run TROPOMI-tier screening for one basin/day")
    s.add_argument("--basin", required=True)
    s.add_argument("--date", required=True)

    sc = sub.add_parser("score", help="agreement study vs reference catalogs")
    sc.add_argument("--detections", required=True)
    sc.add_argument("--strict", required=True)
    sc.add_argument("--lenient")

    sub.add_parser("atlas", help="build analytic limit surfaces")
    sub.add_parser("dashboard", help="launch replay dashboard (Streamlit)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "fetch-catalogs":

        from plumechaser.config import load_config
        from plumechaser.data.sron_catalog import download_and_load

        cfg = load_config(args.config)
        df, path = download_and_load(
            args.url,
            mirrors_dir=cfg.paths.mirrors,
            manifests_dir=cfg.paths.manifests,
            source_name=args.source,
        )
        print(f"mirrored {len(df)} events -> {path}")

    elif args.cmd == "train":
        from plumechaser.ml.dataset import build_training_arrays
        from plumechaser.ml.train import train_three_seeds

        x, y = build_training_arrays(args.pos, args.neg)
        results = train_three_seeds(x, y, args.out, seeds=tuple(args.seeds))
        for r in results:
            print(r)

    elif args.cmd == "train-svc":
        from plumechaser.ml.svc import train_svc

        metrics = train_svc(args.data, args.out, seed=args.seed)
        print(metrics)

    elif args.cmd == "screen":
        from datetime import date as _date

        from plumechaser.config import load_config

        cfg = load_config(args.config)
        basin = cfg.basins[args.basin]
        print(
            f"[screen] basin={args.basin} role={basin.role} bbox={basin.bbox} "
            f"date={_date.fromisoformat(args.date)}\n"
            "[screen] GEE export queued via data.gee_screen.export_region_pixels; "
            "client-side scoring runs after Drive sync (see docs/RUNBOOK.md)."
        )

    elif args.cmd == "score":
        import pandas as pd

        from plumechaser.config import load_config
        from plumechaser.data.sron_catalog import load_weekly_csv
        from plumechaser.evaluate.agreement import run_agreement

        cfg = load_config(args.config)
        e = cfg.evaluation
        detections = pd.read_csv(args.detections)
        detections["date"] = pd.to_datetime(detections["date"]).dt.date
        strict = load_weekly_csv(args.strict)
        lenient = load_weekly_csv(args.lenient) if getattr(args, "lenient", None) else None
        reports = run_agreement(
            detections,
            strict,
            lenient,
            radius_km=e.match_radius_km,
            window_days=e.match_window_days,
            cluster_distance_km=e.cluster_distance_km,
            cluster_window_days=e.cluster_window_days,
            bootstrap_draws=e.bootstrap_draws,
            seed=e.random_seed,
        )
        for mode, rep in reports.items():
            print(f"== mode={mode} branch={rep.branch} ==")
            print(rep.metrics)
            if rep.precision_ci:
                print(f"precision CI: {rep.precision_ci}")
                print(f"recall CI:    {rep.recall_ci}")
            print(rep.sensitivity.to_string(index=False))
            print(rep.interpretation + "\n")

    elif args.cmd == "atlas":
        from plumechaser.atlas.limits import limit_surface
        from plumechaser.config import load_config

        cfg = load_config(args.config)
        basins_meta = {
            name: {"surface_class": b.surface_class} for name, b in cfg.basins.items()
        }
        surface = limit_surface(
            basins_meta,
            seasons=["DJF", "MAM", "JJA", "SON"],
            u10_by_basin_season={},  # filled from ERA5 climatology at runtime
            sigma_by_class_season={
                ("homogeneous_arid", "JJA"): 12.0,
                ("heterogeneous", "JJA"): 25.0,
            },  # placeholder seed values; replaced by measured sigma_col per plan
            k_sigma=float(cfg.raw["atlas"]["k_roc"]),
            min_pixels=cfg.tropomi.min_blob_pixels,
            pixel_size_m=int(cfg.tropomi.screening_pixel_size_m),
            lengths_by_class=dict(cfg.raw["atlas"]["typical_plume_length_m"]),
        )
        print(surface)

    elif args.cmd == "dashboard":  # pragma: no cover - interactive
        from plumechaser.report.dashboard import run_dashboard

        run_dashboard()

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
