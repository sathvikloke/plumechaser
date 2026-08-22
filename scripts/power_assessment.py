#!/usr/bin/env python
"""Power assessment against a mirrored SRON/CAMS weekly catalog.

Reads the newest mirror under data/mirrors/sron_weekly/, filters to the
frozen hindcast window and basin polygons from config, clusters events at
central parameters (25 km / 14 d), and reports the branch assignment per
docs/ANALYSIS_PLAN.md section 5.

    python scripts/power_assessment.py [--csv path.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plumechaser.config import load_config  # noqa: E402
from plumechaser.data.sron_catalog import filter_events, load_weekly_csv  # noqa: E402
from plumechaser.evaluate.agreement import assign_branch, cluster_ids  # noqa: E402

HINDCAST_START = "2025-01-01"


def newest_mirror(mirrors_dir: str | Path) -> Path:
    files = sorted(Path(mirrors_dir).rglob("*.csv"))
    if not files:
        raise SystemExit(f"No mirrored catalogs under {mirrors_dir} - run fetch-catalogs first")
    return files[-1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="explicit catalog path")
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    csv_path = Path(args.csv) if args.csv else newest_mirror(cfg.paths.mirrors / "sron_weekly")
    df = load_weekly_csv(csv_path)
    hc = filter_events(df, date_range=(HINDCAST_START, None))

    print(f"# Power assessment ({pd.Timestamp.today().date()})")
    print(f"catalog: {csv_path}")
    print(f"hindcast window: {HINDCAST_START} -> {hc['date'].max()} | global events: {len(hc)}")
    print()

    champ_frames = []
    rows = []
    for name, basin in cfg.basins.items():
        sub = filter_events(hc, bbox=basin.bbox).copy()
        sub["basin"] = name
        n_cl = len(set(cluster_ids(sub.reset_index(drop=True), 25.0, 14)))
        rows.append((name, basin.role, len(sub), n_cl))
        if basin.role == "champion":
            champ_frames.append(sub)
        print(f"| {name} | {basin.role} | {len(sub)} | {n_cl} |")

    print()
    champ = pd.concat(champ_frames).reset_index(drop=True)
    n_eff = len(set(cluster_ids(champ, 25.0, 14)))
    branch = assign_branch(n_eff)
    print(f"CHAMPION SET: events={len(champ)} clusters={n_eff} -> BRANCH={branch.upper()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
