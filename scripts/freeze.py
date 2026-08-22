#!/usr/bin/env python
"""Freeze ceremony tooling (dry-run by default).

    python scripts/freeze.py                 # preview what freeze would do
    python scripts/freeze.py --execute       # actually write lock + tag

Steps:
  1. Export the exact environment -> runs/environment.lock (pip freeze)
  2. SHA-256 manifest over config/default.yaml, docs/ANALYSIS_PLAN.md and
     every file under src/ + scripts/ -> data/manifests/freeze_manifest.jsonl
  3. Git tag `freeze-YYYY-MM-DD` (--execute only)

The Zenodo deposit of ANALYSIS_PLAN + config + lock remains a MANUAL step by
design (restricted-access upload needs human account access).
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from plumechaser.data.mirror import append_manifest  # noqa: E402


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def collect_files() -> list[Path]:
    files = [REPO / "config" / "default.yaml", REPO / "docs" / "ANALYSIS_PLAN.md"]
    for sub in ("src", "scripts"):
        files += sorted((REPO / sub).rglob("*.py"))
    return [f for f in files if f.exists()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=f"freeze-{date.today().isoformat()}")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args(argv)

    print(f"FREEZE {'(EXECUTE)' if args.execute else '(DRY RUN)'} tag={args.tag}\n")

    # 1) environment lock
    lock_path = REPO / "runs" / "environment.lock"
    print(f"[1] environment.lock -> {lock_path.relative_to(REPO)}")
    if args.execute:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        pip = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True,
            check=True,
        )
        lock_path.write_text(pip.stdout)
        print(f"    {len(pip.stdout.splitlines())} packages pinned")

    # 2) integrity manifest
    entries = []
    for f in collect_files():
        rel = str(f.relative_to(REPO))
        digest = sha256_file(f)
        entries.append({"path": rel, "sha256": digest})
        print(f"[2] {digest[:12]}  {rel}")
    if args.execute:
        append_manifest(
            REPO / "data" / "manifests" / "freeze_manifest.jsonl",
            {"kind": "freeze", "tag": args.tag, "files": entries},
        )

    # 3) git tag
    print(f"[3] git tag {args.tag}")
    if args.execute:
        subprocess.run(["git", "tag", "-f", args.tag], check=True, cwd=REPO)
        print("    tagged (push with: git push origin --tags)")

    if not args.execute:
        print("\nDry run only. Re-run with --execute to write lock+manifest+tag.")
        print("Then MANUALLY deposit to Zenodo (restricted):")
        print("  - docs/ANALYSIS_PLAN.md")
        print("  - config/default.yaml")
        print("  - runs/environment.lock")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
