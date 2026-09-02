"""Replay-bundle builder: one self-contained directory per verified event.

Layout consumed by the dashboard (zero live APIs at the fair):

    bundles/<event_id>/
        provenance.json   # everything needed to reproduce/audit the event
        dossier.html      # rendered evidence dossier
        tropomi.png       # optional screening panel
        mbmp_png.png      # optional S2 retrieval panel

Provenance includes the git commit (best-effort), UTC timestamp, config
hash, quantification block, cue decision, and infrastructure-context
verdict -- the full chain a reviewer would ask for.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from plumechaser.config import config_sha256
from plumechaser.report.dossier import DossierInput, render_dossier
from plumechaser.report.status import STATUS_WITHDRAWN, bundle_status


class WithdrawnBundleError(RuntimeError):
    """Raised when a write would overwrite a retracted result."""


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - not a repo / no git
        return "unknown"


def write_bundle(
    dossier: DossierInput,
    out_root: str | Path,
    *,
    cfg_path: str | Path = "config/default.yaml",
    tropomi_png: bytes | None = None,
    mbmp_png: bytes | None = None,
    extra: dict | None = None,
    allow_overwrite_withdrawn: bool = False,
) -> Path:
    """Write <out_root>/<event_id>/ and return that directory.

    Raises:
        WithdrawnBundleError: if this event id names a retracted result.
            Writing would rewrite provenance.json wholesale and silently strip
            the markings that keep the withdrawn numbers out of figures and
            the fair-floor dashboard. Re-running a corrected analysis should
            use a new event id; the retracted bundle is the audit trail.
    """
    bdir = Path(out_root) / dossier.event_id

    if not allow_overwrite_withdrawn:
        existing = bdir / "provenance.json"
        prev: dict | None = None
        if existing.exists():
            try:
                prev = json.loads(existing.read_text())
            except (OSError, json.JSONDecodeError):
                prev = None
        if bundle_status(prev or {"event_id": dossier.event_id},
                         bdir.name) == STATUS_WITHDRAWN:
            raise WithdrawnBundleError(
                f"{dossier.event_id} is a withdrawn result "
                f"(docs/SUPERSEDED_RESULTS.md). Overwriting it would strip its "
                f"markings. Use a new event id, or pass "
                f"allow_overwrite_withdrawn=True if you really mean to."
            )

    bdir.mkdir(parents=True, exist_ok=True)

    provenance = {
        "event_id": dossier.event_id,
        "basin": dossier.basin,
        "det_date": dossier.det_date,
        "lon": dossier.lon,
        "lat": dossier.lat,
        "event_class": dossier.event_class,
        "z_peak": dossier.z_peak,
        "cue_action": dossier.cue_action,
        "cue_reason": dossier.cue_reason,
        "context_verdict": dossier.context_verdict,
        "quant": asdict(dossier.quant) if dossier.quant is not None else None,
        "wind_source": dossier.wind_source,
        "u10_ms": dossier.u10_ms,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": _git_commit(),
        "config_sha256": config_sha256(cfg_path),
    }
    if extra:
        provenance |= extra
    (bdir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str))  # numpy scalars safe

    render_dossier(dossier, bdir / "dossier.html")

    if tropomi_png is not None:
        (bdir / "tropomi_png").write_bytes(tropomi_png)
    if mbmp_png is not None:
        (bdir / "mbmp_png").write_bytes(mbmp_png)

    return bdir


def bundle_integrity(bundle_dir: str | Path) -> str:
    """SHA-256 over sorted file hashes -- stamped into lab notebook at freeze."""
    bdir = Path(bundle_dir)
    h = hashlib.sha256()
    for f in sorted(bdir.rglob("*")):
        if f.is_file():
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()
