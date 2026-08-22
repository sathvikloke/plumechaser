"""Mirrored data access with SHA-256 manifest and schema-drift detection.

Every external artifact enters the project through :func:`fetch_url` or is
registered through :func:`register_local_file`. Layout::

    <mirrors>/<source>/<yyyymmdd>_<hash8>/<filename>
    <manifests>/<source>.manifest.jsonl   # append-only

Schema fingerprints hash the sorted ``column:dtype`` pairs of a parsed table;
any drift vs the last recorded fingerprint raises :class:`SchemaDriftError`
so a silent provider-side change cannot corrupt an in-flight hindcast.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


class SchemaDriftError(RuntimeError):
    """Raised when a source's column schema differs from its last fingerprint."""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def append_manifest(manifest_path: str | Path, entry: dict) -> None:
    p = Path(manifest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    entry = {**entry, "recorded_at": datetime.now(timezone.utc).isoformat()}
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def read_manifest(manifest_path: str | Path) -> list[dict]:
    p = Path(manifest_path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def fetch_url(
    url: str,
    dest_dir: str | Path,
    manifest_path: str | Path | None = None,
    timeout: int = 120,
    session: requests.Session | None = None,
) -> Path:
    """Download ``url`` into the hashed mirror layout and record it."""
    s = session or requests.Session()
    resp = s.get(url, timeout=timeout)
    resp.raise_for_status()
    tmp = Path(dest_dir) / "_tmp_download"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(resp.content)

    digest = sha256_file(tmp)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    final_dir = Path(dest_dir) / f"{stamp}_{digest[:8]}"
    final_dir.mkdir(parents=True, exist_ok=True)
    name = url.split("?")[0].rstrip("/").split("/")[-1] or "download.bin"
    final = final_dir / name
    tmp.rename(final)

    if manifest_path is not None:
        append_manifest(
            manifest_path,
            {"url": url, "path": str(final), "sha256": digest, "bytes": final.stat().st_size},
        )
    return final


def register_local_file(path: str | Path, manifest_path: str | Path) -> dict:
    """Register an already-local file into the manifest (returns its entry)."""
    p = Path(path)
    entry = {
        "url": f"file://{p.resolve()}",
        "path": str(p),
        "sha256": sha256_file(p),
        "bytes": p.stat().st_size,
    }
    append_manifest(manifest_path, entry)
    return entry


def schema_fingerprint(df: pd.DataFrame) -> str:
    """Stable hash over sorted 'column:dtype' pairs of a DataFrame."""
    sig = ";".join(f"{c}:{df[c].dtype}" for c in sorted(df.columns))
    return hashlib.sha256(sig.encode()).hexdigest()


def check_schema_drift(
    df: pd.DataFrame,
    source_name: str,
    manifest_path: str | Path,
    allow_new_schema: bool = False,
) -> str:
    """Compare current fingerprint against history; raise on unacknowledged drift."""
    fp = schema_fingerprint(df)
    history = [e for e in read_manifest(manifest_path) if e.get("kind") == "schema"]
    if history:
        last = history[-1]
        if last["fingerprint"] != fp and not allow_new_schema:
            raise SchemaDriftError(
                f"{source_name}: schema changed {last['fingerprint'][:8]} -> {fp[:8]}. "
                "Update parser + re-run with allow_new_schema=True to acknowledge."
            )
    elif not allow_new_schema:
        pass  # first sighting: recorded below, no prior baseline to violate
    append_manifest(
        manifest_path,
        {
            "kind": "schema",
            "source": source_name,
            "fingerprint": fp,
            "columns": sorted(df.columns.tolist()),
        },
    )
    return fp
