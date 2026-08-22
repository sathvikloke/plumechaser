"""SRON weekly super-emitter catalog: mirrored fetch + canonical event table.

Sources (configurable; defaults point at the public CAMS/SRON endpoints):
  * CAMS Methane Hotspot Explorer weekly CSV (machine-readable since May 2024)
  * SRON FTP weekly plume listings

Canonical output columns for every loader in this package:
    id, date (datetime.date), lon, lat, rate_t_h (nullable), source
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from plumechaser.data.mirror import check_schema_drift, fetch_url
from plumechaser.geo import in_bbox

COLUMN_ALIASES = {
    "id": ["id", "plume_id", "detection_id"],
    "date": ["date", "observation_date", "time", "day"],
    "lon": ["lon", "longitude", "lon_deg"],
    "lat": ["lat", "latitude", "lat_deg"],
    "rate_t_h": ["rate_t_h", "source_rate_t/h", "flux", "emission_rate", "ch4_flux"],
}

# Optional passthrough columns preserved when present (CAMS weekly schema)
PASSTHROUGH = {"source_type", "source_country", "uncertainty_t/h"}


def _resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed: dict[str, str] = {}
    for canon, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and canon not in renamed.values():
                renamed[alias] = canon
                break
    out = df.rename(columns=renamed)
    missing = {"date", "lon", "lat"} - set(out.columns)
    if missing:
        raise ValueError(f"catalog CSV missing required columns {missing}; has {list(df.columns)}")
    return out


def load_weekly_csv(path: str | Path) -> pd.DataFrame:
    """Parse a SRON/CAMS weekly detections CSV into canonical form."""
    raw = pd.read_csv(path)
    df = _resolve_columns(raw)
    # CAMS ships dates as YYYYMMDD ints; pandas would parse those as epoch ns.
    date_str = df["date"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    yyyymmdd = pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")
    iso = pd.to_datetime(date_str, errors="coerce")
    df["date"] = yyyymmdd.fillna(iso).dt.date
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    if "rate_t_h" in df.columns:
        df["rate_t_h"] = pd.to_numeric(df["rate_t_h"], errors="coerce")
    else:
        df["rate_t_h"] = pd.NA
    df = df.dropna(subset=["date", "lon", "lat"]).reset_index(drop=True)
    if "id" not in df.columns:
        df["id"] = [f"sron-{i:05d}" for i in range(len(df))]
    df["source"] = "sron_weekly"
    keep = ["id", "date", "lon", "lat", "rate_t_h", "source"]
    keep += [c for c in sorted(PASSTHROUGH) if c in df.columns]
    return df[keep]


def download_and_load(
    url: str,
    *,
    mirrors_dir: str | Path,
    manifests_dir: str | Path,
    source_name: str = "sron_weekly",
    allow_new_schema: bool = False,
) -> tuple[pd.DataFrame, Path]:
    """Fetch the latest catalog file, fingerprint its schema, return events."""
    mdir = Path(mirrors_dir) / source_name
    mpath = Path(manifests_dir) / f"{source_name}.manifest.jsonl"
    path = fetch_url(url, mdir, mpath)
    df = load_weekly_csv(path)
    check_schema_drift(df, source_name, mpath, allow_new_schema=allow_new_schema)
    return df, path


def filter_events(
    events: pd.DataFrame,
    bbox: tuple[float, float, float, float] | None = None,
    date_range: tuple[str | None, str | None] = (None, None),
) -> pd.DataFrame:
    """Spatial/temporal subset; dates accept ISO strings."""
    out = events
    if bbox is not None:
        mask = [
            in_bbox(r.lon, r.lat, bbox) for r in out.itertuples()
        ]
        out = out[mask]
    start, end = date_range
    if start is not None:
        s = pd.to_datetime(start).date()
        out = out[out["date"] >= s]
    if end is not None:
        e = pd.to_datetime(end).date()
        out = out[out["date"] <= e]
    return out.reset_index(drop=True)
