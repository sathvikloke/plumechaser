"""SRON catalog loading, schema fingerprinting, drift detection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from plumechaser.data.mirror import (
    SchemaDriftError,
    check_schema_drift,
    read_manifest,
    register_local_file,
    schema_fingerprint,
)
from plumechaser.data.sron_catalog import filter_events, load_weekly_csv


def _write_catalog(path: Path, lon_col="longitude", extra=True):
    rows = {
        "plume_id": ["s1", "s2", "s3"],
        "date": ["2025-01-05", "2025-02-11", "2025-03-20"],
        lon_col: [59.1, 4.5, -102.0],
        "latitude": [39.0, 31.5, 32.0],
    }
    if extra:
        rows["flux"] = [12.0, 8.0, 30.0]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_alias_resolution_and_canonical_columns(tmp_path):
    p = tmp_path / "weekly.csv"
    _write_catalog(p)
    df = load_weekly_csv(p)
    assert set(df.columns) >= {"id", "date", "lon", "lat", "rate_t_h", "source"}
    assert df.loc[0, "rate_t_h"] == 12.0
    assert df.loc[0, "id"] == "s1"


def test_cams_yyyymmdd_integer_dates_parse(tmp_path):
    """Live CAMS schema: date column arrives as int64 YYYYMMDD, not ISO."""
    p = tmp_path / "cams.csv"
    pd.DataFrame(
        {
            "id": ["35155_32_20_2704_160_S5P", "35156_32_20_2336_128_S5P"],
            "date": [20240726, 20240726],          # ints, as pandas infers them
            "time_UTC": ["07:34:38", "09:10:59"],
            "lat": [54.65, 36.44],
            "lon": [85.79, 61.70],
            "source_rate_t/h": [40, 20],
            "uncertainty_t/h": [15, 8],
            "source_type": ["Coal", "Oil"],
            "source_country": ["Russian Federation", "Turkmenistan"],
        }
    ).to_csv(p, index=False)
    df = load_weekly_csv(p)
    assert str(df.loc[0, "date"]) == "2024-07-26"
    assert df["rate_t_h"].tolist() == [40.0, 20.0]
    assert set(df["source_type"]) == {"Coal", "Oil"}


def test_missing_required_column_raises(tmp_path):
    p = tmp_path / "bad.csv"
    df = pd.DataFrame({"x": [1], "y": [2]})
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="required columns"):
        load_weekly_csv(p)


def test_filter_events_bbox_and_dates(tmp_path):
    p = tmp_path / "weekly.csv"
    _write_catalog(p)
    df = load_weekly_csv(p)
    korpezhe_only = filter_events(df, bbox=(58.0, 38.5, 60.0, 40.0))
    assert len(korpezhe_only) == 1 and korpezhe_only.iloc[0]["id"] == "s1"
    feb_onwards = filter_events(df, date_range=("2025-02-01", None))
    assert set(feb_onwards["id"]) == {"s2", "s3"}


class TestSchemaFingerprinting:
    def test_fingerprint_stable(self, tmp_path):
        a = pd.DataFrame({"b": [1], "a": [2]})
        b = pd.DataFrame({"a": [3], "b": [4]})
        assert schema_fingerprint(a) == schema_fingerprint(b)

    def test_drift_raises_then_acknowledged(self, tmp_path):
        man = tmp_path / "src.manifest.jsonl"
        df_v1 = pd.DataFrame({"lon": [1.0], "lat": [2.0]})
        df_v2 = pd.DataFrame({"lon": [1.0], "lat": [2.0], "new_col": [9]})
        check_schema_drift(df_v1, "test_src", man)          # baseline
        with pytest.raises(SchemaDriftError):
            check_schema_drift(df_v2, "test_src", man)      # unacknowledged
        fp = check_schema_drift(df_v2, "test_src", man, allow_new_schema=True)
        assert fp == schema_fingerprint(df_v2)
        entries = read_manifest(man)
        # baseline + acknowledged update; the RAISED attempt writes nothing
        assert sum(1 for e in entries if e.get("kind") == "schema") == 2


def test_register_local_file(tmp_path):
    csv = tmp_path / "data.csv"
    _write_catalog(csv)
    man = tmp_path / "m.jsonl"
    entry = register_local_file(csv, man)
    assert Path(entry["path"]).exists() and len(entry["sha256"]) == 64
    assert read_manifest(man)[0]["bytes"] == csv.stat().st_size
