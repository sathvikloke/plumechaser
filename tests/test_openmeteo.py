"""Open-Meteo wind fetch tests (mocked network; live check done manually)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from plumechaser.data.openmeteo import openmeteo_winds
from plumechaser.data.winds import wind_rose_climatology

FIXTURE = {
    "hourly": {
        "time": ["2025-06-01T00:00", "2025-06-01T01:00", "2025-06-01T02:00"],
        "wind_speed_10m": [4.0, None, 8.0],
        "wind_direction_10m": [90.0, 90.0, 270.0],
    }
}


def _fake_response():
    resp = MagicMock()
    resp.json.return_value = FIXTURE
    resp.raise_for_status.return_value = None
    return resp


def test_openmeteo_parsing_and_conventions():
    with patch("plumechaser.data.openmeteo.requests.get",
               return_value=_fake_response()) as mock_get:
        df = openmeteo_winds(lon=59.5, lat=39.2,
                             start="2025-06-01", end="2025-06-01")
        called_url = mock_get.call_args
        assert "archive-api.open-meteo.com" in called_url.args[0]

    assert len(df) == 2  # None speed row dropped
    row0 = df.iloc[0]
    # dir=90 (from East): u=-speed*sin(90)=-speed, v~0
    assert row0["direction_from"] == 90.0
    assert abs(row0["u10"] - (-4.0)) < 1e-9
    assert abs(row0["v10"]) < 1e-9
    row2 = df.iloc[1]
    assert row2["direction_from"] == 270.0
    assert abs(row2["u10"] - 8.0) < 1e-9  # -8*sin(270) = +8


def test_openmeteo_feeds_wind_rose():
    with patch("plumechaser.data.openmeteo.requests.get",
               return_value=_fake_response()):
        df = openmeteo_winds(lon=59.5, lat=39.2,
                             start="2025-06-01", end="2025-06-30")
    rose = wind_rose_climatology(df, month=6)
    assert set(rose) == {"mean_speed_ms", "modal_direction_deg", "n_samples"}
    assert rose["n_samples"] == 2
