"""Anonymous historical winds via the Open-Meteo ERA5 archive (no API key).

Non-commercial use of Open-Meteo's ERA5 endpoint is key-free, which unblocks
wind-driven IME quantification and persistent-source wind roses BEFORE the
Copernicus CDS account/key arrives. Output schema matches
:data:`plumechaser.data.winds` so ``wind_rose_climatology`` works unchanged.

Meteorological convention: direction is where wind comes FROM, degrees.
u10 = -speed * sin(dir), v10 = -speed * cos(dir).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


def openmeteo_winds(
    lon: float,
    lat: float,
    start: str,
    end: str,
    timeout: int = 60,
) -> pd.DataFrame:
    """Hourly 10 m winds for a point; canonical columns time,u10,v10,speed,
    direction_from."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": "wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }
    resp = requests.get(BASE_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    speed = np.asarray(hourly["wind_speed_10m"], dtype=float)
    dir_from = np.asarray(hourly["wind_direction_10m"], dtype=float)
    rad = np.radians(dir_from)
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(hourly["time"]),
            "speed": speed,
            "direction_from": dir_from,
            "u10": -speed * np.sin(rad),
            "v10": -speed * np.cos(rad),
        }
    )
    return df.dropna(subset=["speed"]).reset_index(drop=True)
