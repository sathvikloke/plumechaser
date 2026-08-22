"""Wind data: ERA5 fetch (optional cdsapi) + climatology helpers.

The pipeline treats wind as the dominant quantification uncertainty, so the
same winds module serves three consumers: IME effective-wind conversion,
persistent-source composite rotation (wind roses), and the ERA5-vs-GEOS-FP
delta term on Korpezhe. Fetching requires the 'cds' extra and a Copernicus
CDS API key configured via ~/.cdsapirc; all downstream math is dependency-free.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ERA5_VARIABLES = ["10m_u_component_of_wind", "10m_v_component_of_wind"]


def era5_point_timeseries(
    lon: float,
    lat: float,
    start: str,
    end: str,
    out_nc: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch hourly u10/v10 at a point via CDS API; returns canonical DataFrame.

    Requires ``pip install plumechaser[cds]`` and a valid ``~/.cdsapirc``.
    Columns: time (UTC), u10, v10, speed, direction_from (degrees, met convention).
    """
    try:
        import cdsapi
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - env guard
        raise ImportError("install the 'cds' + 'data' extras for ERA5 access") from exc

    c = cdsapi.Client()
    request = {
        "variable": ERA5_VARIABLES,
        "location": {"latitude": lat, "longitude": lon},
        "date": f"{start}/{end}",
        "time": [f"{h:02d}:00" for h in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    target = Path(out_nc) if out_nc else (
        Path("data/mirrors/era5") / f"era5_{lat}_{lon}_{start}_{end}.nc"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    c.retrieve("reanalysis-era5-single-levels", request, str(target))
    ds = xr.open_dataset(target)
    df = ds.to_dataframe().reset_index()
    ds.close()
    u = df["u10"].astype(float)
    v = df["v10"].astype(float)
    out = pd.DataFrame(
        {
            "time": pd.to_datetime(df["valid_time"] if "valid_time" in df else df["time"]),
            "u10": u.values,
            "v10": v.values,
        }
    )
    out["speed"] = np.hypot(out["u10"], out["v10"])
    out["direction_from"] = (270.0 - np.degrees(
        np.arctan2(-out["v10"], -out["u10"])
    )) % 360.0
    return out


def wind_rose_climatology(winds: pd.DataFrame, month: int) -> dict[str, float]:
    """Monthly wind rose summary from a timeseries: mean speed + modal direction bin.

    Direction bins are 30-degree meteorological sectors ('direction FROM').
    Used to rotate persistent-source composites upwind before stacking.
    """
    sub = winds[winds["time"].dt.month == month]
    if sub.empty:
        raise ValueError(f"no samples for month {month}")
    bins = (sub["direction_from"] // 30).astype(int) % 12
    modal_bin = int(bins.mode().iloc[0])
    return {
        "mean_speed_ms": round(float(sub["speed"].mean()), 3),
        "modal_direction_deg": float(modal_bin * 30 + 15),
        "n_samples": int(len(sub)),
    }
