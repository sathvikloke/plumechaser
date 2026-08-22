"""Google Earth Engine TROPOMI CH4 screening helpers (thin integration layer).

Heavy lifting happens server-side; this module only assembles requests so the
core detection math can run on small exported arrays (identical logic lives
in :mod:`plumechaser.detect.background` and is unit-tested there).

Requires the 'gee' extra, a GEE noncommercial registration, and
``earthengine authenticate`` beforehand.
"""

from __future__ import annotations

import ee  # noqa: I001  (guarded at call time, but import documents the dep)


def daily_ch4_image(
    collection_id: str,
    band: str,
    day: str,
):
    """First OFFL CH4 image intersecting ``day`` (YYYY-MM-DD), band selected."""
    start = ee.Date(day)
    return (
        ee.ImageCollection(collection_id)
        .select(band)
        .filterDate(start, start.advance(1, "day"))
        .first()
    )


def background_median_image(
    collection_id: str,
    band: str,
    center_day: str,
    window_days: int = 30,
):
    """Rolling-window median composite centred on ``center_day``."""
    start = ee.Date(center_day).advance(-(window_days // 2), "day")
    end = ee.Date(center_day).advance(window_days // 2 + 1, "day")
    return (
        ee.ImageCollection(collection_id)
        .select(band)
        .filterDate(start, end)
        .median()
    )


def export_region_pixels(
    image,
    bbox: tuple[float, float, float, float],
    scale_m: int = 1113,
    filename_prefix: str = "plumechaser_screen",
) -> dict:
    """Kick off an Export.toDrive task for the ROI; returns the task config.

    bbox order matches config: (lon_min, lat_min, lon_max, lat_max).
    """
    region = ee.Geometry.Rectangle(list(bbox))
    task = ee.batch.Export.image.toDrive(
        image=image.clip(region),
        description=filename_prefix,
        region=region,
        scale=scale_m,
        maxPixels=int(1e9),
        fileFormat="GeoTIFF",
    )
    task.start()
    return {"task_id": task.id, "bbox": list(bbox), "scale_m": scale_m}
