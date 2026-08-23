#!/usr/bin/env python
"""Production-grade retrieval on REAL pixels via UNEP's open marss2l package.

Runs the actual MARS-S2L operational models (LGPL, pip-installable) on our
target event WITHOUT Google Earth Engine:
  * pixels: Sentinel-2 L1C TOA from the anonymous GCS public mirror
  * discovery/cloud info: element84 STAC (anonymous)
  * angles: STAC item mean solar/viewing zenith
  * winds: Open-Meteo ERA5 (anonymous)

    python scripts/mars2l_demo.py --lon 58.52 --lat 39.68 \
        --date 2026-08-05 --basin korpezhe --rate-t-h 26
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
RES_M = 20.0


def stac_items(lon, lat, d0, d1, max_cloud=20.0):
    body = {
        "collections": ["sentinel-2-l2a"],
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "datetime": f"{d0}T00:00:00Z/{d1}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "limit": 50,
    }
    r = requests.post(STAC_URL, json=body, timeout=60)
    r.raise_for_status()
    return sorted(r.json()["features"], key=lambda f: f["properties"]["datetime"])


def pick_pair(items, event_date):
    dated = [(date.fromisoformat(f["properties"]["datetime"][:10]), f) for f in items]
    tgt_c = [(abs((d - event_date).days), d, f) for d, f in dated
             if -1 <= (d - event_date).days <= 4]
    _, t_d, t_f = min(tgt_c, key=lambda x: x[0])
    refs = [(d, f) for d, f in dated if 4 <= (t_d - d).days <= 16]
    assert refs, "no background scene within 4-16 d before target"
    r_d, r_f = min(refs, key=lambda x: abs((x[0] - t_d).days))
    return t_f, t_d, r_f, r_d


def main(argv=None) -> int:
    from real_s2_demo import vrt_window, gcs_day_safes, gcs_band_href

    ap = argparse.ArgumentParser()
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--basin", default="korpezhe")
    ap.add_argument("--rate-t-h", type=float, default=None)
    ap.add_argument("--mgrs", default=None)
    ap.add_argument("--half-km", type=float, default=8.0)
    ap.add_argument("--event-id", default=None)
    ap.add_argument("--varon-ueff", action="store_true",
                    help="use Varon 2020 Ueff=0.33*U10+0.45 (matches S2 literature)")
    ap.add_argument("--core-threshold", type=float, default=None,
                    help="restrict flux IME to continuous_pred > threshold")
    args = ap.parse_args(argv)

    event_date = date.fromisoformat(args.date)

    # ---- discovery via STAC (gives dates + cloud + angles) ----------------
    feats = stac_items(args.lon, args.lat,
                       event_date - timedelta(days=30),
                       event_date + timedelta(days=6))
    t_item, t_date, b_item, b_date = pick_pair(feats, event_date)
    mgrs = args.mgrs or t_item["id"].split("_")[1]
    props_t = t_item["properties"]
    sza = float(props_t.get("s2:mean_solar_zenith",
              props_t.get("view:sun_elevation") is not None and
              90.0 - float(props_t["view:sun_elevation"]) or 35.0))
    vza = float(props_t.get("s2:mean_viewing_zenith", 10.0))
    sat = "S2A" if t_item["id"].startswith("S2A") else "S2B"
    print(f"tile {mgrs} | target {t_item['id']} ({t_date}, "
          f"cloud {props_t['eo:cloud_cover']:.2f}%)")
    print(f"background {b_item['id']} ({b_date}) | SZA={sza:.1f} VZA={vza:.1f}")

    # ---- pixels from GCS L1C (TOA) ----------------------------------------
    def find_gcs(day, offsets):
        for off in offsets:
            d = day + timedelta(days=off)
            safes = gcs_day_safes(mgrs, d.strftime("%Y%m%d"))
            if safes:
                return d, safes[0]
        raise SystemExit(f"no GCS L1C near {day}")

    # mirror lags ~12 d and lacks S2C: search around the event window
    t_date, t_safe = find_gcs(event_date, [-1, -2, 0, -3, 1, -4, 2])
    b_date, b_safe = find_gcs(t_date - timedelta(days=5),
                              [0, -1, 1, -2, 2, -3, 3])
    sat = "S2A" if "S2A" in t_safe else "S2B"
    print(f"GCS pixels: target {sat} {t_safe[:40]}... | bg {b_safe[:40]}...")

    def load_bands(safe):
        chans = {}
        for band in BANDS:
            href, off = gcs_band_href(safe, mgrs, band)
            arr = vrt_window(href, args.lon, args.lat, args.half_km, RES_M)
            arr = np.nan_to_num(arr * 1e4, nan=0.0)
            chans[band] = ((arr + off) / 1e4).astype(np.float32)
        return chans

    print("downloading target bands...")
    tc = load_bands(t_safe)
    print("downloading background bands...")
    bc = load_bands(b_safe)

    # ---- GeoTensors -------------------------------------------------------
    from georeader.geotensor import GeoTensor
    from affine import Affine
    import rasterio
    from rasterio.warp import transform as warp_transform

    href0, _ = gcs_band_href(t_safe, mgrs, "B11")
    with rasterio.open(href0) as src:
        dst_crs = src.crs
    xs, ys = warp_transform("EPSG:4326", dst_crs, [args.lon], [args.lat])
    n = int(args.half_km * 1000 / RES_M)
    transform = Affine(RES_M, 0, xs[0] - n * RES_M, 0, -RES_M, ys[0] + n * RES_M)

    def gt(chans):
        vals = np.stack([chans[b] for b in BANDS], axis=0)  # channels-first
        return GeoTensor(values=vals, transform=transform, crs=str(dst_crs),
                         fill_value_default=0,
                         attrs={"band_names": BANDS.copy()})

    t_gt, b_gt = gt(tc), gt(bc)
    valid_np = np.ones(t_gt.values.shape[1:], dtype=bool)
    from georeader.geotensor import GeoTensor as _GT
    valid_gt = _GT(values=valid_np, transform=transform, crs=str(dst_crs),
                   fill_value_default=False)

    # ---- winds ------------------------------------------------------------
    from plumechaser.data.openmeteo import openmeteo_winds

    wf = openmeteo_winds(args.lon, args.lat,
                         str(t_date - timedelta(days=1)),
                         str(t_date + timedelta(days=1)))
    speed = float(wf["speed"].median())
    u10 = float(wf["u10"].median())
    v10 = float(wf["v10"].median())
    print(f"winds: {speed:.2f} m/s (u={u10:.2f}, v={v10:.2f})")

    # ---- MBMP ratio (their implementation, with normalize+coregister) -----
    from marss2l.mars_sentinel2 import mixing_ratio_methane as mm

    print('DEBUG shapes: t_gt', t_gt.shape, 'b_gt', b_gt.shape,
          'valid', valid_gt.values.shape)
    mbmp = mm.ratio_IL(
        t_gt, b_gt,
        b12_index=BANDS.index("B12"), b11_index=BANDS.index("B11"),
        validmask=valid_gt, fill_value_ratio_il=1.0,
        normalize=True, corregister=True,
    )
    ratio_arr = np.asarray(mbmp.values if hasattr(mbmp, "values") else mbmp)
    finite_ratio = ratio_arr[np.isfinite(ratio_arr)]
    print(f"MBMP ratio: median {np.median(finite_ratio):.4f} "
          f"MAD-sigma {np.std(finite_ratio):.4f}")

    # ---- MARS-S2L segmentation model --------------------------------------
    from marss2l.mars_sentinel2 import plume_detection_model as pdm
    from marss2l.mars_sentinel2.s2lutils import get_channels_to_pred
    from marss2l.loaders import BANDS_S2_IN_L8

    model = pdm.load_model(model_name="MARS-S2L")
    img_pred = get_channels_to_pred(t_gt, channels=BANDS,
                                    channels_model=BANDS_S2_IN_L8)
    bg_pred = get_channels_to_pred(b_gt, channels=BANDS,
                                   channels_model=BANDS_S2_IN_L8)
    binary_mask, scene_score, is_plume, continuous = model.predict(
        image_predict=img_pred, background_image=bg_pred,
        wind_vector=np.array([u10, v10]),
        validmask=GeoTensor(values=valid_np, transform=transform,
                            crs=str(dst_crs)),
    )
    bm = np.asarray(binary_mask.values if hasattr(binary_mask, "values")
                    else binary_mask).astype(bool)
    print(f"MARS-S2L: is_plume={is_plume} scene_score={scene_score:.3f} "
          f"| plume px {int(bm.sum())}")

    # ---- quantification ---------------------------------------------------
    ch4_out = None
    if is_plume:
        from marss2l.mars_sentinel2 import mixing_ratio_methane as mm2
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as ttc
        from marss2l.mars_sentinel2 import quantification as qmod

        mbmp_q = mm2.ratio_IL(
            t_gt, b_gt,
            b12_index=BANDS.index("B12"), b11_index=BANDS.index("B11"),
            validmask=valid_gt, fill_value_ratio_il=1.0,
            plumemaskbool=bm, normalize=True, corregister=True,
        )
        tr = ttc.TransmittanceCH4InterpolationFromDict()
        ch4 = tr.deltach4_from_ratio_transmittance(
            satellite=sat, sza=sza, vza=vza,
            ratio_il=np.asarray(mbmp_q.values if hasattr(mbmp_q, "values")
                                else mbmp_q),
        )
        ch4_arr = np.asarray(ch4.values if hasattr(ch4, "values") else ch4)
        mask_for_flux = bm.astype(float)
        if args.core_threshold is not None:
            cont_arr = np.asarray(
                continuous.values if hasattr(continuous, "values") else continuous)
            mask_for_flux = (cont_arr > args.core_threshold).astype(float)
            print(f"core-threshold {args.core_threshold}: "
                  f"flux mask px {int(mask_for_flux.sum())} "
                  f"(was {int(bm.sum())})")
        kw = {}
        if args.varon_ueff:
            kw = dict(a_u_eff=0.33, b_u_eff=0.45)
        qout = qmod.obtain_flux_rate(
            methane_enhancement_image=ch4_arr,
            plume_mask_binary=mask_for_flux,
            wind_speed=speed, resolution=(RES_M, RES_M),
            units_methane_enhancement="ppb", seed=20270307,
            **kw,
        )
        ch4_out = ch4_arr
        print("QUANTIFICATION:", {k: round(v, 1) if isinstance(v, float) else v
                                   for k, v in qout.items()})
        if args.rate_t_h:
            print(f"catalog rate: {args.rate_t_h:.0f} t/h")
    else:
        print("MARS-S2L reports no plume at target date — honest null")

    # ---- bundle -----------------------------------------------------------
    from plumechaser.report.bundle import write_bundle
    from plumechaser.report.dossier import DossierInput

    eid = args.event_id or f"EVT-{t_date:%Y%m%d}-{args.basin}-MARSS2L"
    d = DossierInput(
        event_id=eid, basin=args.basin, det_date=str(t_date),
        lon=args.lon, lat=args.lat, event_class="catalog_targeted",
        z_peak=float(scene_score * 10),
        persistence_passes=int(is_plume), persistence_dates=[str(t_date)],
        cue_action="cue_sentinel2",
        cue_reason="marss2l production retrieval on catalog event",
        quant=None, u10_ms=speed, wind_source="Open-Meteo/ERA5",
        context_verdict=("PLUME DETECTED (MARS-S2L)" if is_plume
                         else "no plume detected by production model"),
        provenance=(
            f"engine: marss2l==0.2.10 MARS-S2L (LGPL)\n"
            f"target: {t_item['id']}\nbackground: {b_item['id']}\n"
            f"SZA/VZA: {sza:.1f}/{vza:.1f}\n"
            f"scene_score: {scene_score:.3f}\n"
            f"mbmp_ratio_median: {float(np.median(finite_ratio)):.4f}"
        ),
    )
    bdir = write_bundle(d, REPO / "bundles", extra={
        "is_plume": bool(is_plume), "scene_score": float(scene_score),
        "plume_px": int(bm.sum()), "q_output":
            ({k: (round(v, 2) if isinstance(v, float) else v)
              for k, v in qout.items()} if is_plume and 'qout' in dir() else None),
    })

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(ratio_arr, cmap="plasma_r", vmin=0.90, vmax=1.0)
    axes[0].set_title("MBMP ratio B12/B11")
    plt.colorbar(im0, ax=axes[0], fraction=0.04)
    if ch4_out is not None:
        med = np.nanmedian(ch4_out)
        im1 = axes[1].imshow(np.nan_to_num(ch4_out, nan=med), cmap="plasma",
                             vmin=0, vmax=max(1500.0, np.nanpercentile(ch4_out, 99)))
        axes[1].set_title(f"dXCH4 ppb — Q vs catalog")
        plt.colorbar(im1, ax=axes[1], fraction=0.04)
    else:
        axes[1].imshow(bm.astype(float), cmap="gray")
        axes[1].set_title("binary mask (none)")
    fig.savefig(bdir / "mbmp_png.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"bundle: {bdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
