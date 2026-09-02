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

Input conventions that this script must honour (each one burned us once)
-----------------------------------------------------------------------
1. **DN scale.** ``plume_detection_model.predict`` documents its inputs as
   "TOA reflectances multiplied by 10000", and the loaded MARS-S2L model has
   ``norm_data=False``, i.e. it normalises by dividing by 5000. Handing it
   0-1 reflectance shrinks every radiance channel by 1e4, leaving the network
   with an effectively constant image. The band-ratio (MBMP) channel is
   scale-invariant and survives, so the model still returns confident output
   -- it is just output produced without any radiance context, which is
   exactly the context needed to reject surface artifacts.
2. **Same-platform background.** S2A and S2B SWIR calibration differs and
   does not cancel in a B12/B11 ratio. Same platform also means the same
   relative orbit at exact 10-day multiples, which additionally cancels BRDF.
3. **Real valid mask.** Nodata pixels become negative DN once the baseline
   >= 05 offset (-1000) is applied; they must be excluded, not fed in as
   valid dark ground.
4. **Honesty gates apply here too.** A production model is not exempt from
   the withhold rule (analysis plan section 7).
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
# CloudSEN12 (UNetMobV2_V2, bundled with marss2l) needs the full L1C stack.
ALL_L1C_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
                 "B8A", "B09", "B10", "B11", "B12"]
CLOUDY_CLASSES = (1, 2, 3)  # thick cloud, thin cloud, cloud shadow
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
    from real_s2_demo import gcs_band_href, gcs_day_safes, vrt_window

    ap = argparse.ArgumentParser()
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--basin", default="korpezhe")
    ap.add_argument("--rate-t-h", type=float, default=None)
    ap.add_argument("--mgrs", default=None)
    ap.add_argument("--half-km", type=float, default=8.0)
    ap.add_argument("--event-id", default=None)
    ap.add_argument("--raw-u10-ueff", action="store_true",
                    help="use raw U10 as Ueff (marss2l default a=1,b=0); the "
                         "script otherwise uses the config Varon coefficients")
    ap.add_argument("--core-threshold", type=float, default=None,
                    help="restrict flux IME to continuous_pred > threshold")
    ap.add_argument("--no-cloud-mask", action="store_true",
                    help="skip CloudSEN12 screening (faster: 6 bands not 13)")
    ap.add_argument("--exact-date", action="store_true",
                    help="require the target scene to be ON --date. Mandatory "
                         "for controlled-release work, where a neighbouring "
                         "overpass carries a different (or zero) release.")
    ap.add_argument("--exclude-background", default=None, metavar="START:END",
                    help="ISO date range whose scenes may NOT be used as "
                         "background, e.g. a controlled-release campaign "
                         "window. A background inside the window carries its "
                         "own emission and would cancel the target's.")
    args = ap.parse_args(argv)

    from plumechaser.config import load_config

    cfg = load_config(REPO / "config" / "default.yaml")
    event_date = date.fromisoformat(args.date)

    # ---- discovery via STAC (gives dates + cloud + angles) ----------------
    # STAC supplies observation angles and the tile id only; pixels always
    # come from the GCS L1C mirror. On archive dates STAC may have no usable
    # pair, which must not abort a campaign -- fall back to nominal angles and
    # an explicit --mgrs, and say so, because the angles feed the RTM
    # calibration and nominal values are a real (documented) approximation.
    t_item = None
    mgrs = args.mgrs
    sza, vza = 35.0, 5.0
    try:
        feats = stac_items(args.lon, args.lat,
                           event_date - timedelta(days=30),
                           event_date + timedelta(days=6))
        t_item, stac_t_date, b_item, stac_b_date = pick_pair(feats, event_date)
        mgrs = args.mgrs or t_item["id"].split("_")[1]
        props_t = t_item["properties"]
        sza = float(props_t.get("s2:mean_solar_zenith",
                  props_t.get("view:sun_elevation") is not None and
                  90.0 - float(props_t["view:sun_elevation"]) or 35.0))
        vza = float(props_t.get("s2:mean_viewing_zenith", 10.0))
        print(f"tile {mgrs} | target {t_item['id']} ({stac_t_date}, "
              f"cloud {props_t['eo:cloud_cover']:.2f}%)")
        print(f"STAC context only (angles): {b_item['id']} ({stac_b_date}) | "
              f"SZA={sza:.1f} VZA={vza:.1f}")
    except Exception as exc:  # noqa: BLE001 - any STAC failure is recoverable
        if not mgrs:
            raise SystemExit(
                f"STAC lookup failed ({exc}) and no --mgrs given; "
                f"pass --mgrs to proceed"
            ) from exc
        print(f"WARNING: STAC lookup failed ({exc})")
        print(f"         using nominal SZA={sza:.1f} VZA={vza:.1f} — angles "
              f"feed the RTM calibration, so treat ppb as approximate")
    print("pixels come from the GCS L1C mirror, selected below")

    # ---- pixels from GCS L1C (TOA) ----------------------------------------
    _href_cache: dict[tuple[str, str], tuple[str, float]] = {}

    def band_href(safe, band):
        key = (safe, band)
        if key not in _href_cache:
            _href_cache[key] = gcs_band_href(safe, mgrs, band)
        return _href_cache[key]

    def rel_orbit(safe: str) -> str:
        return next((p for p in safe.split("_")
                     if len(p) == 4 and p[0] == "R" and p[1:].isdigit()), "R???")

    def window_nodata_frac(safe: str) -> float:
        """How much of OUR window this granule actually contains.

        One MGRS tile/day can hold several SAFEs, and a granule at the swath
        edge may cover only part of the tile. Taking the first SAFE blindly
        is how a background scene ended up missing 43% of the window.
        """
        href, _ = band_href(safe, "B12")
        arr = vrt_window(href, args.lon, args.lat, args.half_km, RES_M)
        return float((~np.isfinite(arr)).mean())

    excl_start = excl_end = None
    if args.exclude_background:
        a, b = args.exclude_background.split(":")
        excl_start, excl_end = date.fromisoformat(a), date.fromisoformat(b)

    def excluded(d) -> bool:
        return (excl_start is not None
                and excl_start <= d <= excl_end)

    def pick_covering(day, offsets, platform=None, max_nodata=0.02,
                      honour_exclusion=False):
        """Nearest SAFE to `day` whose pixels actually cover the window."""
        best = (None, None, 1.0)
        for off in offsets:
            d = day + timedelta(days=off)
            if honour_exclusion and excluded(d):
                continue
            safes = gcs_day_safes(mgrs, d.strftime("%Y%m%d"))
            if platform is not None:
                safes = [s for s in safes if s.startswith(platform)]
            for s in safes:
                frac = window_nodata_frac(s)
                if frac <= max_nodata:
                    return d, s, frac
                if frac < best[2]:
                    best = (d, s, frac)
        return best

    # mirror lags ~12 d and lacks S2C: search around the event window
    target_offsets = [0] if args.exact_date else [-1, -2, 0, -3, 1, -4, 2]
    t_date, t_safe, t_nd = pick_covering(event_date, target_offsets)
    if t_safe is None:
        raise SystemExit(
            f"no GCS L1C on {event_date}" if args.exact_date
            else f"no GCS L1C near {event_date}"
        )
    if args.exact_date and t_date != event_date:
        raise SystemExit(
            f"--exact-date: nearest covering scene is {t_date}, not {event_date}"
        )
    sat = "S2A" if t_safe.startswith("S2A") else "S2B"

    # ---- geometry (needed before any cloud mask can be built) -------------
    import rasterio
    from affine import Affine
    from georeader.geotensor import GeoTensor
    from rasterio.warp import transform as warp_transform

    href0, _ = band_href(t_safe, "B11")
    with rasterio.open(href0) as src:
        dst_crs = src.crs
    xs, ys = warp_transform("EPSG:4326", dst_crs, [args.lon], [args.lat])
    n = int(args.half_km * 1000 / RES_M)
    transform = Affine(RES_M, 0, xs[0] - n * RES_M, 0, -RES_M, ys[0] + n * RES_M)

    fetch_bands = BANDS if args.no_cloud_mask else ALL_L1C_BANDS

    def load_bands(safe, quiet=False):
        """DN-scale bands (TOA reflectance x 10000) plus a validity mask.

        vrt_window returns DN * 1e-4 with NaN at nodata; we return to DN,
        apply the per-band baseline >= 05 offset, and mark anything that is
        non-finite or non-positive after the offset as invalid.
        """
        chans: dict[str, np.ndarray] = {}
        valid = None
        for band in fetch_bands:
            href, off = band_href(safe, band)
            arr = vrt_window(href, args.lon, args.lat, args.half_km, RES_M)
            dn = arr * 1e4
            nodata = ~np.isfinite(dn)
            dn = np.nan_to_num(dn, nan=0.0) + off
            negative = (dn <= 0) & ~nodata
            ok = ~nodata & ~negative
            # Invalid pixels go to 0 DN, marss2l's own fill value; leaving the
            # -1000 the offset produces would feed negative radiances in.
            dn[~ok] = 0.0
            if not quiet:
                print(f"    {band}: nodata {nodata.mean():>6.1%} | "
                      f"negative-after-offset {negative.mean():>6.1%}")
            chans[band] = dn.astype(np.float32)
            valid = ok if valid is None else (valid & ok)
        return chans, valid

    def clear_mask(chans):
        """Clear-sky mask from CloudSEN12 (UNetMobV2_V2), bundled with marss2l."""
        from marss2l.mars_sentinel2.s2lutils import compute_cloud_mask

        vals = np.stack([chans[b] for b in ALL_L1C_BANDS], axis=0)
        g = GeoTensor(values=vals, transform=transform, crs=str(dst_crs),
                      fill_value_default=0,
                      attrs={"band_names": ALL_L1C_BANDS.copy()})
        cm = compute_cloud_mask(g, ALL_L1C_BANDS, satellite=sat)
        cm_arr = np.asarray(cm.values if hasattr(cm, "values") else cm)
        return ~np.isin(cm_arr, CLOUDY_CLASSES)

    print(f"target scene: {t_safe} (coverage {1 - t_nd:.1%})")
    print("downloading target bands...")
    tc, t_ok = load_bands(t_safe)
    t_clear = None
    if not args.no_cloud_mask:
        t_clear = clear_mask(tc)
        print(f"target cloud/shadow: {1 - t_clear.mean():.1%}")
        if 1 - t_clear.mean() > cfg.sentinel2.max_cloud_fraction:
            print("WARNING: target exceeds the frozen cloud limit "
                  f"({cfg.sentinel2.max_cloud_fraction:.0%})")

    # ---- background selection --------------------------------------------
    # Frozen reference rules (config sentinel2.reference) want a clear scene
    # close in time. Three hard constraints on top, each learned the hard way:
    # same platform (SWIR calibration), same relative orbit (BRDF), full
    # window coverage (partial granules), and now low in-window cloud -- a
    # cloudy background is what turned Permian into a false detection.
    max_cloud = cfg.sentinel2.max_cloud_fraction
    b_date = b_safe = bc = b_ok = b_clear = None
    b_nd, b_cloud = 1.0, 1.0
    tried: list[str] = []
    # Same relative orbit means multiples of 10 days, either side of the
    # target -- the frozen reference rules allow +/-12 d, and when a
    # controlled-release window blocks the earlier passes the later ones are
    # the only clean references available. Nearest in time first.
    lags = [s * m for m in (10, 20, 30, 40, 50, 60, 70) for s in (1, -1)]
    for lag in lags:
        d, s, nd = pick_covering(t_date - timedelta(days=lag), [0], platform=sat,
                                 honour_exclusion=True)
        if s is None or nd > 0.02:
            tried.append(f"{-lag:+d}d: no covering same-platform granule"
                         + (" (in excluded window)"
                            if excluded(t_date - timedelta(days=lag)) else ""))
            continue
        print(f"trying background {-lag:+d}d: {s[:44]}...")
        cand_c, cand_ok = load_bands(s, quiet=True)
        cand_clear = clear_mask(cand_c) if not args.no_cloud_mask else None
        cloud_frac = 0.0 if cand_clear is None else float(1 - cand_clear.mean())
        tried.append(f"-{lag}d: coverage {1 - nd:.0%}, cloud {cloud_frac:.0%}")
        print(f"    coverage {1 - nd:.1%} | cloud/shadow {cloud_frac:.1%}")
        if cloud_frac < b_cloud:
            b_date, b_safe, b_nd, b_cloud = d, s, nd, cloud_frac
            bc, b_ok, b_clear = cand_c, cand_ok, cand_clear
        if cloud_frac <= max_cloud:
            break
    if b_safe is None:
        raise SystemExit(f"no same-platform ({sat}) covering GCS L1C "
                         f"background before {t_date}")
    if b_cloud > max_cloud:
        print(f"WARNING: best available background is {b_cloud:.1%} cloudy, "
              f"above the frozen {max_cloud:.0%} limit — "
              f"treat this retrieval as reference-limited")

    same_orbit = rel_orbit(b_safe) == rel_orbit(t_safe)
    pairing = (f"{sat} / {rel_orbit(t_safe)} vs {rel_orbit(b_safe)}"
               f" ({(t_date - b_date).days} d)")
    print(f"background chosen: {b_safe}")
    print(f"pairing: {pairing} -> "
          f"{'same orbit' if same_orbit else 'CROSS-ORBIT (BRDF residual)'}")

    def gt(chans):
        vals = np.stack([chans[b] for b in BANDS], axis=0)  # channels-first
        return GeoTensor(values=vals, transform=transform, crs=str(dst_crs),
                         fill_value_default=0,
                         attrs={"band_names": BANDS.copy()})

    # The boundary where the 2026-08-25 scale bug lived: our arrays cross into
    # someone else's model here, and every band-ratio operation downstream is
    # invariant to a 10^4 error, so this check is the only thing that can
    # catch it.
    from plumechaser.data.scale import assert_dn_scale

    for label, chans in (("target", tc), ("background", bc)):
        v = assert_dn_scale(chans["B12"], f"{label} B12")
        print(f"input scale {label}: DN ok (p99={v.p99:,.0f})")

    t_gt, b_gt = gt(tc), gt(bc)
    valid_np = t_ok & b_ok
    if t_clear is not None and b_clear is not None:
        valid_np = valid_np & t_clear & b_clear
        cloud_note = (f"cloud screening: CloudSEN12 UNetMobV2_V2 "
                      f"(target {1 - t_clear.mean():.1%}, "
                      f"background {b_cloud:.1%} cloud/shadow)")
    else:
        cloud_note = "cloud screening: DISABLED"

    valid_frac = float(valid_np.mean())
    print(f"valid pixels: {valid_frac:.1%} of the window "
          f"({int((~valid_np).sum())} masked as nodata/negative-DN)")
    if valid_frac < 0.5:
        print("WARNING: over half the window is invalid — treat with suspicion")

    valid_gt = GeoTensor(values=valid_np, transform=transform,
                         crs=str(dst_crs), fill_value_default=False)

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
    from marss2l.loaders import BANDS_S2_IN_L8
    from marss2l.mars_sentinel2 import plume_detection_model as pdm
    from marss2l.mars_sentinel2.s2lutils import get_channels_to_pred

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
    qout = None
    qout_withheld = None
    gate = None
    if is_plume:
        from marss2l.mars_sentinel2 import mixing_ratio_methane as mm2
        from marss2l.mars_sentinel2 import quantification as qmod
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as ttc

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
        # Honesty gates apply to production retrievals too (plan section 7).
        # The ppb limit must match the scale of the field being gated: this
        # is an RTM retrieval, whose columns run several times larger than
        # our simplified chain's for the same physical noise. Convert the
        # calibration-independent anchor at this scene's geometry rather than
        # comparing against the simplified chain's 80 ppb.
        from plumechaser.retrieve.calibration import (
            CalibrationError,
            load_calibration,
        )
        from plumechaser.retrieve.gates import (
            evaluate_gates,
            sigma_ppb_limit_for_scale,
        )

        sigma_limit = cfg.gates.sigma_col_ppb_limit
        scale_note = "simplified-chain limit (RTM calibration unavailable)"
        try:
            cal = load_calibration(REPO / "config" / "rtm_calibration.json")
            sigma_limit = sigma_ppb_limit_for_scale(
                cal.c1(sat, sza, vza), cfg.gates.sigma_log_ratio_limit)
            scale_note = (f"RTM scale at SZA={sza:.1f} VZA={vza:.1f} "
                          f"(anchor {cfg.gates.sigma_log_ratio_limit})")
        except CalibrationError as exc:
            print(f"WARNING: {exc}")

        gate = evaluate_gates(
            ch4_arr, mask_for_flux.astype(bool), valid=valid_np,
            sigma_col_ppb_limit=sigma_limit,
            mask_fraction_limit=cfg.gates.mask_fraction_limit,
        )
        print(f"GATES: sigma_col {gate.sigma_col_ppb:.1f} ppb "
              f"vs {sigma_limit:.0f} ppb limit [{scale_note}] | "
              f"mask {gate.mask_fraction:.1%} of valid -> {gate.verdict}")
        for reason in gate.reasons:
            print(f"  tripped: {reason}")

        kw = {} if args.raw_u10_ueff else dict(
            a_u_eff=cfg.ime.ueff_slope, b_u_eff=cfg.ime.ueff_intercept)
        q_raw = qmod.obtain_flux_rate(
            methane_enhancement_image=ch4_arr,
            plume_mask_binary=mask_for_flux,
            wind_speed=speed, resolution=(RES_M, RES_M),
            units_methane_enhancement="ppb",
            seed=cfg.evaluation.random_seed,
            **kw,
        )
        ch4_out = ch4_arr

        # Report the decomposition either way: it is the audit evidence.
        from plumechaser.retrieve.flux_audit import audit_q_output

        fa = audit_q_output(
            q_raw, event_id=args.event_id or "run",
            catalog_rate_t_h=args.rate_t_h, window_px=int(ch4_arr.size),
        )
        print(f"  mean in-mask dXCH4 : {fa.mean_enhancement_ppb:.0f} ppb "
              f"({fa.column_enhancement_factor:.2f}x the 1800 ppb background "
              f"over {fa.plume_area_km2:.1f} km^2)")

        if gate.artifact_dominated:
            qout_withheld = q_raw
            print("QUANTIFICATION WITHHELD — artifact-dominated")
        else:
            qout = q_raw
            print("QUANTIFICATION:", {k: round(v, 1) if isinstance(v, float) else v
                                      for k, v in q_raw.items()})
        if args.rate_t_h:
            print(f"catalog rate: {args.rate_t_h:.0f} t/h "
                  f"(ratio {fa.ratio_to_catalog:.1f}x)")
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
        context_verdict=(
            gate.verdict if gate is not None
            else ("PLUME DETECTED (MARS-S2L)" if is_plume
                  else "no plume detected by production model")),
        provenance=(
            f"engine: marss2l==0.2.10 MARS-S2L (LGPL)\n"
            f"pixels target: {t_safe}\npixels background: {b_safe}\n"
            f"pairing: {pairing} ({'same' if same_orbit else 'CROSS'}-orbit)\n"
            f"angles: {t_item['id'] if t_item else 'NOMINAL (STAC unavailable)'}\n"
            f"input scale: DN (TOA reflectance x 10000)\n"
            f"{cloud_note}\n"
            f"valid pixels: {valid_frac:.1%}\n"
            f"SZA/VZA: {sza:.1f}/{vza:.1f}\n"
            f"scene_score: {scene_score:.3f}\n"
            f"mbmp_ratio_median: {float(np.median(finite_ratio)):.4f}"
        ),
    )

    def _round(q):
        return ({k: (round(v, 2) if isinstance(v, float) else v)
                 for k, v in q.items()} if q else None)

    bdir = write_bundle(d, REPO / "bundles", extra={
        "is_plume": bool(is_plume), "scene_score": float(scene_score),
        "plume_px": int(bm.sum()),
        "pixels_target_safe": t_safe,
        "pixels_background_safe": b_safe,
        "same_relative_orbit": bool(same_orbit),
        "background_window_coverage": round(1 - b_nd, 4),
        "background_cloud_fraction": round(b_cloud, 4),
        "background_candidates_tried": tried,
        "input_scale": "DN (TOA reflectance x 10000)",
        "cloud_screening": cloud_note,
        "valid_fraction": round(valid_frac, 4),
        "gates": (gate.as_dict() if gate is not None else None),
        "gate_sigma_limit_ppb": (round(sigma_limit, 1) if gate is not None else None),
        "gate_sigma_scale": (scale_note if gate is not None else None),
        "q_output": _round(qout),
        # Kept for the audit trail only; withheld from every headline by rule.
        "q_output_withheld_artifact_dominated": _round(qout_withheld),
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
        axes[1].set_title("dXCH4 ppb — Q vs catalog")
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
