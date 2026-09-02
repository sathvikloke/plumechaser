#!/usr/bin/env python
"""Real Sentinel-2 pixels through the full PlumeChaser retrieval chain.

Targets a REAL catalog event (CAMS weekly), pulls target+reference scenes,
runs MBMP -> artifact gates -> IME with Open-Meteo winds -> evidence bundle.

Sources:
  * gcs-l1c (default): Sentinel-2 L1C TOA from Google's anonymous public
    mirror. True TOA reflectance where our alpha coefficients live.
  * aws-l2a: element84 STAC L2A BOA COGs. Documented to be artifact-prone
    for band-ratio retrievals (per-pass atmospheric correction); kept only
    for comparison.

Retrieval hygiene implemented here (every item burned us at least once;
see the flux audit in docs/S2_REAL_DATA_FINDINGS.md):
  1. Same-platform target/reference pairing (S2A-vs-S2B SWIR calibration
     differences do not cancel in band ratios), preferring the same
     relative orbit -- for one platform those are the passes at exact
     10-day multiples, which also cancels BRDF. Cross-orbit comparison
     dates are used only to make up a shortfall, and say so loudly.
  2. Coverage-aware scene selection: one MGRS tile/day can hold several
     SAFEs and a granule at the swath edge may cover only part of the
     tile, so every candidate is probed with one cheap single-band window
     read before it is accepted (``pick_covering``).
  3. Sub-pixel co-registration of every comparison pass via FFT phase
     correlation on B11 before differencing.
  4. Baseline >=05 radiometric offsets parsed per-band (L1C band_ids
     B11=11, B12=12; the 13-entry L1C list includes B10, unlike L2A).
     Nodata pixels, and any pixel the -1000 DN offset drives non-positive,
     are excluded from the valid mask and filled with 0 DN -- left
     negative, a pair of them forms a *positive* band ratio that the
     log-ratio retrieval would accept as signal.

Honesty gates: sigma_col > 80 ppb or plume-mask fraction > 15% marks the run
ARTIFACT-DOMINATED; no quantification is claimed in that case.
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from collections.abc import Callable, Iterable, Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
GCS_BASE = "https://storage.googleapis.com/storage/v1/b/gcp-public-data-sentinel-2/o"
GCS_PUBLIC = "https://storage.googleapis.com/gcp-public-data-sentinel-2"
ALPHA_11, ALPHA_12 = 3.0e-5, 1.2e-4  # config defaults

SIGMA_ARTIFACT_LIMIT = 80.0
MASK_FRAC_LIMIT = 0.15

# A granule must cover the retrieval window to be usable at all. Not a
# science threshold: it is the scene-selection rule shared with
# scripts/mars2l_demo.py so the two chains cannot disagree about which
# scenes exist.
MAX_WINDOW_NODATA = 0.02


def _http_json(url: str, params: dict | None = None) -> dict:
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------- GCS mirror

def gcs_day_safes(mgrs: str, yyyymmdd: str) -> list[str]:
    """SAFE directory names for one MGRS tile + day."""
    base = f"tiles/{mgrs[:2]}/{mgrs[2]}/{mgrs[3:5]}"
    items: list[str] = []
    for plat in ("S2A", "S2B"):
        js = _http_json(GCS_BASE, {
            "prefix": f"{base}/{plat}_MSIL1C_{yyyymmdd}",
            "maxResults": 30,
            "fields": "items(name)",
        })
        items += [i["name"] for i in js.get("items", [])]
    return sorted({n.split("/")[4] for n in items
                   if len(n.split("/")) > 4 and ".SAFE" in n.split("/")[4]})


def gcs_band_href(safe: str, mgrs: str, band: str) -> tuple[str, float]:
    """Public URL of one band JP2 plus its radiometric add-offset."""
    f"{GCS_PUBLIC}/tiles/{mgrs[:2]}/{mgrs[2]}/{mgrs[3:5]}/{safe}"
    pre = f"tiles/{mgrs[:2]}/{mgrs[2]}/{mgrs[3:5]}/{safe}"
    js = _http_json(GCS_BASE, {"prefix": f"{pre}/GRANULE/", "maxResults": 10})
    gids = {i["name"].split("/GRANULE/")[1].split("/")[0]
            for i in js.get("items", []) if "/GRANULE/" in i["name"]}
    assert gids, f"no GRANULE under {pre}"
    img_prefix = f"{pre}/GRANULE/{sorted(gids)[0]}/IMG_DATA/"
    js2 = _http_json(GCS_BASE, {"prefix": img_prefix, "maxResults": 60})
    names = [i["name"] for i in js2.get("items", [])
             if i["name"].endswith(f"_{band}.jp2")]
    assert names, f"band {band} not found under {img_prefix}"
    href = f"{GCS_PUBLIC}/{names[0]}"

    # L1C carries 13 RADIO_ADD_OFFSET entries (band_id 0-12) INCLUDING the
    # B10 cirrus band; the L2A ordering omits B10 and would shift B11/B12
    # onto the wrong entries. Verified against a real MTD_DS.xml (13 entries).
    band_id = {"B01": 0, "B02": 1, "B03": 2, "B04": 3, "B05": 4,
               "B06": 5, "B07": 6, "B08": 7, "B8A": 8, "B09": 9,
               "B10": 10, "B11": 11, "B12": 12}[band]
    mtd_pre = pre + "/DATASTRIP/"
    jsm = _http_json(GCS_BASE, {"prefix": mtd_pre, "maxResults": 5})
    mtd_names = [i["name"] for i in jsm.get("items", [])
                 if i["name"].endswith("MTD_DS.xml")]
    offset = 0.0
    if mtd_names:
        xml = requests.get(
            f"{GCS_PUBLIC}/{mtd_names[0]}", timeout=60).text
        pairs = re.findall(
            r'<RADIO_ADD_OFFSET\s+band_id="(\d+)">(-?\d+)</RADIO_ADD_OFFSET>', xml)
        lut = {int(b): float(v) for b, v in pairs}
        offset = lut.get(band_id, 0.0)
    return href, offset


# ------------------------------------------------------- scene bookkeeping

def safe_platform(safe: str) -> str:
    """Platform id (S2A/S2B/S2C) from a SAFE directory name."""
    return safe.split("_")[0]


def rel_orbit(safe: str) -> str:
    """Relative-orbit token (``Rnnn``) of a SAFE name, ``R???`` when absent.

    Same platform *and* same relative orbit means the same viewing geometry,
    so BRDF cancels in the band ratio; for a single platform those are the
    passes separated by exact 10-day multiples.
    """
    return next((p for p in safe.split("_")
                 if len(p) == 4 and p[0] == "R" and p[1:].isdigit()), "R???")


def window_nodata_frac(window: np.ndarray) -> float:
    """Fraction of the retrieval window a granule does not actually contain."""
    return float((~np.isfinite(np.asarray(window, dtype=np.float64))).mean())


def pick_covering(
    safes: Iterable[str],
    probe: Callable[[str], float],
    max_nodata: float = MAX_WINDOW_NODATA,
) -> tuple[str | None, float, bool]:
    """First SAFE whose pixels actually cover the window, else the best one.

    One MGRS tile/day can hold several SAFEs, and a granule at the swath
    edge may cover only part of the tile, so ``safes[0]`` is not safe:
    taking it blindly gave the sibling chain a background missing 43% of the
    window, which then entered the retrieval as -1000 DN. ``probe(safe)``
    returns that SAFE's window nodata fraction (one cheap single-band read).

    Returns ``(safe, nodata_frac, covered)``; ``(None, 1.0, False)`` when
    there is nothing to choose from.
    """
    best_safe: str | None = None
    best_frac = 1.0
    for safe in safes:
        frac = probe(safe)
        if frac <= max_nodata:
            return safe, frac, True
        if best_safe is None or frac < best_frac:
            best_safe, best_frac = safe, frac
    return best_safe, best_frac, False


def dn_with_valid(
    window: np.ndarray, offset: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """DN-scale band plus validity, with the baseline >=05 offset applied.

    ``vrt_window`` hands back DN * 1e-4 with NaN wherever the granule holds
    nodata. Restoring DN and adding RADIO_ADD_OFFSET (-1000 for baseline
    >=05) drives those nodata pixels -- and any genuinely dark pixel below
    the offset -- negative. Both classes are invalid and must never reach
    the retrieval: two negative DNs form a *positive* B11/B12 ratio, which
    the log-ratio retrieval happily accepts as if it were signal. Invalid
    pixels are dropped from the mask and filled with 0 DN.

    Returns ``(dn, valid, nodata, negative)``.
    """
    dn = np.asarray(window, dtype=np.float64) * 1e4
    nodata = ~np.isfinite(dn)
    dn = np.nan_to_num(dn, nan=0.0) + float(offset)
    negative = (dn <= 0) & ~nodata
    valid = ~nodata & ~negative
    dn[~valid] = 0.0
    return dn, valid, nodata, negative


def nan_invalid(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Copy of ``arr`` with invalid pixels as NaN (the retrieval's sentinel).

    Registration and the log-ratio both treat NaN as "no pixel here"; the
    0 DN fill from :func:`dn_with_valid` would otherwise register as very
    dark ground and drag the phase-correlation peak.
    """
    out = np.array(arr, dtype=np.float64, copy=True)
    out[~valid] = np.nan
    return out


def select_references(
    t_safe: str,
    t_day: date,
    safes_on: Callable[[date], Sequence[str]],
    choose: Callable[[Sequence[str]], tuple[str, float] | None],
    want: int,
    min_lag_days: int = 3,
    max_lag_days: int = 21,
) -> tuple[list[tuple[date, str, float]], int]:
    """Comparison scenes for the MBPD median, in preference order.

    Same platform first (S2A/S2B SWIR calibration differences do not cancel
    in a band ratio), then same relative orbit (BRDF; exact 10-day multiples
    for one platform). Cross-orbit passes are used only to make up a
    shortfall — mixing orbits adds variance rather than averaging it down,
    so the caller warns when any is used.

    ``safes_on(day)`` lists the SAFEs for a day and ``choose(safes)`` returns
    ``(safe, nodata_frac)`` for the one that covers the retrieval window, or
    None if none does. Both are injected, which keeps this rule network-free.

    Returns ``(picked, n_same_orbit)``, ``picked`` most recent first as
    ``[(day, safe, nodata_frac)]``.
    """
    plat, orbit = safe_platform(t_safe), rel_orbit(t_safe)
    days = [t_day - timedelta(days=k)
            for k in range(min_lag_days, max_lag_days + 1)]

    def gather(candidate_days: Iterable[date], n: int, same_orbit: bool):
        picked: list[tuple[date, str, float]] = []
        for day in candidate_days:
            if len(picked) >= n:
                break
            cands = [s for s in safes_on(day)
                     if safe_platform(s) == plat
                     and (rel_orbit(s) == orbit) == same_orbit]
            if not cands:
                continue
            chosen = choose(cands)
            if chosen is not None:
                picked.append((day, chosen[0], chosen[1]))
        return picked

    same = gather(days, want, same_orbit=True)
    picked = list(same)
    if len(picked) < want:
        seen = {d for d, _, _ in picked}
        picked += gather([d for d in days if d not in seen],
                         want - len(picked), same_orbit=False)
        picked.sort(key=lambda r: r[0], reverse=True)
    return picked, len(same)


# ------------------------------------------------------------- registration

def _center_crop(a: np.ndarray, frac: float = 0.7) -> np.ndarray:
    h, w = a.shape
    ch, cw = int(h * frac), int(w * frac)
    r0, c0 = (h - ch) // 2, (w - cw) // 2
    return a[r0:r0 + ch, c0:cw + c0]


def _subpixel_peak(corr: np.ndarray, iy: int, ix: int) -> tuple[float, float]:
    """Parabolic refinement of the correlation peak (sub-pixel shift)."""
    def para(m1, m0, p1):
        denom = (m1 - 2 * m0 + p1)
        return 0.0 if abs(denom) < 1e-12 else 0.5 * (m1 - p1) / denom
    h, w = corr.shape
    dy = para(corr[(iy - 1) % h, ix], corr[iy, ix], corr[(iy + 1) % h, ix])
    dx = para(corr[iy, (ix - 1) % w], corr[iy, ix], corr[iy, (ix + 1) % w])
    return float(dy), float(dx)


def phase_shift(ref: np.ndarray, tgt: np.ndarray) -> tuple[float, float]:
    """Sub-pixel (dy, dx) shifting ``ref`` to best align with ``tgt``."""
    a = np.nan_to_num(_center_crop(tgt) - np.nanmean(_center_crop(tgt)))
    b = np.nan_to_num(_center_crop(ref) - np.nanmean(_center_crop(ref)))
    a -= a.mean()
    b -= b.mean()
    fa, fb = np.fft.rfft2(a), np.fft.rfft2(b)
    cross = fa * np.conj(fb)
    cross /= np.abs(cross) + 1e-9
    corr = np.fft.irfft2(cross, s=a.shape)
    iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
    dy_i = iy - a.shape[0] if iy > a.shape[0] // 2 else iy
    dx_i = ix - a.shape[1] if ix > a.shape[1] // 2 else ix
    dy_f, dx_f = _subpixel_peak(corr, iy, ix)
    return round(dy_i + dy_f, 2), round(dx_i + dx_f, 2)


def shift_array(arr: np.ndarray, dy: float, dx: float) -> np.ndarray:
    from scipy import ndimage

    if abs(dy) < 0.05 and abs(dx) < 0.05:
        return arr
    shifted = ndimage.shift(
        np.nan_to_num(arr, nan=np.nanmedian(arr)), (dy, dx),
        order=3, mode="nearest",
    )
    shifted[np.isnan(arr)] = np.nan   # keep original validity mask
    return shifted


# ------------------------------------------------------------ AWS L2A path

def stac_scenes(lon: float, lat: float, d0: date, d1: date,
                max_cloud: float = 20.0):
    body = {
        "collections": ["sentinel-2-l2a"],
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "datetime": f"{d0.isoformat()}T00:00:00Z/{d1.isoformat()}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "limit": 50,
    }
    r = requests.post(STAC_URL, json=body, timeout=60)
    r.raise_for_status()
    return sorted(r.json()["features"], key=lambda f: f["properties"]["datetime"])


def vrt_window(href: str, lon: float, lat: float, half_km: float = 10.0,
               res_m: float = 20.0) -> np.ndarray:
    import rasterio
    from affine import Affine
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import transform as warp_transform

    with rasterio.open(href) as src:
        dst_crs = src.crs
        nodata = src.nodata if src.nodata is not None else 0.0
        scale = 1e-4
        xs, ys = warp_transform("EPSG:4326", dst_crs, [lon], [lat])
        x0, y0 = xs[0], ys[0]
        # Clamp anchor so the full window stays inside the granule.
        m = half_km * 1000 * 1.05
        x0 = min(max(x0, src.bounds.left + m), src.bounds.right - m)
        y0 = min(max(y0, src.bounds.bottom + m), src.bounds.top - m)

    n = int(half_km * 1000 / res_m)
    transform = Affine(res_m, 0, x0 - n * res_m, 0, -res_m, y0 + n * res_m)
    with rasterio.open(href) as src:
        with WarpedVRT(src, crs=dst_crs, transform=transform,
                       width=2 * n, height=2 * n,
                       resampling=Resampling.bilinear) as vrt:
            arr = vrt.read(1).astype(np.float64)
    arr[arr <= nodata] = np.nan
    if args_is_l1c(href):
        pass  # scaling handled by caller (offset-aware)
    return arr * scale


def args_is_l1c(href: str) -> bool:  # pragma: no cover - tiny tag helper
    return "_L1C_" in href or "/L1C/" in href


def auto_mgrs(lon: float, lat: float, around: date) -> str:
    body = {
        "collections": ["sentinel-2-l2a"],
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "datetime": f"{around.isoformat()}T00:00:00Z/"
                    f"{(around + timedelta(days=10)).isoformat()}T23:59:59Z",
        "limit": 1,
    }
    r = requests.post(STAC_URL, json=body, timeout=60)
    r.raise_for_status()
    feats = r.json().get("features", [])
    assert feats, "STAC returned nothing for MGRS lookup"
    return feats[0]["id"].split("_")[2]


# ------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--basin", default="permian")
    ap.add_argument("--event-id", default=None)
    ap.add_argument("--rate-t-h", type=float, default=None)
    ap.add_argument("--source", choices=["gcs-l1c", "aws-l2a"], default="gcs-l1c")
    ap.add_argument("--mgrs", default=None, help="MGRS tile (gcs-l1c); "
                    "auto-derived from STAC when omitted")
    ap.add_argument("--half-km", type=float, default=10.0)
    ap.add_argument("--inject-ppb", type=float, default=None,
                    help="synthetic Gaussian plume injected into target "
                         "bands BEFORE retrieval (Gate-B style recovery test)")
    ap.add_argument("--n-refs", type=int, default=4,
                    help="number of comparison dates (MBPD; AMT 2022 caps at 12)")
    args = ap.parse_args(argv)

    event_date = date.fromisoformat(args.date)
    mgrs = args.mgrs or auto_mgrs(args.lon, args.lat, event_date)
    print(f"MGRS tile: {mgrs}")

    # ---- discover target + same-platform reference -----------------------
    scene_extra: dict = {}
    # Gate-B synthetic injection exists only on the L1C path, but the recovery
    # report below reads this unconditionally; initialise it here so
    # `--source aws-l2a` does not die with a NameError.
    injected_truth = None
    if args.source == "gcs-l1c":
        # Both listings and window probes are memoised: gcs_band_href costs
        # three HTTP calls, and a probed SAFE is usually loaded right after.
        _href_cache: dict[tuple[str, str], tuple[str, float]] = {}
        _safes_cache: dict[date, list[str]] = {}
        _probe_cache: dict[str, float] = {}

        def band_href(safe: str, band: str) -> tuple[str, float]:
            key = (safe, band)
            if key not in _href_cache:
                _href_cache[key] = gcs_band_href(safe, mgrs, band)
            return _href_cache[key]

        def safes_on(day: date) -> list[str]:
            if day not in _safes_cache:
                _safes_cache[day] = gcs_day_safes(mgrs, day.strftime("%Y%m%d"))
            return _safes_cache[day]

        def probe(safe: str) -> float:
            """Window nodata fraction of one SAFE (single cheap B12 read)."""
            if safe not in _probe_cache:
                href, _ = band_href(safe, "B12")
                _probe_cache[safe] = window_nodata_frac(
                    vrt_window(href, args.lon, args.lat, args.half_km))
            return _probe_cache[safe]

        def choose(safes) -> tuple[str, float] | None:
            safe, nd, covered = pick_covering(safes, probe)
            if safe is None:
                return None
            if not covered:
                print(f"  skipped {safe[:44]}...: granule covers only "
                      f"{1 - nd:.0%} of the retrieval window")
                return None
            return safe, nd

        # Target: nearest day to the event whose granule actually covers the
        # window (days probed outward from the event, earlier date on ties).
        t_day = t_safe = None
        t_nd = 1.0
        for off in sorted(range(-16, 9), key=lambda o: (abs(o), o)):
            d = event_date + timedelta(days=off)
            chosen = choose(safes_on(d))
            if chosen is not None:
                t_day, t_safe, t_nd = d, chosen[0], chosen[1]
                break
        if t_safe is None:
            raise SystemExit("no GCS L1C granule covering the retrieval "
                             f"window near {event_date}")

        plat, t_orbit = safe_platform(t_safe), rel_orbit(t_safe)
        refs, n_same_orbit = select_references(
            t_safe, t_day, safes_on, choose, args.n_refs)
        n_refs = len(refs)
        if n_refs == 0:
            raise SystemExit("no same-platform comparison dates before target")
        ref_safes = [s for _, s, _ in refs]
        cross_orbit = [s for _, s, _ in refs if rel_orbit(s) != t_orbit]
        print(f"target    : {t_safe} ({t_day}) [{plat} {t_orbit}] "
              f"window coverage {1 - t_nd:.0%}")
        for d, s, nd in refs:
            print(f"comparison: {s} ({d}) [{rel_orbit(s)}] "
                  f"coverage {1 - nd:.0%}")
        if cross_orbit:
            print(f"WARNING: only {n_same_orbit} same-orbit ({t_orbit}) "
                  f"comparison date(s) available; falling back to "
                  f"{len(cross_orbit)} cross-orbit reference(s) "
                  f"({', '.join(sorted({rel_orbit(s) for s in cross_orbit}))})"
                  " — BRDF does not cancel across orbits and sqrt(N) "
                  "averaging does not apply to them")

        def load_pass(safe, quiet: bool = True):
            """(B11, B12, valid) at reflectance scale, offsets applied.

            Nodata and any pixel driven non-positive by the baseline >=05
            offset are excluded from ``valid`` and filled with 0 DN rather
            than left negative (see :func:`dn_with_valid`).
            """
            out = []
            valid = None
            for band in ("B11", "B12"):
                href, off = band_href(safe, band)
                window = vrt_window(href, args.lon, args.lat, args.half_km)
                dn, ok, nodata, negative = dn_with_valid(window, off)
                if not quiet:
                    print(f"    {band}: nodata {nodata.mean():>6.1%} | "
                          f"negative-after-offset {negative.mean():>6.1%}")
                out.append(dn / 1e4)
                valid = ok if valid is None else (valid & ok)
            return out[0], out[1], valid

        b11_t, b12_t, valid_t = load_pass(t_safe, quiet=False)

        # ---- optional synthetic injection BEFORE registration --------------
        injected_truth = None
        if args.inject_ppb:
            yy, xx = np.mgrid[0:b11_t.shape[0], 0:b11_t.shape[1]]
            injected_truth = args.inject_ppb * np.exp(
                -(((yy - b11_t.shape[0] / 2) ** 2) / 60
                  + ((xx - b11_t.shape[1] / 2) ** 2) / 60))
            b12_t = b12_t * np.exp(-ALPHA_12 * injected_truth)
            b11_t = b11_t * np.exp(-ALPHA_11 * injected_truth)
            print(f"SYNTHETIC injection: {args.inject_ppb:.0f} ppb peak")

        from plumechaser.retrieve.mbmp import log_band_ratio

        # Invalid pixels travel as NaN from here on: registration and the
        # log-ratio both read NaN as "no pixel", while the 0 DN fill would
        # look like very dark ground.
        b11_t_r = nan_invalid(b11_t, valid_t)
        b12_t_r = nan_invalid(b12_t, valid_t)
        u_t = log_band_ratio(b11_t_r, b12_t_r)
        u_refs = []
        shifts = []
        for i, safe in enumerate(ref_safes):
            b11_i, b12_i, valid_i = load_pass(safe)
            b11_i = nan_invalid(b11_i, valid_i)
            b12_i = nan_invalid(b12_i, valid_i)
            dy, dxp = phase_shift(b11_i, b11_t_r)
            shifts.append((dy, dxp))
            b11_i = shift_array(b11_i, dy, dxp)
            b12_i = shift_array(b12_i, dy, dxp)
            u_refs.append(log_band_ratio(b11_i, b12_i))
            print(f"  ref {i+1}/{n_refs}: {safe} shift=({dy},{dxp})")
        with warnings.catch_warnings():
            # A pixel invalid in every comparison pass is legitimately NaN
            # here; that is the answer, not a problem worth a stack of
            # "All-NaN slice encountered" lines.
            warnings.simplefilter("ignore", RuntimeWarning)
            u_r_med = np.nanmedian(np.stack(u_refs), axis=0)
        t_id = t_safe
        r_id = f"{n_refs}-date median ({ref_safes[-1][:15]}..{ref_safes[0][:15]})"
        t_date = t_day
        sensor_note = (f"Sentinel-2 L1C TOA (public GCS mirror); "
                       f"MBPD {n_refs} comparison dates "
                       f"({n_same_orbit} same-orbit {t_orbit}, "
                       f"{len(cross_orbit)} cross-orbit)")
        scene_extra = {
            "target_safe": t_safe,
            "target_window_coverage": round(1 - t_nd, 4),
            "comparison_safes": ref_safes,
            "comparison_window_coverage": [round(1 - nd, 4) for _, _, nd in refs],
            "platform": plat,
            "target_relative_orbit": t_orbit,
            "same_relative_orbit_refs": n_same_orbit,
            "cross_orbit_refs": [rel_orbit(s) for s in cross_orbit],
        }

    else:
        feats = stac_scenes(args.lon, args.lat,
                            event_date - timedelta(days=25),
                            event_date + timedelta(days=6))
        dated = [(date.fromisoformat(f["properties"]["datetime"][:10]), f)
                 for f in feats]
        cands = [(abs((dt - event_date).days), dt, f) for dt, f in dated
                 if -1 <= (dt - event_date).days <= 5]
        _, t_date, t_feat = min(cands, key=lambda x: (x[0], x[1]))
        refs = [(dt, f) for dt, f in dated
                if dt <= t_date - timedelta(days=5)]
        r_date, r_feat = min(refs, key=lambda x: abs((x[0] - t_date).days))
        print(f"target   : {t_feat['id']} ({t_date})")
        print(f"reference: {r_feat['id']} ({r_date})")
        b11_t = vrt_window(t_feat["assets"]["swir16"]["href"], args.lon, args.lat)
        b12_t = vrt_window(t_feat["assets"]["swir22"]["href"], args.lon, args.lat)
        b11_r = vrt_window(r_feat["assets"]["swir16"]["href"], args.lon, args.lat)
        b12_r = vrt_window(r_feat["assets"]["swir22"]["href"], args.lon, args.lat)
        t_id, r_id = t_feat["id"], r_feat["id"]
        sensor_note = "Sentinel-2 L2A (BOA)"

    dy, dx_px = 0, 0  # gcs path registers per-comparison-date above

    # ---- retrieval --------------------------------------------------------
    from plumechaser.retrieve.mbmp import mbmp_enhancement_ppb, plume_mask, robust_scene_sigma

    if args.source == "gcs-l1c":
        dx_map = (u_t - u_r_med) / (ALPHA_12 - ALPHA_11)
    else:
        dx_map = mbmp_enhancement_ppb(b11_t, b12_t, b11_r, b12_r,
                                      ALPHA_11, ALPHA_12)
    # Every invalid pixel (nodata, or non-positive after the baseline
    # offset) is NaN by construction, so this counts real pixels only.
    valid_frac = float(np.isfinite(dx_map).mean())
    print(f"valid pixels: {valid_frac:.1%} of the window "
          f"({int((~np.isfinite(dx_map)).sum())} masked as "
          f"nodata/negative-DN or uncovered)")
    if valid_frac < 0.5:
        print("WARNING: over half the window is invalid — treat with suspicion")
    sigma_col = robust_scene_sigma(dx_map[np.isfinite(dx_map)])
    mask = plume_mask(dx_map, threshold_sigma=3.0)
    mask_frac = float(mask.mean())
    artifact_dominated = bool(sigma_col > SIGMA_ARTIFACT_LIMIT
                              or mask_frac > MASK_FRAC_LIMIT)
    print(f"sigma_col = {sigma_col:.1f} ppb | plume pixels = {int(mask.sum())} "
          f"({mask_frac:.1%} of window)")
    if injected_truth is not None:
        cy, cx = np.array(injected_truth.shape) // 2
        core = dx_map[cy-15:cy+15, cx-15:cx+15]
        ring = np.concatenate([
            dx_map[cy-60:cy+60, cx-90:cx-45].ravel(),
            dx_map[cy-60:cy+60, cx+45:cx+90].ravel(),
        ])
        core_med = float(np.nanmedian(core))
        ring_med = float(np.nanmedian(ring))
        rec = core_med - ring_med
        print(f"RECOVERY (bg-subtracted): injected {args.inject_ppb:.0f} ppb -> "
              f"{rec:.0f} ppb ({rec/args.inject_ppb:.0%}); "
              f"core med {core_med:.0f}, ring med {ring_med:.0f}")
        if artifact_dominated:
            print("(natural photometric artifacts dominate; recovery still "
                  "validates the radiative-transfer arithmetic)")


    # ---- winds (after variable unification) -------------------------------
    u10 = 4.0
    wind_source = "assumed 4 m/s"
    try:
        from plumechaser.data.openmeteo import openmeteo_winds

        wf = openmeteo_winds(args.lon, args.lat,
                             str(t_date - timedelta(days=2)),
                             str(t_date + timedelta(days=1)))
        if len(wf):
            u10 = float(wf["speed"].median())
            wind_source = f"Open-Meteo/ERA5 median {u10:.2f} m/s"
    except Exception as exc:  # noqa: BLE001
        print(f"winds unavailable: {exc}")
    print(f"winds: {wind_source}")

    result = None
    if mask.sum() >= 20 and not artifact_dominated:
        from plumechaser.retrieve.ime import quantitate

        result = quantitate(np.nan_to_num(dx_map), mask, u10_ms=u10,
                            pixel_area_m2=400.0, mc_samples=300, seed=20270307)
        print(f"Q = {result.q_kg_h:,.0f} kg/h "
              f"[{result.ci_low:,.0f}, {result.ci_high:,.0f}]")
        if args.rate_t_h:
            print(f"catalog rate: {args.rate_t_h:.0f} t/h")
    elif artifact_dominated:
        print("!! ARTIFACT-DOMINATED retrieval — quantification withheld "
              "(honesty gate)")
    else:
        print("no coherent plume above threshold")

    # ---- bundle -----------------------------------------------------------
    from plumechaser.report.bundle import write_bundle
    from plumechaser.report.dossier import DossierInput

    eid = args.event_id or f"EVT-{t_date:%Y%m%d}-{args.basin}"
    verdict = ("no_infrastructure_demo") if not result else "demo_quantified"
    d = DossierInput(
        event_id=eid, basin=args.basin, det_date=str(t_date),
        lon=args.lon, lat=args.lat, event_class="catalog_targeted",
        z_peak=float(np.nanmax(dx_map) / max(sigma_col, 1e-6)),
        persistence_passes=1, persistence_dates=[str(t_date)],
        cue_action="cue_sentinel2",
        cue_reason=f"catalog-targeted retrieval (event {args.date})",
        quant=result, u10_ms=u10, wind_source=wind_source,
        context_verdict=("ARTIFACT-DOMINATED — quantification withheld"
                         if artifact_dominated else verdict),
        provenance=(
            f"sensor: {sensor_note}\n"
            f"target: {t_id}\nreference: {r_id}\n"
            f"registration shift (dy,dx): {(dy, dx_px)}\n"
            f"valid pixels: {valid_frac:.1%}\n"
            f"sigma_col_ppb: {sigma_col:.1f}\n"
            f"alpha_b11/b12_per_ppb: {ALPHA_11}/{ALPHA_12}\n"
            f"honesty_gate: {'ARTIFACT-DOMINATED' if artifact_dominated else 'passed'}"
        ),
    )
    bdir = write_bundle(d, REPO / "bundles", extra={
        "reference_id": r_id,
        "sigma_col_ppb": round(sigma_col, 2),
        "plume_pixels": int(mask.sum()),
        "artifact_dominated": artifact_dominated,
        "valid_fraction": round(valid_frac, 4),
        "calibration": "simplified-alpha demo grade",
        **scene_extra,
    })

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    finite_med = np.nanmedian(dx_map)
    im = ax.imshow(np.nan_to_num(dx_map, nan=finite_med), cmap="RdBu_r",
                   vmin=-3 * sigma_col,
                   vmax=max(3 * sigma_col, float(np.nanmax(dx_map)) * 0.8))
    ax.contour(mask.astype(float), levels=[0.5], colors="k", linewidths=0.7)
    tag = "ARTIFACT-DOMINATED" if artifact_dominated else "gate passed"
    ax.set_title(f"{eid}\ndXCH4 (ppb) — sigma {sigma_col:.0f} · {tag}",
                 fontsize=9)
    plt.colorbar(im, fraction=0.04)
    fig.savefig(bdir / "mbmp_png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    from plumechaser.report.bundle import bundle_integrity

    print(f"bundle: {bdir}")
    print(f"integrity: {bundle_integrity(bdir)[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
