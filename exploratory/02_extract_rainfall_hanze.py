"""
02_extract_rainfall_hanze.py
Daily area-mean rainfall (ERA5-Land, via Earth Engine) for every HANZE event:
import os
from BEFORE days before the event start until AFTER days after it (capped at the end date).

Usage (Windows, from the project folder):
    python 02_extract_rainfall_hanze.py --limit 3        # smoke test on 3 events
    python 02_extract_rainfall_hanze.py                  # full run (resumable)
    python 02_extract_rainfall_hanze.py --all            # also events with no people/loss figure

Needs: earthengine-api, pandas, numpy   (no geopandas / pyshp needed)
Reads : <processed>/hanze_events.csv, <processed>/hanze_event_units.csv
        + the NUTS3 region shapefile that ships with HANZE (Regions_v2024_simplified.zip)
Writes: <processed>/hanze_rain_daily.csv   (hanze_id, date, rel_day, precip_mm, used_centroid)
        <processed>/hanze_rain_failed.txt  (ids that failed; re-run to retry them)
"""
import argparse
import io
import math
import struct
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("FLOOD_DATA_DIR", "data"))
DATASET = "ECMWF/ERA5_LAND/DAILY_AGGR"
BAND = "total_precipitation_sum"      # metres of water per day
SCALE_M = 11132                        # native ERA5-Land pixel size


# ----------------------------------------------------------------------------
# Minimal shapefile + DBF readers and ETRS89-LAEA (EPSG:3035) -> lon/lat
# (the HANZE regions are stored in metres, Earth Engine needs lon/lat)
# ----------------------------------------------------------------------------
def read_dbf(data, encoding="utf-8"):
    n_rec = struct.unpack("<I", data[4:8])[0]
    hdr_len, rec_len = struct.unpack("<HH", data[8:12])
    fields, pos = [], 32
    while data[pos:pos + 1] != b"\r":
        d = data[pos:pos + 32]
        fields.append((d[:11].split(b"\x00")[0].decode("ascii"), d[16]))
        pos += 32
    rows = []
    for i in range(n_rec):
        rec = data[hdr_len + i * rec_len: hdr_len + (i + 1) * rec_len]
        p, row = 1, {}
        for name, ln in fields:
            row[name] = rec[p:p + ln].decode(encoding, errors="replace").strip()
            p += ln
        rows.append(row)
    return rows


def read_shp_polygons(data):
    """List (one per record) of rings; each ring is an (n, 2) array."""
    pos, out = 100, []
    while pos + 8 <= len(data):
        _, clen = struct.unpack(">ii", data[pos:pos + 8])
        pos += 8
        content = data[pos:pos + clen * 2]
        pos += clen * 2
        stype = struct.unpack("<i", content[:4])[0]
        if stype == 0:
            out.append([])
            continue
        if stype not in (5, 15, 25):
            raise ValueError(f"unsupported shape type {stype}")
        nparts, npts = struct.unpack("<ii", content[36:44])
        parts = list(struct.unpack(f"<{nparts}i", content[44:44 + 4 * nparts])) + [npts]
        p0 = 44 + 4 * nparts
        pts = np.frombuffer(content[p0:p0 + 16 * npts], dtype="<f8").reshape(npts, 2)
        out.append([pts[parts[i]:parts[i + 1]] for i in range(nparts)])
    return out


_A = 6378137.0
_F = 1 / 298.257222101
_E2 = 2 * _F - _F * _F
_E = math.sqrt(_E2)
_LAT0, _LON0 = math.radians(52.0), math.radians(10.0)
_FE, _FN = 4321000.0, 3210000.0


def _q(phi):
    s = np.sin(phi)
    return (1 - _E2) * (s / (1 - _E2 * s * s) - (1 / (2 * _E)) * np.log((1 - _E * s) / (1 + _E * s)))


_QP, _Q0 = _q(math.pi / 2), _q(_LAT0)
_B0 = math.asin(_Q0 / _QP)
_RQ = _A * math.sqrt(_QP / 2)
_D = _A * (math.cos(_LAT0) / math.sqrt(1 - _E2 * math.sin(_LAT0) ** 2)) / (_RQ * math.cos(_B0))


def laea_inverse(xy):
    x, y = xy[:, 0] - _FE, xy[:, 1] - _FN
    rho = np.sqrt((x / _D) ** 2 + (_D * y) ** 2)
    ce = 2 * np.arcsin(np.clip(rho / (2 * _RQ), -1, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        sinb = np.where(rho > 0, np.cos(ce) * math.sin(_B0) + _D * y * np.sin(ce) * math.cos(_B0) / rho,
                        math.sin(_B0))
    beta = np.arcsin(np.clip(sinb, -1, 1))
    lon = _LON0 + np.arctan2(x * np.sin(ce), _D * rho * math.cos(_B0) * np.cos(ce)
                             - _D * _D * y * math.sin(_B0) * np.sin(ce))
    lat = (beta + (_E2 / 3 + 31 * _E2 ** 2 / 180 + 517 * _E2 ** 3 / 5040) * np.sin(2 * beta)
           + (23 * _E2 ** 2 / 360 + 251 * _E2 ** 3 / 3780) * np.sin(4 * beta)
           + (761 * _E2 ** 3 / 45360) * np.sin(6 * beta))
    return np.degrees(np.column_stack([lon, lat]))


MIN_RING_M2 = 1e5   # rings below 0.1 km2 are simplification slivers (invalid for Earth Engine)


def shoelace(ring):
    """Signed area in projected metres^2; shapefile exterior rings are clockwise (negative)."""
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])


def unit_polygons(rings):
    """Rings of one NUTS3 unit -> list of polygons [[exterior, hole, ...], ...] in lon/lat.
    Drops sliver rings and keeps every hole attached to its own exterior ring."""
    rings = [r for r in rings if len(r) >= 4]
    if not rings:
        return []
    biggest = max(rings, key=lambda r: abs(shoelace(r)))
    polys, current = [], None
    for ring in rings:
        area = shoelace(ring)
        keep = abs(area) >= MIN_RING_M2 or ring is biggest
        ll = np.round(laea_inverse(ring), 5).tolist() if keep else None
        if area < 0:                       # exterior ring
            if keep:
                polys.append([ll])
                current = polys[-1]
            else:
                current = None             # also skip the holes that belong to it
        elif keep and current is not None:  # hole ring
            current.append(ll)
    if not polys:                          # unexpected orientation: treat large rings as exteriors
        polys = [[np.round(laea_inverse(r), 5).tolist()] for r in rings
                 if abs(shoelace(r)) >= MIN_RING_M2 or r is biggest]
    return polys


def load_nuts3_polygons(root):
    shp = next(iter(sorted(root.rglob("NUTS3_regions_v2024_simplified.shp"))), None)
    if shp is not None:
        shp_b, dbf_b = shp.read_bytes(), shp.with_suffix(".dbf").read_bytes()
    else:
        zips = sorted(root.rglob("Regions_v2024_simplified.zip"))
        if not zips:
            raise FileNotFoundError("Regions_v2024_simplified.zip (or the NUTS3 .shp) not found under " + str(root))
        with zipfile.ZipFile(zips[0]) as z:
            shp_b = z.read("NUTS3_regions_v2024_simplified.shp")
            dbf_b = z.read("NUTS3_regions_v2024_simplified.dbf")
    codes = [r["Code"] for r in read_dbf(dbf_b)]
    shapes = read_shp_polygons(shp_b)
    assert len(codes) == len(shapes)
    return {c: unit_polygons(s) for c, s in zip(codes, shapes)}


# ----------------------------------------------------------------------------
# Earth Engine
# ----------------------------------------------------------------------------
def init_ee():
    import ee
    ee.Initialize(project=os.environ.get("EE_PROJECT"))
    return ee


def fetch_series(ee, poly_coords, start, end_excl, use_centroid=False):
    """Daily area-mean rainfall as [(date_str, metres or None), ...]."""
    geom = ee.Geometry.MultiPolygon(poly_coords, geodesic=False)
    if use_centroid:
        geom = geom.centroid(1)
    ic = ee.ImageCollection(DATASET).filterDate(start, end_excl).select(BAND)

    def per_day(img):
        v = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom,
                             scale=SCALE_M, maxPixels=1e9).get(BAND)
        return ee.Feature(None, {"date": img.date().format("YYYY-MM-dd"), "v": v})

    feats = ee.FeatureCollection(ic.map(per_day)).getInfo()["features"]
    return [(f["properties"]["date"], f["properties"].get("v")) for f in feats]


def event_rows(ee, ev, polys, before, after, retries=3):
    start = pd.Timestamp(ev.start_date)
    end = min(pd.Timestamp(ev.end_date), start + pd.Timedelta(days=after))
    w0, w1 = start - pd.Timedelta(days=before), end + pd.Timedelta(days=1)
    for centroid in (False, True):
        for attempt in range(retries):
            try:
                series = fetch_series(ee, polys, w0.strftime("%Y-%m-%d"), w1.strftime("%Y-%m-%d"), centroid)
                break
            except Exception as exc:  # network / quota hiccup
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        if any(v is not None for _, v in series):
            return [(ev.hanze_id, d, (pd.Timestamp(d) - start).days,
                     None if v is None else v * 1000.0, centroid) for d, v in series]
    return []  # nothing at all returned


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT), help="data folder containing raw/ and processed/")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--before", type=int, default=30, help="days before event start")
    ap.add_argument("--after", type=int, default=14, help="max days after event start")
    ap.add_argument("--all", action="store_true", help="include events without a people/loss figure")
    args = ap.parse_args()

    root = Path(args.root)
    proc = root / "processed"
    out_csv, fail_txt = proc / "hanze_rain_daily.csv", proc / "hanze_rain_failed.txt"

    ev = pd.read_csv(proc / "hanze_events.csv", parse_dates=["start_date", "end_date"])
    units = pd.read_csv(proc / "hanze_event_units.csv")
    ev = ev[ev["year_ge_1950"]]                      # ERA5-Land starts in 1950
    if not args.all:
        ev = ev[ev["people_affected"].notna() | ev["loss_eur_2025"].notna()]
    done = set(pd.read_csv(out_csv, usecols=["hanze_id"])["hanze_id"]) if out_csv.exists() else set()
    ev = ev[~ev["hanze_id"].isin(done)]
    if args.limit:
        ev = ev.head(args.limit)
    print(f"events to process: {len(ev)} (already done: {len(done)})")
    if ev.empty:
        return

    nuts = load_nuts3_polygons(root / "raw")
    by_event = units.groupby("hanze_id")["nuts3"].apply(list).to_dict()
    ee = init_ee()

    def job(row):
        polys = [p for code in by_event[row.hanze_id] for p in nuts[code]]
        return row.hanze_id, event_rows(ee, row, polys, args.before, args.after)

    t0, n_ok, failed = time.time(), 0, []
    header = not out_csv.exists()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(job, r): r.hanze_id for r in ev.itertuples(index=False)}
        for i, fut in enumerate(as_completed(futs), 1):
            hid = futs[fut]
            try:
                _, rows = fut.result()
                if not rows:
                    raise RuntimeError("no rainfall values returned")
                pd.DataFrame(rows, columns=["hanze_id", "date", "rel_day", "precip_mm", "used_centroid"]) \
                  .to_csv(out_csv, mode="a", header=header, index=False)
                header = False
                n_ok += 1
            except Exception as exc:
                failed.append(f"{hid}\t{exc}")
            if i % 25 == 0 or i == len(futs):
                el = time.time() - t0
                print(f"  {i}/{len(futs)} done | ok {n_ok} | failed {len(failed)} | "
                      f"{el / 60:.1f} min elapsed | ETA {el / i * (len(futs) - i) / 60:.1f} min")
    fail_txt.write_text("\n".join(failed), encoding="utf-8")
    print(f"finished: {n_ok} ok, {len(failed)} failed (see {fail_txt.name}). Re-run to retry failures.")


if __name__ == "__main__":
    main()
