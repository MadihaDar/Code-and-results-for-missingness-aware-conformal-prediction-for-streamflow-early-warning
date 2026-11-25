"""Flood-day test: coverage and warnings on documented HANZE flood days."""
import os
import argparse
import glob
import struct
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(os.environ.get("FLOOD_DATA_DIR", "data"))
ORDER = ["none", "q_delay3", "q_delay7", "q_blocks", "combined"]
METHODS = ["complete", "mondrian", "mondrian_basin", "mondrian_flow"]


# ---- NUTS3 polygons (metre-based ETRS89-LAEA -> lon/lat), minimal readers ----
def read_dbf(d):
    n = struct.unpack("<I", d[4:8])[0]
    hl, rl = struct.unpack("<HH", d[8:12])
    fields, pos = [], 32
    while d[pos:pos + 1] != b"\r":
        fields.append((d[pos:pos + 11].split(b"\x00")[0].decode("ascii"), d[pos + 16]))
        pos += 32
    out = []
    for i in range(n):
        rec, p, row = d[hl + i * rl: hl + (i + 1) * rl], 1, {}
        for name, ln in fields:
            row[name] = rec[p:p + ln].decode("utf-8", errors="replace").strip()
            p += ln
        out.append(row)
    return out


def read_shp(d):
    pos, out = 100, []
    while pos + 8 <= len(d):
        _, cl = struct.unpack(">ii", d[pos:pos + 8])
        c = d[pos + 8: pos + 8 + cl * 2]
        pos += 8 + cl * 2
        if struct.unpack("<i", c[:4])[0] == 0:
            out.append([])
            continue
        npart, npts = struct.unpack("<ii", c[36:44])
        parts = list(struct.unpack(f"<{npart}i", c[44:44 + 4 * npart])) + [npts]
        pts = np.frombuffer(c[44 + 4 * npart: 44 + 4 * npart + 16 * npts], dtype="<f8").reshape(npts, 2)
        out.append([pts[parts[i]:parts[i + 1]] for i in range(npart)])
    return out


A, F = 6378137.0, 1 / 298.257222101
E2 = 2 * F - F * F
E = math.sqrt(E2)
LAT0, LON0, FE, FN = math.radians(52.0), math.radians(10.0), 4321000.0, 3210000.0


def _q(p):
    s = np.sin(p)
    return (1 - E2) * (s / (1 - E2 * s * s) - np.log((1 - E * s) / (1 + E * s)) / (2 * E))


QP, Q0 = _q(math.pi / 2), _q(LAT0)
B0, RQ = math.asin(Q0 / QP), A * math.sqrt(QP / 2)
D = A * (math.cos(LAT0) / math.sqrt(1 - E2 * math.sin(LAT0) ** 2)) / (RQ * math.cos(B0))


def laea_inv(xy):
    x, y = xy[:, 0] - FE, xy[:, 1] - FN
    rho = np.sqrt((x / D) ** 2 + (D * y) ** 2)
    ce = 2 * np.arcsin(np.clip(rho / (2 * RQ), -1, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        sb = np.where(rho > 0, np.cos(ce) * math.sin(B0) + D * y * np.sin(ce) * math.cos(B0) / rho, math.sin(B0))
    b = np.arcsin(np.clip(sb, -1, 1))
    lon = LON0 + np.arctan2(x * np.sin(ce), D * rho * math.cos(B0) * np.cos(ce) - D * D * y * math.sin(B0) * np.sin(ce))
    lat = (b + (E2 / 3 + 31 * E2 ** 2 / 180 + 517 * E2 ** 3 / 5040) * np.sin(2 * b)
           + (23 * E2 ** 2 / 360 + 251 * E2 ** 3 / 3780) * np.sin(4 * b) + (761 * E2 ** 3 / 45360) * np.sin(6 * b))
    return np.degrees(np.column_stack([lon, lat]))


def shoelace(r):
    return 0.5 * np.sum(r[:-1, 0] * r[1:, 1] - r[1:, 0] * r[:-1, 1])


def unit_polys(rings):
    """[(exterior, [holes])] in lon/lat; sliver rings dropped."""
    rings = [r for r in rings if len(r) >= 4]
    big = max(rings, key=lambda r: abs(shoelace(r)))
    polys, cur = [], None
    for r in rings:
        a = shoelace(r)
        keep = abs(a) >= 1e5 or r is big
        ll = laea_inv(r) if keep else None
        if a < 0:
            cur = None
            if keep:
                polys.append((ll, []))
                cur = polys[-1][1]
        elif keep and cur is not None:
            cur.append(ll)
    return polys or [(laea_inv(big), [])]


def load_nuts(root):
    shp = next(iter(sorted(root.rglob("NUTS3_regions_v2024_simplified.shp"))), None)
    if shp is not None:
        sb, db = shp.read_bytes(), shp.with_suffix(".dbf").read_bytes()
    else:
        z = next(iter(sorted(root.rglob("Regions_v2024_simplified.zip"))))
        with zipfile.ZipFile(z) as zf:
            sb, db = zf.read("NUTS3_regions_v2024_simplified.shp"), zf.read("NUTS3_regions_v2024_simplified.dbf")
    return {r["Code"]: unit_polys(s) for r, s in zip(read_dbf(db), read_shp(sb))}


def in_ring(x, y, r):
    x1, y1, x2, y2 = r[:-1, 0], r[:-1, 1], r[1:, 0], r[1:, 1]
    c = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / np.where(y2 == y1, 1e-12, y2 - y1) + x1)
    return np.count_nonzero(c) % 2 == 1


def locate(lon, lat, nuts):
    for code, polys in nuts.items():
        for ext, holes in polys:
            if ext[:, 0].min() <= lon <= ext[:, 0].max() and ext[:, 1].min() <= lat <= ext[:, 1].max():
                if in_ring(lon, lat, ext) and not any(in_ring(lon, lat, h) for h in holes):
                    return code
    return None


# ---- event-day matrix ----
def event_matrix(nb, T, dates, gids, static, nuts, ev, units, before, after):
    st = static.set_index("gauge_id").reindex(gids)
    unit = [locate(r.gauge_lon, r.gauge_lat, nuts) for r in st.itertuples()]
    by_unit = {}
    for b, u in enumerate(unit):
        by_unit.setdefault(u, []).append(b)
    dix = pd.Series(np.arange(T), index=pd.DatetimeIndex(dates))
    eid = np.full((nb, T), -1, np.int32)
    pairs = 0
    for e, row in enumerate(ev.itertuples()):
        d0 = dix.index.searchsorted(pd.Timestamp(row.start_date) - pd.Timedelta(days=before))
        d1 = dix.index.searchsorted(min(pd.Timestamp(row.end_date), pd.Timestamp(row.start_date) + pd.Timedelta(days=after)), side="right")
        for u in units.get(row.hanze_id, []):
            for b in by_unit.get(u, []):
                seg = eid[b, d0:d1]
                seg[seg < 0] = e
                pairs += 1
    return eid, unit, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(BASE))
    ap.add_argument("--tag", default="rl")
    ap.add_argument("--before", type=int, default=1)
    ap.add_argument("--after", type=int, default=7)
    ap.add_argument("--types", nargs="+", default=["River", "Flash", "River/Coastal"])
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--test-start", default="2008-01-01")
    ap.add_argument("--test-end", default="2017-12-31")
    args = ap.parse_args()
    base, proc = Path(args.base), Path(args.base) / "processed"
    rng = np.random.default_rng(0)
    static = pd.read_csv(proc / "trackA_static.csv")
    ev = pd.read_csv(proc / "hanze_events.csv", parse_dates=["start_date", "end_date"])
    units = pd.read_csv(proc / "hanze_event_units.csv").groupby("hanze_id").nuts3.apply(list).to_dict()
    ev = ev[(ev.start_date >= args.test_start) & (ev.start_date <= args.test_end) & ev.event_type.isin(args.types)].reset_index(drop=True)
    nuts = load_nuts(base / "raw")
    print(f"HANZE events in test period ({', '.join(args.types)}): {len(ev)}")

    out, out_w = [], []
    for f in sorted(glob.glob(str(proc / f"trackA_rows_{args.tag}_lead*.npz"))):
        lead = int(f.split("lead")[-1].split(".")[0])
        z = np.load(f)
        gids, dates, thr, y, bi, dt = z["gauge_ids"], z["dates"], z["thr"], z["y"], z["bi"].astype(int), z["dt"].astype(int)
        eid_m, unit, pairs = event_matrix(len(gids), len(dates), dates, gids, static, nuts, ev, units, args.before, args.after)
        eid = eid_m[bi, dt]
        isev = eid >= 0
        hi_obs = y >= thr[bi]
        ids = np.unique(eid[isev])
        remap = np.full(len(ev), -1)
        remap[ids] = np.arange(len(ids))
        e_idx = remap[eid[isev]] if isev.any() else np.array([], int)
        print(f"\nlead {lead}: basins {len(gids)} | gauges located in a NUTS3 unit: {sum(u is not None for u in unit)} | "
              f"basin-event pairs {pairs} | event rows {int(isev.sum())} | events with data {len(ids)}")
        if not isev.any():
            continue
        print(f"  high-flow share on event days {hi_obs[isev].mean():.3f} vs all days {hi_obs.mean():.3f} "
              f"(enrichment x{hi_obs[isev].mean() / max(hi_obs.mean(), 1e-9):.1f}; sanity check of the matching)")
        boot = rng.integers(0, len(ids), size=(args.reps, len(ids)))
        for key in sorted({k.split("|")[0] for k in z.files if "|" in k}):
            for sc in ORDER:
                if f"{key}|{sc}|pt" not in z.files:
                    continue
                pt, lo, hi = z[f"{key}|{sc}|pt"], z[f"{key}|{sc}|lo"], z[f"{key}|{sc}|hi"]
                cnt = lambda v: np.bincount(e_idx, weights=v[isev].astype(float), minlength=len(ids))
                if f"{key}|{sc}|lamU" in z.files:      # risk-controlled warnings (target recall 0.90)
                    kinds = {"U_bins": hi + z[f"{key}|{sc}|lamU"] >= thr[bi], "U_complete": hi + z[f"{key}|{sc}|lamU_c"][0] >= thr[bi],
                             "P_bins": pt + z[f"{key}|{sc}|lamP"] >= thr[bi]}
                    h_e = cnt(hi_obs)
                    for kind, wn in kinds.items():
                        r_e = cnt(wn & hi_obs)
                        bs = r_e[boot].sum(1) / np.maximum(h_e[boot].sum(1), 1)
                        lowflow = ~isev & ~hi_obs
                        out_w.append(dict(lead=lead, model=key, scenario=sc, kind=kind, recall_event=r_e.sum() / max(h_e.sum(), 1),
                                          recall_lo=np.percentile(bs, 2.5), recall_hi=np.percentile(bs, 97.5),
                                          n_hi_event_rows=int(h_e.sum()), false_alarm_lowflow=wn[lowflow].mean()))
                for m in METHODS:
                    if f"{key}|{sc}|w_{m}" not in z.files:
                        continue
                    w = z[f"{key}|{sc}|w_{m}"]
                    L, U = lo - w, hi + w
                    cov, above = (y >= L) & (y <= U), y > U
                    w_up, w_pt = U >= thr[bi], pt >= thr[bi]
                    n_e, c_e, a_e = cnt(np.ones(len(y))), cnt(cov), cnt(above)
                    h_e = cnt(hi_obs)
                    r_up, r_pt = cnt(w_up & hi_obs), cnt(w_pt & hi_obs)
                    rec = dict(lead=lead, model=key, scenario=sc, method=m, event_rows=int(n_e.sum()), events=len(ids))
                    for name, num, den in (("coverage_event", c_e, n_e), ("miss_above_event", a_e, n_e),
                                           ("recall_upper", r_up, h_e), ("recall_point", r_pt, h_e)):
                        rec[name] = num.sum() / max(den.sum(), 1)
                        bs = num[boot].sum(1) / np.maximum(den[boot].sum(1), 1)
                        rec[name + "_lo"], rec[name + "_hi"] = np.percentile(bs, [2.5, 97.5])
                    ne = ~isev
                    rec["coverage_other"] = cov[ne].mean()
                    rec["warn_rate_upper_lowflow"] = w_up[ne & ~hi_obs].mean()
                    rec["warn_rate_point_lowflow"] = w_pt[ne & ~hi_obs].mean()
                    out.append(rec)
    if not out:
        return
    res = pd.DataFrame(out)
    res.to_csv(proc / f"trackA_floodday_{args.tag}.csv", index=False)
    resw = pd.DataFrame(out_w)
    if not resw.empty:
        resw.to_csv(proc / f"trackA_floodday_warnings_{args.tag}.csv", index=False)
    pd.set_option("display.width", 220)
    for (lead, model), g in res.groupby(["lead", "model"]):
        t = pd.DataFrame({m: g[g.method == m].set_index("scenario").apply(
            lambda r: f"{r.coverage_event:.3f} [{r.coverage_event_lo:.3f}-{r.coverage_event_hi:.3f}]", axis=1) for m in METHODS}).reindex(ORDER)
        print(f"\n[lead {lead}] [{model}] coverage on flood days [95% CI over events]\n{t.to_string()}")
        g2 = g[g.method == "mondrian"].set_index("scenario").reindex(ORDER)
        t2 = pd.DataFrame({"recall high flow, upper bound": g2.recall_upper.round(3), "recall, point forecast": g2.recall_point.round(3),
                           "warn rate low-flow days (upper)": g2.warn_rate_upper_lowflow.round(3),
                           "warn rate low-flow days (point)": g2.warn_rate_point_lowflow.round(3)})
        print(f"\n[lead {lead}] [{model}] warnings with the missingness-bin upper bound vs the point forecast\n{t2.to_string()}")
    for (lead, model), g in resw.groupby(["lead", "model"]) if not resw.empty else []:
        t = pd.DataFrame({k: g[g.kind == k].set_index("scenario").apply(
            lambda r: f"{r.recall_event:.3f} [{r.recall_lo:.2f}-{r.recall_hi:.2f}]", axis=1) for k in ["U_bins", "U_complete", "P_bins"]}).reindex(ORDER)
        fa = pd.DataFrame({f"FA {k}": g[g.kind == k].set_index("scenario").false_alarm_lowflow.round(3) for k in ["U_bins", "U_complete", "P_bins"]}).reindex(ORDER)
        print(f"\n[lead {lead}] [{model}] risk-controlled warnings: recall of high-flow flood days [95% CI over events] and false-alarm rate on non-flood low-flow days\n{pd.concat([t, fa], axis=1).to_string()}")
    print(f"\nsaved trackA_floodday_{args.tag}.csv")


if __name__ == "__main__":
    main()
