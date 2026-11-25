"""Track A runs: seeds, lead times, region transfer, block outages."""
import os
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

PROC = Path(os.environ.get("FLOOD_DATA_DIR", "data")) / "processed"
EPS = 0.01                                   # log(q + EPS), q in mm/d
TRAIN, CAL, TEST = ("1981-01-01", "1999-12-31"), ("2000-01-01", "2007-12-31"), ("2008-01-01", "2017-12-31")
SCEN = {                                     # q_drop, q_delay, forcing_delay, block outages
    "none": (0, 0, 0, 0), "q_drop30": (.3, 0, 0, 0), "q_drop50": (.5, 0, 0, 0), "q_delay3": (0, 3, 0, 0),
    "q_delay7": (0, 7, 0, 0), "f_delay1": (0, 0, 1, 0), "f_delay3": (0, 0, 3, 0), "q_blocks": (0, 0, 0, 1),
    "combined": (.3, 3, 1, 0)}
FORCING = ("p", "t", "swe", "sm1", "sm2", "sm3", "pet")
STATIC = ["frac_snow", "aridity_ERA5_LAND", "p_mean", "seasonality_ERA5_LAND", "high_prec_freq",
          "moisture_index_ERA5_LAND", "area"]
NAMES = ["lq_last", "age_q", "lq_mean7", "lq_max7", "p0", "p1", "p2", "p_sum3", "p_sum7", "p_sum14", "p_sum30",
         "t0", "t_mean7", "swe0", "dswe7", "sm1", "sm2", "sm3", "pet_sum7", "f_delay", "sin_doy", "cos_doy"] + STATIC
AGE_COL, FD_COL = NAMES.index("age_q"), NAMES.index("f_delay")
PB_METHODS = ("complete", "mondrian", "mondrian_region", "mondrian_basin", "mondrian_flow", "exactmask", "scenario")
ALPHA_W = (0.05, 0.1)                        # target miss rates for high-flow warnings
ROW_SCEN = ("none", "q_delay3", "q_delay7", "q_blocks", "combined")
ROW_METHODS = ("complete", "mondrian", "mondrian_basin", "mondrian_flow")
KEYS = 310                                   # key = staleness (0-30) * 10 + forcing delay


def shift(a, k):                             # value at t-k (k<0: future)
    if k == 0:
        return a.copy()
    out = np.full_like(a, np.nan)
    if k > 0:
        out[k:] = a[:-k]
    else:
        out[:k] = a[-k:]
    return out


def roll(a, w, fn):
    return getattr(pd.DataFrame(a).rolling(w, min_periods=1), fn)().to_numpy()


def block_mask(T, B, rng, down=10, up=40):   # outage blocks, ~20% downtime
    m = np.zeros((T, B), bool)
    for b in range(B):
        t = int(rng.geometric(1 / up))
        while t < T:
            d = int(rng.geometric(1 / down))
            m[t:t + d, b] = True
            t += d + int(rng.geometric(1 / up))
    return m


def build(raw, dates, static, scen, rng):
    """Mask/delay raw inputs, then compute features. Returns X (T,B,F), staleness age (T,B), forcing delay."""
    qd, kq, kf, blk = SCEN[scen]
    q = raw["q"].copy()
    if qd:
        q[rng.random(q.shape) < qd] = np.nan
    if blk:
        q[block_mask(*q.shape, rng)] = np.nan
    q = shift(q, kq)
    f = {k: shift(raw[k], kf) for k in FORCING}
    T, B = q.shape
    lq = np.log(q + EPS)
    t_idx = np.arange(T)[:, None]
    last = np.maximum.accumulate(np.where(np.isfinite(lq), t_idx, -1), axis=0)
    age = np.where(last >= 0, t_idx - last + kq, 99).astype(np.float32)
    lq_last = np.where(last >= 0, np.take_along_axis(lq, np.maximum(last, 0), axis=0), np.nan)
    doy = pd.DatetimeIndex(dates).dayofyear.to_numpy()[:, None] * np.ones((1, B))
    feats = {
        "lq_last": lq_last, "age_q": np.minimum(age, 30), "lq_mean7": roll(lq, 7, "mean"), "lq_max7": roll(lq, 7, "max"),
        "p0": f["p"], "p1": shift(f["p"], 1), "p2": shift(f["p"], 2), "p_sum3": roll(f["p"], 3, "sum"),
        "p_sum7": roll(f["p"], 7, "sum"), "p_sum14": roll(f["p"], 14, "sum"), "p_sum30": roll(f["p"], 30, "sum"),
        "t0": f["t"], "t_mean7": roll(f["t"], 7, "mean"), "swe0": f["swe"], "dswe7": f["swe"] - shift(f["swe"], 7),
        "sm1": f["sm1"], "sm2": f["sm2"], "sm3": f["sm3"], "pet_sum7": roll(f["pet"], 7, "sum"),
        "f_delay": np.full((T, B), kf), "sin_doy": np.sin(2 * np.pi * doy / 365.25),
        "cos_doy": np.cos(2 * np.pi * doy / 365.25)}
    for c in STATIC:
        feats[c] = np.ones((T, 1)) * static[c].to_numpy()[None, :]
    assert list(feats) == NAMES
    return np.stack([v.astype(np.float32) for v in feats.values()], axis=-1), age, kf


def bin_id(age, kf):                         # staleness bucket (0 | 1-2 | 3-6 | 7+) x forcing delayed?
    return np.digitize(age, [1, 3, 7]) * 2 + (1 if kf > 0 else 0)


def qhat(s, alpha):
    n = len(s)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n) - 1
    return np.partition(s, k)[k]


def group_lut(pool, g1, g2, n1, n2, fallback, alpha, min_n=200):
    """Quantile per (g1, g2) group; groups with fewer than min_n rows keep the fallback."""
    lut, key = fallback.copy(), g1 * n2 + g2
    order = np.argsort(key, kind="stable")
    ks, ps = key[order], pool[order]
    cut = np.flatnonzero(np.diff(ks)) + 1
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(ks)]):
        if b - a >= min_n:
            lut[ks[a] // n2, ks[a] % n2] = qhat(ps[a:b], alpha)
    return lut


def period(dates, p):
    d = pd.DatetimeIndex(dates)
    return np.where((d >= p[0]) & (d <= p[1]))[0]


def run(args, seed, lead, z, st_all):
    rng = np.random.default_rng(seed)
    srcs = np.unique(z["source"])
    pick = np.concatenate([np.sort(rng.choice(np.where(z["source"] == s)[0],
                                              min(args.n_per_source, (z["source"] == s).sum()), replace=False)) for s in srcs])
    data, dates, src = z["data"][pick], z["dates"], z["source"][pick]
    st = st_all.iloc[pick].reset_index(drop=True)
    raw = {n: data[:, :, i].T for i, n in enumerate(z["var_names"])}
    Y = shift(np.log(raw["q"] + EPS), -lead)
    tr, ca, te = (period(dates, p) for p in (TRAIN, CAL, TEST))
    thr = np.nanquantile(Y[tr], 0.95, axis=0)
    qcls = np.nanquantile(Y[tr], [0.5, 0.9, 0.95], axis=0)
    fcls = lambda pt, bi: (pt >= qcls[0][bi]).astype(int) + (pt >= qcls[1][bi]) + (pt >= qcls[2][bi])   # forecast level 0-3
    gids = z["gauge_ids"][pick]
    rix = np.searchsorted(srcs, src)                       # region index per basin
    train_src = args.train_source or list(srcs)
    cal_src = args.cal_source or train_src
    test_src = args.test_source or list(srcs)
    hold = np.zeros(len(pick), bool)
    if args.holdout > 0:                                    # basins unseen in training and calibration
        hr = np.random.default_rng(seed + 12345)
        for r in range(len(srcs)):
            idx = np.where(rix == r)[0]
            hold[hr.choice(idx, int(round(args.holdout * len(idx))), replace=False)] = True

    F = {}
    for s in SCEN:
        X, age, kf = build(raw, dates, st, s, rng)
        F[s] = {}
        for name, ix in (("tr", tr), ("ca", ca), ("te", te)):
            y = Y[ix]
            ok = np.isfinite(y) & np.isfinite(raw["q"][ix])
            bi = np.broadcast_to(np.arange(len(pick))[None, :], y.shape)
            tt = np.broadcast_to(np.arange(len(ix))[:, None], y.shape)
            F[s][name] = (X[ix][ok], y[ok], bin_id(age[ix][ok], kf), bi[ok], tt[ok])
        del X, age

    def get(name, s, sources, stride=1, frac=1.0):
        X, y, b, bi, tt = F[s][name]
        keep = np.isin(src[bi], sources) & (tt % stride == 0)
        if args.holdout > 0:
            keep &= (hold[bi] if name == "te" else ~hold[bi])
        if frac < 1:
            keep &= rng.random(len(y)) < frac
        return X[keep], y[keep], b[keep], bi[keep], tt[keep]

    kw = dict(max_iter=args.max_iter, learning_rate=0.1, l2_regularization=1.0, random_state=0)
    lo_q, hi_q = args.alpha / 2, 1 - args.alpha / 2
    models = {}
    for mname, (scens, frac) in {"trained_complete": (["none"], 1.0), "trained_mixed": (list(SCEN), 0.25)}.items():
        parts = [get("tr", s, train_src, args.train_stride, frac) for s in scens]
        Xtr, ytr = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
        models[mname] = [HGB(**kw).fit(Xtr, ytr), HGB(loss="quantile", quantile=lo_q, **kw).fit(Xtr, ytr),
                         HGB(loss="quantile", quantile=hi_q, **kw).fit(Xtr, ytr)]

    def predict(ms, X):
        pt, lo, hi = (m.predict(X) for m in ms)
        return pt, np.minimum(lo, hi), np.maximum(lo, hi)

    def codes(X):
        return np.minimum(X[:, AGE_COL], 30).astype(int), X[:, FD_COL].astype(int)

    rows, basin_rows, rowbuf, warn_rows = [], [], {}, []
    for mname, ms in models.items():
        cal = {}
        for s in SCEN:
            Xc, yc, bc, bic, _ = get("ca", s, cal_src)
            pt, lo, hi = predict(ms, Xc)
            ac, kc = codes(Xc)
            cal[s] = dict(sc=np.abs(yc - pt), cqr=np.maximum(lo - yc, yc - hi), b=bc, a=ac, k=kc, r=rix[bic], i=bic,
                          y=yc, hi=hi, pt=pt, f=fcls(pt, bic))
        pb, pa, pk, pr, pbi, pf = (np.concatenate([cal[s][c] for s in SCEN]) for c in ("b", "a", "k", "r", "i", "f"))
        present = np.unique(pa * 10 + pk)
        qh = {}
        for sc in ("sc", "cqr"):
            pool = np.concatenate([cal[s][sc] for s in SCEN])
            qp = qhat(pool, args.alpha)
            bins = np.array([qhat(pool[pb == b], args.alpha) if (pb == b).sum() >= 200 else qp for b in range(8)])
            lut_br = np.tile(bins[:, None], (1, len(srcs)))
            for b in range(8):
                for r in range(len(srcs)):
                    m_br = (pb == b) & (pr == r)
                    if m_br.sum() >= 200:
                        lut_br[b, r] = qhat(pool[m_br], args.alpha)
            lut_bb = group_lut(pool, pb, pbi, 8, len(pick), lut_br[:, rix], args.alpha) if sc == "cqr" else None
            lut_bf = group_lut(pool, pb, pf, 8, 4, np.tile(bins[:, None], (1, 4)), args.alpha) if sc == "cqr" else None
            fallback = np.array([bins[bin_id(np.array([key // 10]), key % 10)[0]] for key in range(KEYS)])
            lut_e, lut_n = fallback.copy(), np.full(KEYS, qp)
            for key in present:                 # exact mask (staleness, forcing delay) / nested supersets
                a_, k_ = key // 10, key % 10
                m_e = (pa == a_) & (pk == k_)
                if m_e.sum() >= 200:
                    lut_e[key] = qhat(pool[m_e], args.alpha)
                lut_n[key] = qhat(pool[(pa >= a_) & (pk >= k_)], args.alpha)
            qh[sc] = dict(complete=qhat(cal["none"][sc], args.alpha), pooled=qp, lut=bins, lut_br=lut_br, lut_bb=lut_bb, lut_bf=lut_bf, lut_e=lut_e, lut_n=lut_n,
                          scen={s: qhat(cal[s][sc], args.alpha) for s in SCEN})
        thr_c = {s: thr[cal[s]["i"]] for s in SCEN}
        ev = {s: cal[s]["y"] >= thr_c[s] for s in SCEN}
        pool_h = np.concatenate([(thr_c[s] - cal[s]["hi"])[ev[s]] for s in SCEN])
        pool_p = np.concatenate([(thr_c[s] - cal[s]["pt"])[ev[s]] for s in SCEN])
        pool_b = np.concatenate([cal[s]["b"][ev[s]] for s in SCEN])
        lam = {}
        for aw in ALPHA_W:                      # warn if forecast + margin >= flood threshold; margin = conformal quantile
            g_h, g_p = qhat(pool_h, aw), qhat(pool_p, aw)
            lam[aw] = dict(
                h=np.array([qhat(pool_h[pool_b == b], aw) if (pool_b == b).sum() >= 100 else g_h for b in range(8)]),
                p=np.array([qhat(pool_p[pool_b == b], aw) if (pool_b == b).sum() >= 100 else g_p for b in range(8)]),
                hc=qhat((thr_c["none"] - cal["none"]["hi"])[ev["none"]], aw),
                pc=qhat((thr_c["none"] - cal["none"]["pt"])[ev["none"]], aw))
        for s in SCEN:
            Xt, yt, bt, bi, tt = get("te", s, test_src)
            pt, lo, hi = predict(ms, Xt)
            hf, ph = yt >= thr[bi], pt >= thr[bi]
            for aw in ALPHA_W:
                kinds = {"U_complete": hi + lam[aw]["hc"] >= thr[bi], "U_bins": hi + lam[aw]["h"][bt] >= thr[bi],
                         "P_complete": pt + lam[aw]["pc"] >= thr[bi], "P_bins": pt + lam[aw]["p"][bt] >= thr[bi]}
                for kind, wn in kinds.items():
                    for sname in ["all"] + list(np.unique(src[bi])):
                        m = np.ones(len(yt), bool) if sname == "all" else (src[bi] == sname)
                        warn_rows.append(dict(seed=seed, lead=lead, model=mname, source=sname, scenario=s, alpha_w=aw, kind=kind,
                                              recall=wn[m & hf].mean() if (m & hf).any() else np.nan,
                                              false_alarm=wn[m & ~hf].mean() if (m & ~hf).any() else np.nan,
                                              n_hi=int((m & hf).sum()), n_lo=int((m & ~hf).sum())))
            at, kt = codes(Xt)
            key = at * 10 + kt
            if args.row_level and s in ROW_SCEN:
                rowbuf["y"], rowbuf["bi"], rowbuf["dt"] = yt.astype(np.float32), bi.astype(np.int16), te[tt].astype(np.int16)
                for nm, arr in (("pt", pt), ("lo", lo), ("hi", hi), ("lamU", lam[0.1]["h"][bt]), ("lamP", lam[0.1]["p"][bt])):
                    rowbuf[f"{mname}|{s}|{nm}"] = np.asarray(arr).astype(np.float32)
                rowbuf[f"{mname}|{s}|lamU_c"] = np.float32([lam[0.1]["hc"]])
                rowbuf[f"{mname}|{s}|lamP_c"] = np.float32([lam[0.1]["pc"]])
            for sc, (c_lo, c_hi) in (("SC", (pt, pt)), ("CQR", (lo, hi))):
                k = qh[sc.lower()]
                for meth, w in (("complete", np.full(len(yt), k["complete"])), ("pooled", np.full(len(yt), k["pooled"])),
                                ("mondrian", k["lut"][bt]), ("mondrian_region", k["lut_br"][bt, rix[bi]]),
                                ("exactmask", k["lut_e"][key]),
                                ("nested", k["lut_n"][key]), ("scenario", np.full(len(yt), k["scen"][s])),
                                ("mondrian_basin", k["lut_bb"][bt, bi] if k["lut_bb"] is not None else None),
                                ("mondrian_flow", k["lut_bf"][bt, fcls(pt, bi)] if k["lut_bf"] is not None else None)):
                    if (sc == "CQR" and meth == "pooled") or w is None:
                        continue
                    L, U = c_lo - w, c_hi + w
                    cov = (yt >= L) & (yt <= U)
                    if args.row_level and sc == "CQR" and s in ROW_SCEN and meth in ROW_METHODS:
                        rowbuf[f"{mname}|{s}|w_{meth}"] = w[:1].astype(np.float32) if meth == "complete" else w.astype(np.float32)
                    if args.per_basin and sc == "CQR" and meth in PB_METHODS:
                        nb = len(pick)
                        cnt = lambda wt=None: np.bincount(bi, weights=wt, minlength=nb)
                        basin_rows.append(pd.DataFrame(dict(
                            seed=seed, lead=lead, model=mname, scenario=s, method=f"CQR_{meth}", gauge_id=gids,
                            source=src, n=cnt(), covered=cnt(cov), n_ph=cnt(ph.astype(float)),
                            covered_ph=cnt((cov & ph).astype(float)), width_sum=cnt(U - L))))
                    for sname in ["all"] + list(np.unique(src[bi])):
                        m = np.ones(len(yt), bool) if sname == "all" else (src[bi] == sname)
                        rows.append(dict(seed=seed, lead=lead, model=mname, source=sname, scenario=s,
                                         method=f"{sc}_{meth}", n=int(m.sum()), coverage=cov[m].mean(),
                                         coverage_highflow=cov[m & hf].mean() if (m & hf).any() else np.nan,
                                         coverage_predhigh=cov[m & ph].mean() if (m & ph).any() else np.nan,
                                         width_log=(U - L)[m].mean()))
    if args.row_level:
        np.savez_compressed(Path(args.proc) / f"trackA_rows_{args.tag}_lead{lead}.npz", thr=thr, gauge_ids=gids,
                            source=src, dates=dates, **rowbuf)
    return rows, (pd.concat(basin_rows) if basin_rows else None), warn_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--leads", type=int, nargs="+", default=[1])
    ap.add_argument("--n-per-source", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--train-stride", type=int, default=2, help="use every k-th day for training")
    ap.add_argument("--train-source", nargs="+", default=None)
    ap.add_argument("--cal-source", nargs="+", default=None)
    ap.add_argument("--test-source", nargs="+", default=None)
    ap.add_argument("--per-basin", action="store_true", help="also write per-basin coverage counts")
    ap.add_argument("--row-level", action="store_true", help="save row-level intervals for flood-day tests")
    ap.add_argument("--holdout", type=float, default=0.0, help="share of basins kept unseen (test only)")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--proc", default=str(PROC))
    args = ap.parse_args()
    proc = Path(args.proc)
    z = np.load(proc / "trackA_series.npz")
    st = pd.read_csv(proc / "trackA_static.csv")
    st["area"] = np.log(st["area"])
    out, t0 = proc / f"trackA_runs_{args.tag}.csv", time.time()
    rows, basin, warns = [], [], []
    for seed in args.seeds:
        for lead in args.leads:
            r, b, wr = run(args, seed, lead, z, st)
            rows += r
            warns += wr
            pd.DataFrame(rows).to_csv(out, index=False)
            pd.DataFrame(warns).to_csv(proc / f"trackA_warnings_{args.tag}.csv", index=False)
            if b is not None:
                basin.append(b)
                pd.concat(basin).to_csv(proc / f"trackA_basinlevel_{args.tag}.csv", index=False)
            print(f"seed {seed} lead {lead} done | {(time.time() - t0) / 60:.1f} min", flush=True)
    res = pd.DataFrame(rows)
    res = res[res.source == "all"]
    pd.set_option("display.width", 220)
    show = ["CQR_complete", "CQR_mondrian", "CQR_mondrian_basin", "CQR_mondrian_flow", "CQR_scenario"]
    for (lead, model), g in res.groupby(["lead", "model"]):
        g = g[g.method.isin(show)]
        for key in ("coverage", "width_log"):
            m = g.pivot_table(index="scenario", columns="method", values=key, aggfunc="mean").reindex(list(SCEN))[show]
            tab = m.round(3).astype(str)
            if len(args.seeds) > 1 and key == "coverage":
                sd = g.pivot_table(index="scenario", columns="method", values=key, aggfunc="std").reindex(list(SCEN))[show]
                tab = tab + "(" + (sd * 1000).round(0).astype("Int64").astype(str) + ")"
            print(f"\n[lead {lead}] [{model}] {key} (target {1 - args.alpha:.2f})\n{tab.to_string()}")
    w = pd.DataFrame(warns)
    w = w[(w.source == "all") & (w.alpha_w == 0.1)]
    for (lead, model), g in w.groupby(["lead", "model"]):
        for key in ("recall", "false_alarm"):
            t = g.pivot_table(index="scenario", columns="kind", values=key, aggfunc="mean").reindex(list(SCEN))
            print(f"\n[lead {lead}] [{model}] high-flow warnings {key} (target recall 0.90)\n{t.round(3).to_string()}")
    print(f"\nsaved {out} | total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
