"""Track A: conformal streamflow forecasts under simulated missing inputs."""
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
SCEN = {                                     # q_drop, q_delay (days), forcing_delay (days)
    "none": (0, 0, 0), "q_drop30": (.3, 0, 0), "q_drop50": (.5, 0, 0), "q_delay3": (0, 3, 0),
    "q_delay7": (0, 7, 0), "f_delay1": (0, 0, 1), "f_delay3": (0, 0, 3), "combined": (.3, 3, 1)}
FORCING = ("p", "t", "swe", "sm1", "sm2", "sm3", "pet")
STATIC = ["frac_snow", "aridity_ERA5_LAND", "p_mean", "seasonality_ERA5_LAND", "high_prec_freq",
          "moisture_index_ERA5_LAND", "area"]


def shift(a, k):                             # value at t-k (k<0: future)
    out = np.full_like(a, np.nan)
    if k == 0:
        return a.copy()
    if k > 0:
        out[k:] = a[:-k]
    else:
        out[:k] = a[-k:]
    return out


def roll(a, w, fn):
    return getattr(pd.DataFrame(a).rolling(w, min_periods=1), fn)().to_numpy()


def build(raw, dates, static, scen, rng):
    """Mask/delay raw inputs, then compute features. Returns X (T,B,F), age (T,B), forcing delay."""
    qd, kq, kf = SCEN[scen]
    q = raw["q"].copy()
    if qd:
        q[rng.random(q.shape) < qd] = np.nan
    q = shift(q, kq)
    f = {k: shift(raw[k], kf) for k in FORCING}
    T, B = q.shape
    lq = np.log(q + EPS)
    t_idx = np.arange(T)[:, None]
    last = np.maximum.accumulate(np.where(np.isfinite(lq), t_idx, -1), axis=0)
    age = np.where(last >= 0, t_idx - last + kq, 99).astype(np.float32)   # staleness incl. gauge delay
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
    X = np.stack([v.astype(np.float32) for v in feats.values()], axis=-1)
    return X, age, kf


def bin_id(age, kf):
    a = np.digitize(age, [1, 3, 7])            # 0:0, 1:1-2, 2:3-6, 3:7+
    return a * 2 + (1 if kf > 0 else 0)


def qhat(s, alpha):
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.sort(s)[min(k, n) - 1]


def period(dates, p):
    d = pd.DatetimeIndex(dates)
    return np.where((d >= p[0]) & (d <= p[1]))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-source", type=int, default=40)
    ap.add_argument("--lead", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proc", default=str(PROC))
    args = ap.parse_args()
    proc, rng = Path(args.proc), np.random.default_rng(args.seed)
    z = np.load(proc / "trackA_series.npz")
    st = pd.read_csv(proc / "trackA_static.csv")
    st["area"] = np.log(st["area"])
    pick = np.concatenate([np.sort(rng.choice(np.where(z["source"] == s)[0], min(args.n_per_source, (z["source"] == s).sum()),
                                              replace=False)) for s in np.unique(z["source"])])
    data, dates, src = z["data"][pick], z["dates"], z["source"][pick]
    st = st.iloc[pick].reset_index(drop=True)
    raw = {n: data[:, :, i].T for i, n in enumerate(z["var_names"])}          # (T,B)
    Y = shift(np.log(raw["q"] + EPS), -args.lead)                              # true future target
    tr, ca, te = (period(dates, p) for p in (TRAIN, CAL, TEST))
    thr = np.nanquantile(Y[tr], 0.95, axis=0)                                  # per-basin high-flow threshold
    print(f"basins {len(pick)} | lead {args.lead} d | alpha {args.alpha}", flush=True)

    t0, F = time.time(), {}
    for s in SCEN:
        X, age, kf = build(raw, dates, st, s, rng)
        F[s] = {}
        for name, ix in (("tr", tr), ("ca", ca), ("te", te)):
            y = Y[ix]
            ok = np.isfinite(y) & np.isfinite(raw["q"][ix])
            b_idx = np.broadcast_to(np.arange(len(pick))[None, :], y.shape)
            F[s][name] = (X[ix][ok], y[ok], bin_id(age[ix][ok], kf), b_idx[ok])
        del X, age
    print(f"features built in {time.time() - t0:.0f}s", flush=True)

    def stack(name, scens, frac=1.0):
        parts = []
        for s in scens:
            X, y, b, bi = F[s][name]
            keep = rng.random(len(y)) < frac
            parts.append((X[keep], y[keep], b[keep], bi[keep]))
        return [np.concatenate(p) for p in zip(*parts)]

    kw = dict(max_iter=300, learning_rate=0.08, l2_regularization=1.0, random_state=0)
    lo_q, hi_q = args.alpha / 2, 1 - args.alpha / 2
    models = {}
    for mname, (scens, frac) in {"trained_complete": (["none"], 1.0), "trained_mixed": (list(SCEN), 0.25)}.items():
        Xtr, ytr = stack("tr", scens, frac)[:2]
        models[mname] = [HGB(**kw).fit(Xtr, ytr), HGB(loss="quantile", quantile=lo_q, **kw).fit(Xtr, ytr),
                         HGB(loss="quantile", quantile=hi_q, **kw).fit(Xtr, ytr)]
        print(f"{mname} trained at {time.time() - t0:.0f}s", flush=True)

    def predict(ms, X):
        pt, lo, hi = (m.predict(X) for m in ms)
        return pt, np.minimum(lo, hi), np.maximum(lo, hi)

    rows = []
    for mname, ms in models.items():
        cal = {}
        for s in SCEN:
            Xc, yc, bc, _ = F[s]["ca"]
            pt, lo, hi = predict(ms, Xc)
            cal[s] = dict(sc=np.abs(yc - pt), cqr=np.maximum(lo - yc, yc - hi), b=bc)
        qh = {}
        for sc in ("sc", "cqr"):
            pool = np.concatenate([cal[s][sc] for s in SCEN])
            pb = np.concatenate([cal[s]["b"] for s in SCEN])
            qp = qhat(pool, args.alpha)
            qh[sc] = dict(complete=qhat(cal["none"][sc], args.alpha), pooled=qp,
                          bins={b: (qhat(pool[pb == b], args.alpha) if (pb == b).sum() >= 200 else qp)
                                for b in np.unique(pb)})
        for s in SCEN:
            Xt, yt, bt, bi = F[s]["te"]
            pt, lo, hi = predict(ms, Xt)
            hf, ph = yt >= thr[bi], pt >= thr[bi]      # high by outcome / by forecast
            for sc, (c_lo, c_hi) in (("SC", (pt, pt)), ("CQR", (lo, hi))):
                k = qh[sc.lower()]
                for meth, w in (("complete", np.full(len(yt), k["complete"])), ("pooled", np.full(len(yt), k["pooled"])),
                                ("mondrian", np.array([k["bins"].get(b, k["pooled"]) for b in bt]))):
                    if sc == "CQR" and meth == "pooled":
                        continue
                    L, U = c_lo - w, c_hi + w
                    cov = (yt >= L) & (yt <= U)
                    for sname in ["all"] + list(np.unique(src)):
                        m = np.ones(len(yt), bool) if sname == "all" else (src[bi] == sname)
                        rows.append(dict(model=mname, source=sname, scenario=s, method=f"{sc}_{meth}", n=int(m.sum()),
                                         coverage=cov[m].mean(), coverage_highflow=cov[m & hf].mean() if (m & hf).any() else np.nan,
                                         coverage_predhigh=cov[m & ph].mean() if (m & ph).any() else np.nan,
                                         width_log=(U - L)[m].mean()))
    res = pd.DataFrame(rows)
    res.to_csv(proc / f"trackA_results_lead{args.lead}.csv", index=False)
    pd.set_option("display.width", 200)
    for mname in models:
        for key in ("coverage", "coverage_predhigh", "width_log"):
            t = res[(res.model == mname) & (res.source == "all")].pivot(index="scenario", columns="method", values=key)
            print(f"\n[{mname}] {key} (target coverage {1 - args.alpha:.2f}, all basins)\n{t.reindex(list(SCEN)).round(3).to_string()}")
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> trackA_results_lead{args.lead}.csv")


if __name__ == "__main__":
    main()
