"""Cluster bootstrap over basins and per-basin coverage spread."""
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path(os.environ.get("FLOOD_DATA_DIR", "data")) / "processed"
ORDER = ["none", "q_drop30", "q_drop50", "q_delay3", "q_delay7", "f_delay1", "f_delay3", "q_blocks", "combined"]
SHOW = ["CQR_complete", "CQR_mondrian", "CQR_mondrian_basin", "CQR_mondrian_flow"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(PROC / "trackA_basinlevel_bs.csv"))
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-n", type=int, default=100, help="drop basins with fewer test rows")
    ap.add_argument("--out", default=str(PROC / "trackA_bootstrap_summary.csv"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    d = pd.read_csv(args.file)
    d = d.groupby(["lead", "model", "scenario", "method", "gauge_id", "source"], as_index=False)[
        ["n", "covered", "n_ph", "covered_ph", "width_sum"]].sum()
    keep = d.groupby("gauge_id").n.min()
    ids = np.array(sorted(keep[keep >= args.min_n].index))
    d = d[d.gauge_id.isin(ids)]
    region = d.drop_duplicates("gauge_id").set_index("gauge_id").source.reindex(ids).to_numpy()
    B = len(ids)
    idx = rng.integers(0, B, size=(args.reps, B))             # same resamples for every group (paired)
    print(f"basins used: {B} | regions: {dict(zip(*np.unique(region, return_counts=True)))} | resamples: {args.reps}")

    out = []
    for (lead, model, method, scen), g in d.groupby(["lead", "model", "method", "scenario"]):
        g = g.set_index("gauge_id").reindex(ids)
        n, c, nph, cph, w = (g[k].to_numpy(float) for k in ("n", "covered", "n_ph", "covered_ph", "width_sum"))
        cov_b = c / n
        rec = dict(lead=lead, model=model, method=method, scenario=scen, coverage=c.sum() / n.sum(),
                   predhigh=cph.sum() / max(nph.sum(), 1), width=w.sum() / n.sum())
        for name, num, den in (("coverage", c, n), ("predhigh", cph, nph), ("width", w, n)):
            boot = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
            rec[f"{name}_lo"], rec[f"{name}_hi"] = np.percentile(boot, [2.5, 97.5])
        rec.update({f"basin_p{q}": np.percentile(cov_b, q) for q in (5, 25, 50, 75, 95)})
        rec["share_within_5pts"] = float((np.abs(cov_b - 0.9) <= 0.05).mean())
        for r in np.unique(region):
            m = region == r
            rec[f"coverage_{r}"] = c[m].sum() / n[m].sum()
        out.append(rec)
    res = pd.DataFrame(out)
    res.to_csv(args.out, index=False)
    print(f"saved {args.out}")
    pd.set_option("display.width", 230)
    for (lead, model), g in res.groupby(["lead", "model"]):
        for lab, cols in (("coverage [95% CI half-width]", ("coverage", "coverage_lo", "coverage_hi")),):
            t = pd.DataFrame({m: g[g.method == m].set_index("scenario").apply(
                lambda r: f"{r.coverage:.3f}±{(r.coverage_hi - r.coverage_lo) / 2:.3f}", axis=1) for m in SHOW}).reindex(ORDER)
            print(f"\n[lead {lead}] [{model}] {lab}\n{t.to_string()}")
        t = pd.DataFrame({f"{m} p5-p95": g[g.method == m].set_index("scenario").apply(
            lambda r: f"{r.basin_p5:.2f}-{r.basin_p95:.2f} ({r.share_within_5pts:.0%})", axis=1) for m in SHOW}).reindex(ORDER)
        print(f"\n[lead {lead}] [{model}] per-basin coverage 5th-95th percentile (share of basins within 0.85-0.95)\n{t.to_string()}")


if __name__ == "__main__":
    main()
