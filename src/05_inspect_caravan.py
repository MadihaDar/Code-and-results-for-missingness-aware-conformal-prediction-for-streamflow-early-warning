"""
05_inspect_caravan.py
Looks inside the extracted Caravan netCDF files and summarises every basin's streamflow record.
"""
import os
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(os.environ.get("FLOOD_DATA_DIR", "data"))
NC_ROOT = BASE / "raw" / "Caravan" / "extracted" / "Caravan-nc" / "timeseries" / "netcdf"


def longest_nan_run(a):
    """Length (days) of the longest run of NaN in a 1-D array."""
    isn = np.isnan(a).astype(np.int8)
    if not isn.any():
        return 0
    edges = np.diff(np.concatenate(([0], isn, [0])))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return int((ends - starts).max())


def first_match(names, *candidates):
    for c in candidates:
        for n in names:
            if n == c:
                return n
    for c in candidates:
        for n in names:
            if n.startswith(c):
                return n
    return None


def describe_example(xr, path):
    ds = xr.open_dataset(path)
    print(f"\n--- example file: {path.name}")
    print("dims:", dict(ds.sizes))
    print("coords:", list(ds.coords))
    print("variables (name | units):")
    for v in ds.data_vars:
        print(f"   {v:55s} {ds[v].attrs.get('unit', ds[v].attrs.get('units', ''))}")
    print("global attrs:", {k: str(v)[:60] for k, v in list(ds.attrs.items())[:8]})
    ds.close()


def summarise_file(xr, path):
    ds = xr.open_dataset(path)
    names = list(ds.data_vars)
    qvar = first_match(names, "streamflow")
    pvar = first_match(names, "total_precipitation_sum", "total_precipitation")
    if qvar is None:
        raise KeyError("no streamflow variable; variables: " + ", ".join(names[:10]))
    q = ds[qvar]
    tdim = q.dims[0]
    dates = pd.to_datetime(ds[tdim].values)
    qv = q.values.astype(float)
    ok = ~np.isnan(qv)
    row = {"gauge_id": path.stem, "n_days": len(dates), "q_var": qvar, "p_var": pvar,
           "forcing_start": dates.min().date(), "forcing_end": dates.max().date(), "n_valid_q": int(ok.sum())}
    if ok.any():
        i0, i1 = np.where(ok)[0][[0, -1]]
        span = qv[i0:i1 + 1]
        row.update(q_start=dates[i0].date(), q_end=dates[i1].date(), q_span_days=int(i1 - i0 + 1),
                   q_missing_in_span=float(np.isnan(span).mean()), q_longest_gap=longest_nan_run(span),
                   q_mean=float(np.nanmean(qv)), q_p99=float(np.nanpercentile(qv, 99)), q_max=float(np.nanmax(qv)),
                   q_zero_share=float((qv[ok] == 0).mean()))
    if pvar is not None:
        pv = ds[pvar].values.astype(float)
        row["p_missing"] = float(np.isnan(pv).mean())
    ds.close()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["lamah", "camelsgb"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N basins per source")
    ap.add_argument("--root", default=str(NC_ROOT))
    ap.add_argument("--out", default=str(BASE / "processed" / "caravan_basin_summary.csv"))
    args = ap.parse_args()
    try:
        import xarray as xr
    except ImportError:
        raise SystemExit("xarray is missing:  pip install xarray netCDF4")

    rows, errors, t0 = [], [], time.time()
    for src in args.sources:
        files = sorted((Path(args.root) / src).glob("*.nc"))
        if args.limit:
            files = files[:args.limit]
        print(f"\n=== {src}: {len(files)} files")
        if not files:
            continue
        describe_example(xr, files[0])
        for i, f in enumerate(files, 1):
            try:
                r = summarise_file(xr, f)
                r["source"] = src
                rows.append(r)
            except Exception as exc:
                errors.append((f.name, repr(exc)[:150]))
            if i % 200 == 0:
                print(f"  {i}/{len(files)} | {(time.time() - t0) / 60:.1f} min", flush=True)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nsaved {args.out} ({len(df)} basins) | errors: {len(errors)}")
    for e in errors[:5]:
        print("  ", e)
    if df.empty:
        return
    df["valid_years"] = df["n_valid_q"] / 365.25
    g = df.groupby("source")
    out = pd.DataFrame({
        "basins": g.size(),
        "forcing_start": g["forcing_start"].min(), "forcing_end": g["forcing_end"].max(),
        "q_start_median": g["q_start"].apply(lambda s: pd.to_datetime(s).median().date()),
        "q_end_median": g["q_end"].apply(lambda s: pd.to_datetime(s).median().date()),
        "valid_yrs_median": g["valid_years"].median().round(1),
        "basins_>=20yr": g["valid_years"].apply(lambda s: int((s >= 20).sum())),
        "basins_>=30yr": g["valid_years"].apply(lambda s: int((s >= 30).sum())),
        "missing_in_span_median_%": (g["q_missing_in_span"].median() * 100).round(2),
        "longest_gap_median_d": g["q_longest_gap"].median(),
        "p_missing_max_%": (g["p_missing"].max() * 100).round(2),
    })
    print("\nQuality summary:\n" + out.T.to_string())


if __name__ == "__main__":
    main()
