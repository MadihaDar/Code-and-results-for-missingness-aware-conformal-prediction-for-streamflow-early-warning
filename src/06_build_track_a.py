"""
06_build_track_a.py
Selects Caravan basins and stores their raw daily series in one compact file for Track A
(missingness-aware conformal streamflow forecasting).

Why raw series and not ready-made features: when we later simulate missing data we must mask the
RAW inputs first and only then compute lags / rolling sums, otherwise the masking leaks through the features.
"""
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(os.environ.get("FLOOD_DATA_DIR", "data"))
CARAVAN = BASE / "raw" / "Caravan" / "extracted" / "Caravan-nc"
START, END = "1981-01-01", "2017-12-31"
VARS = {                                   # short name -> netCDF variable (names confirmed from the file listing)
    "q": "streamflow",                     # mm/d
    "p": "total_precipitation_sum",        # mm/d
    "t": "temperature_2m_mean",            # deg C
    "swe": "snow_depth_water_equivalent_mean",
    "sm1": "volumetric_soil_water_layer_1_mean",
    "sm2": "volumetric_soil_water_layer_2_mean",
    "sm3": "volumetric_soil_water_layer_3_mean",
    "pet": "potential_evaporation_sum_ERA5_LAND",
}
STATIC_COLS = ["frac_snow", "aridity_ERA5_LAND", "p_mean", "seasonality_ERA5_LAND",
               "high_prec_freq", "moisture_index_ERA5_LAND"]


def load_attributes(sources, caravan):
    frames = []
    for src in sources:
        a = caravan / "attributes" / src
        other = pd.read_csv(a / f"attributes_other_{src}.csv")
        clim = pd.read_csv(a / f"attributes_caravan_{src}.csv")
        keep = ["gauge_id"] + [c for c in STATIC_COLS if c in clim.columns]
        frames.append(other.merge(clim[keep], on="gauge_id", how="left").assign(source=src))
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["lamah", "camelsgb"])
    ap.add_argument("--n-per-source", type=int, default=150)
    ap.add_argument("--area-min", type=float, default=100.0, help="km2")
    ap.add_argument("--area-max", type=float, default=2000.0, help="km2")
    ap.add_argument("--min-valid-years", type=float, default=20.0, help="valid streamflow years inside 1981-2017")
    ap.add_argument("--max-missing", type=float, default=0.05, help="max share of missing days inside the record span")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base", default=str(BASE))
    args = ap.parse_args()
    try:
        import xarray as xr
    except ImportError:
        raise SystemExit("xarray is missing:  pip install xarray netCDF4")

    base = Path(args.base)
    caravan = base / "raw" / "Caravan" / "extracted" / "Caravan-nc"
    proc = base / "processed"
    summ = pd.read_csv(proc / "caravan_basin_summary.csv")
    attrs = load_attributes(args.sources, caravan)
    df = summ.merge(attrs, on=["gauge_id", "source"], how="inner")
    cand = df[(df["area"].between(args.area_min, args.area_max))
              & (df["n_valid_q"] / 365.25 >= args.min_valid_years)
              & (df["q_missing_in_span"] <= args.max_missing)]
    print("candidates after area / record-length / missing filters:")
    print(cand.groupby("source").size().to_string())

    rng = np.random.default_rng(args.seed)
    picks = []
    for src, g in cand.groupby("source"):
        n = min(args.n_per_source, len(g))
        picks.append(g.iloc[np.sort(rng.choice(len(g), size=n, replace=False))])
    sel = pd.concat(picks, ignore_index=True)

    dates = pd.date_range(START, END, freq="D")
    T, V = len(dates), len(VARS)
    data = np.full((len(sel), T, V), np.nan, dtype=np.float32)
    keep = np.ones(len(sel), dtype=bool)
    min_days = int(args.min_valid_years * 365.25)
    for i, row in enumerate(sel.itertuples(index=False)):
        path = caravan / "timeseries" / "netcdf" / row.source / f"{row.gauge_id}.nc"
        ds = xr.open_dataset(path)
        tdim = ds[VARS["q"]].dims[0]
        pos = dates.get_indexer(pd.DatetimeIndex(ds[tdim].values))
        inside = pos >= 0
        for j, name in enumerate(VARS.values()):
            data[i, pos[inside], j] = ds[name].values.astype(np.float32)[inside]
        ds.close()
        if np.isfinite(data[i, :, 0]).sum() < min_days:          # not enough streamflow inside the window
            keep[i] = False
        if (i + 1) % 50 == 0:
            print(f"  read {i + 1}/{len(sel)} basins", flush=True)

    print(f"dropped {int((~keep).sum())} basins with < {args.min_valid_years:g} valid years inside {START[:4]}-{END[:4]}")
    sel, data = sel[keep].reset_index(drop=True), data[keep]
    static_cols = ["gauge_id", "source", "gauge_name", "country", "gauge_lat", "gauge_lon", "area"] + \
                  [c for c in STATIC_COLS if c in sel.columns]
    st = sel[static_cols].copy()
    st["valid_days"] = np.isfinite(data[:, :, 0]).sum(axis=1)
    st["q_missing_share"] = 1 - st["valid_days"] / T
    st.to_csv(proc / "trackA_static.csv", index=False)
    # plain fixed-width unicode arrays (pandas 3 string columns would otherwise be saved as object arrays)
    np.savez_compressed(proc / "trackA_series.npz",
                        gauge_ids=np.array(sel["gauge_id"].tolist(), dtype="U"),
                        source=np.array(sel["source"].tolist(), dtype="U"),
                        dates=dates.values.astype("datetime64[D]"),
                        var_names=np.array(list(VARS), dtype="U"), data=data)
    print(f"\nsaved {proc / 'trackA_series.npz'}  shape {data.shape} = basins x days x variables "
          f"({data.nbytes / 1e6:.0f} MB in memory)")
    print(st.groupby("source").agg(basins=("gauge_id", "size"), area_median=("area", "median"),
                                   frac_snow_median=("frac_snow", "median"),
                                   valid_years_median=("valid_days", lambda s: round(s.median() / 365.25, 1))).to_string())


if __name__ == "__main__":
    main()
