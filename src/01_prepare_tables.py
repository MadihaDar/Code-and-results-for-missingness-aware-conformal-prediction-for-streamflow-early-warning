"""
01_prepare_tables.py
Builds clean, analysis-ready tables from the raw downloads.

"""
import os
import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(os.environ.get("FLOOD_DATA_DIR", "data")) / "raw"
OUT = RAW.parent / "processed"


def find(pattern):
    hits = sorted(RAW.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"No file matching {pattern!r} under {RAW}")
    return hits[0]


def save(df, name):
    path = OUT / name
    df.to_csv(path, index=False)
    print(f"  saved {name}: {len(df)} rows x {df.shape[1]} cols")


# ----------------------------------------------------------------------------
# 1. GFD events (exported from Earth Engine)
# ----------------------------------------------------------------------------
def bbox_from_footprint(s):
    """system:footprint is a coarse bounding ring, not the flood extent."""
    try:
        coords = ast.literal_eval(s)["coordinates"]
        if coords and isinstance(coords[0][0], (list, tuple)):
            coords = coords[0]
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        return min(xs), min(ys), max(xs), max(ys)
    except Exception:
        return (np.nan,) * 4


def prepare_gfd():
    print("GFD")
    g = pd.read_csv(find("gfd_events.csv"))
    bb = pd.DataFrame(g["system:footprint"].apply(bbox_from_footprint).tolist(),
                      columns=["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"],
                      index=g.index)
    g = pd.concat([g, bb], axis=1)
    g["start_date"] = pd.to_datetime(g["system:time_start"], unit="ms").dt.normalize()
    g["end_date"] = pd.to_datetime(g["system:time_end"], unit="ms").dt.normalize()
    out = g.rename(columns={"id": "gfd_id"})[[
        "gfd_id", "start_date", "end_date", "dfo_centroid_x", "dfo_centroid_y",
        "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax",
        "dfo_main_cause", "dfo_severity", "dfo_dead", "dfo_displaced",
        "dfo_country", "gfd_country_code", "countries", "system:index"]]
    out = out.rename(columns={"system:index": "gee_index"})
    save(out, "gfd_events_clean.csv")
    return out


# ----------------------------------------------------------------------------
# 2. FLODIS -> joined to GFD for dates
# ----------------------------------------------------------------------------
def parse_ids(x):
    """GFD_matches_nr is either '1995' or a list-like string '[2718, 2731]'."""
    if pd.isna(x):
        return []
    s = str(x).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [int(i) for i in v]
        return [int(float(v))]
    except Exception:
        return [int(n) for n in re.findall(r"\d+", s)]


def prepare_flodis(gfd):
    print("FLODIS")
    d = pd.read_csv(find("FLODIS_displacement.csv"), low_memory=False)
    d["table"] = "displacement"
    m = pd.read_csv(find("FLODIS_mortality_damage.csv"), low_memory=False)
    m["table"] = "mortality_damage"
    # Identifiers + impact targets only. The *_affected_* and density columns are
    # derived from the observed flood extent (post-event) and are left in the raw
    # files on purpose: do not use them as predictors in a forecasting setting.
    cols = ["table", "ISO3", "year", "displacements", "total_deaths", "no_injured",
            "no_homeles", "no_affected_EMDAT", "total_affected_EMDAT",
            "total_damages_(000_USD)", "GFD_matches_nr", "matching_type"]
    f = pd.concat([d.reindex(columns=cols), m.reindex(columns=cols)], ignore_index=True)
    f.insert(0, "flodis_row", f["table"] + "_" + f.groupby("table").cumcount().astype(str))
    f["gfd_ids"] = f["GFD_matches_nr"].apply(parse_ids)

    long = f[["flodis_row", "gfd_ids"]].explode("gfd_ids").rename(columns={"gfd_ids": "gfd_id"})
    long = long.dropna(subset=["gfd_id"])
    long["gfd_id"] = long["gfd_id"].astype(int)
    long = long.merge(gfd[["gfd_id", "start_date", "end_date", "dfo_centroid_x", "dfo_centroid_y"]],
                      on="gfd_id", how="left")
    agg = long.groupby("flodis_row").agg(
        n_ids=("gfd_id", "size"),
        n_found_in_gfd=("start_date", lambda s: s.notna().sum()),
        event_start=("start_date", "min"),
        event_end=("end_date", "max"),
        centroid_x=("dfo_centroid_x", "mean"),
        centroid_y=("dfo_centroid_y", "mean"),
    ).reset_index()
    out = f.drop(columns=["gfd_ids"]).merge(agg, on="flodis_row", how="left")

    # ---- join diagnostics ----
    ok = out["n_ids"].gt(0) & out["n_ids"].eq(out["n_found_in_gfd"])
    print(f"  FLODIS rows: {len(out)} | all GFD ids found in your GFD export: {int(ok.sum())}")
    print(f"  rows with at least one id missing from GFD export: {int((~ok).sum())}")
    yr = pd.to_datetime(out["event_start"]).dt.year
    both = out["year"].notna() & yr.notna()
    print(f"  FLODIS year == GFD start year (sanity check): "
          f"{int((out.loc[both, 'year'] == yr[both]).sum())} of {int(both.sum())}")
    save(out, "flodis_events.csv")
    return out


# ----------------------------------------------------------------------------
# 3. HANZE
# ----------------------------------------------------------------------------
def prepare_hanze():
    print("HANZE")
    h = pd.read_csv(find("HANZE_events_v3_0_2b.csv"), low_memory=False)
    h = h.rename(columns={"id": "hanze_id", "ppl_affected": "people_affected",
                          "casualty_count": "casualties",
                          "economic_loss_value_eur": "loss_eur_2025"})
    h["start_date"] = pd.to_datetime(h["start_date"])
    h["end_date"] = pd.to_datetime(h["end_date"])
    h["duration_days"] = (h["end_date"] - h["start_date"]).dt.days
    units = (h.set_index("hanze_id")["admin_unit_code"].str.split(";").explode()
             .str.strip().reset_index().rename(columns={"admin_unit_code": "nuts3"}))
    n_units = units.groupby("hanze_id").size().rename("n_nuts3")
    h = h.merge(n_units, on="hanze_id", how="left")
    h["year_ge_1950"] = h["start_date"].dt.year >= 1950
    h["year_ge_1981"] = h["start_date"].dt.year >= 1981
    keep = ["hanze_id", "country_code", "country_name", "start_date", "end_date",
            "duration_days", "event_type", "cause", "n_nuts3", "area_affected",
            "casualties", "people_affected", "loss_eur_2025", "year_ge_1950", "year_ge_1981"]
    save(h[keep], "hanze_events.csv")
    save(units, "hanze_event_units.csv")
    s = h[h["year_ge_1981"]]
    print(f"  events since 1981: {len(s)} | with people_affected: {int(s['people_affected'].notna().sum())}")
    return h


# ----------------------------------------------------------------------------
# 4. Pakistan (bilingual Chinese/English Excel tables)
# ----------------------------------------------------------------------------
def english(s):
    """Return the English text inside (...) or full-width brackets."""
    if pd.isna(s):
        return np.nan
    txt = str(s)
    parts = re.findall(r"[（(]([^（）()]*[A-Za-z][^（）()]*)[）)]", txt)
    if parts:
        return "; ".join(p.strip().replace("\n", " ") for p in parts)
    return txt.strip().replace("\n", " ")


def prepare_pakistan():
    print("Pakistan")
    basic = pd.read_excel(find("*Basic Information*.xlsx"))
    basic.columns = ["event_id", "start_date", "end_date", "season", "flood_type",
                     "origin", "cause", "climate_background", "abnormal_weather"]
    for c in ["season", "flood_type", "origin", "cause", "climate_background", "abnormal_weather"]:
        basic[c] = basic[c].apply(english)
    impact = pd.read_excel(find("*Disaster Impact*.xlsx"))
    impact.columns = ["event_id", "deaths", "affected_population", "affected_villages",
                      "houses_text", "livestock_dead", "cropland_km2", "loss_kusd",
                      "infrastructure_text", "cpi_inflation_pct"]
    for c in ["deaths", "affected_population", "livestock_dead", "cropland_km2", "loss_kusd"]:
        before = impact[c].notna().sum()
        # some cells are text such as "2,496 " (comma / trailing space)
        impact[c] = pd.to_numeric(
            impact[c].astype(str).str.replace(",", "").str.strip(), errors="coerce")
        lost = before - impact[c].notna().sum()
        if lost:
            print(f"  note: {lost} non-numeric values in {c} became NaN")
    for c in ["houses_text", "infrastructure_text", "affected_villages"]:
        impact[c] = impact[c].apply(english)
    space = pd.read_excel(find("*Space and Hydrological*.xlsx"))
    space.columns = ["event_id", "flood_extent_km2", "provinces", "districts", "rivers",
                     "max_water_level_m", "hydro_infrastructure", "max_warning_level"]
    for c in ["provinces", "districts", "rivers"]:
        space[c] = space[c].apply(english)
    space["flood_extent_km2"] = pd.to_numeric(space["flood_extent_km2"], errors="coerce")
    out = (basic.merge(impact, on="event_id", how="left")
                .merge(space[["event_id", "flood_extent_km2", "provinces", "districts", "rivers"]],
                       on="event_id", how="left"))
    out["start_date"] = pd.to_datetime(out["start_date"])
    out["end_date"] = pd.to_datetime(out["end_date"])
    save(out, "pakistan_events.csv")
    print(f"  events: {len(out)} | with affected_population: {int(out['affected_population'].notna().sum())}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="?", help="raw data folder (default $FLOOD_DATA_DIR/raw)")
    ap.add_argument("--only", nargs="+", default=["hanze"], choices=["hanze", "gfd", "flodis", "pakistan"],
                    help="datasets to prepare; only hanze is needed for the paper")
    a = ap.parse_args()
    if a.raw:
        RAW = Path(a.raw)
        OUT = RAW.parent / "processed"
    OUT.mkdir(exist_ok=True)
    print(f"RAW dir: {RAW}\nOUT dir: {OUT}\n")
    gfd = prepare_gfd() if ("gfd" in a.only or "flodis" in a.only) else None
    if "flodis" in a.only:
        prepare_flodis(gfd)
    if "hanze" in a.only:
        prepare_hanze()
    if "pakistan" in a.only:
        prepare_pakistan()
