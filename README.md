# Missingness-aware conformal prediction for streamflow early warning: code and results

## What is here, and what is not

| Folder | Content | Licence |
|---|---|---|
| `src/` | Scripts that produce every number and figure in the paper | MIT (see `LICENSE`) |
| `results/` | Final result tables (coverage, widths, bootstrap intervals, warnings, flood-day tests) | CC BY 4.0 (see `DATA_LICENCES.md`) |
| `results/earlier_runs/` | Earlier runs kept for transparency (older code versions) | CC BY 4.0 |
| `data/basin_ids.csv` | The 300 basins used (gauge ID and source dataset) | CC BY 4.0 |
| `exploratory/` | Scripts for an exploratory HANZE impact analysis and a superseded pilot; **not used for paper results** | MIT |

**Not redistributed:** the Caravan time series, the arrays built from them, row-level predictions and any
Global Flood Database product. They are derived from datasets with their own licences (see `DATA_LICENCES.md`);
`src/` rebuilds everything from the original sources.

## Scenario codes and method names

Scenarios: `none`; `q_drop30` / `q_drop50` (D30 / D50: random gauge dropout); `q_delay3` / `q_delay7` (L3 / L7: gauge data
3 / 7 days late); `f_delay1` / `f_delay3` (F1 / F3: forcing 1 / 3 days late); `q_blocks` (OUT: gauge outage blocks);
`combined` (ALL: 30% dropout, 3 days late gauge, 1 day late forcing).

Models: `trained_complete` (trained on complete data), `trained_mixed` (trained with masked inputs).

Methods (all 90% intervals from conformalised quantile regression, CQR, unless prefixed `SC_` = split conformal):
`complete` calibrated on complete data; `mondrian` per missingness bin (8 bins: staleness bucket x forcing delay);
`mondrian_region`, `mondrian_basin`, `mondrian_flow` bins x region / basin / forecast level; `exactmask`, `nested`, `scenario`
mask-matched baselines in the spirit of Zaffran et al. (2023).

Metrics: `coverage` (share of observations inside the interval); `coverage_predhigh` (coverage on days when the point forecast is
above the basin's training-period 95th percentile); `coverage_highflow` (coverage when the *observed* flow is above it; selection on
the outcome, so expect values below nominal); `width_log` (mean interval width in log(q + 0.01) units, q in mm/d).

## Setup

```
pip install -r requirements.txt
python check_environment.py            # prints versions: paste them into the paper / record them here
export FLOOD_DATA_DIR=/path/to/data    # default: ./data   (Windows: set FLOOD_DATA_DIR=C:\path\to\data)
export EE_PROJECT=<your Earth Engine project>   # only for exploratory/02_extract_rainfall_hanze.py
```

Hardware and software used for the reported runs: [TODO: CPU, RAM, OS, Python and package versions from `check_environment.py`].

## Data to download (not included)

1. **Caravan** (netCDF release; record the version from the `VERSION` file in the archive: [TODO]; DOI: [TODO from the Caravan record]).
   Save the archive as `$FLOOD_DATA_DIR/raw/Caravan/Caravan-nc-download.tar.gz`. We use the LamaH (Central Europe) and CAMELS-GB sources.
2. **HANZE v3.0.2 beta** (https://zenodo.org/records/23670886): `HANZE_events_v3_0_2b.csv` and `Regions_v2024_simplified.zip`
   into `$FLOOD_DATA_DIR/raw/HANZE/`. Newer versions may change event records, so keep the version you use.

## Reproduction (observed run times are from the reported runs; hardware: [TODO])

```
python src/03_caravan_manifest.py                       # one pass over the 25 GB archive, writes a manifest
python src/04_extract_caravan_sources.py                # extracts LamaH + CAMELS-GB only (about 7 GB; about 10 min)
python src/05_inspect_caravan.py                        # per-basin record summary (about 1 min)
python src/06_build_track_a.py                          # selects 150 basins per source (seed 42) -> trackA_series.npz
python src/01_prepare_tables.py                         # HANZE event tables

# main run: 300 basins, leads 1 and 3 (about 56 min)
python src/08_trackA_runs.py --n-per-source 150 --seeds 0 --leads 1 3 --per-basin --row-level --tag final2
# unseen basins: 30% of basins held out (about 16 min)
python src/08_trackA_runs.py --n-per-source 150 --seeds 0 --leads 1 --holdout 0.3 --per-basin --tag holdout
# regional transfer: LamaH -> CAMELS-GB, calibrated on the source or on the target region (about 5 min per seed)
python src/08_trackA_runs.py --seeds 0 1 2 --train-source lamah --test-source camelsgb --tag lamah_to_gb
python src/08_trackA_runs.py --seeds 0 1 2 --train-source lamah --cal-source camelsgb --test-source camelsgb --tag lamah_to_gb_cal

python src/09_bootstrap_basins.py --file $FLOOD_DATA_DIR/processed/trackA_basinlevel_final2.csv
python src/09_bootstrap_basins.py --file $FLOOD_DATA_DIR/processed/trackA_basinlevel_holdout.csv --min-n 50 \
       --out $FLOOD_DATA_DIR/processed/trackA_bootstrap_summary_holdout.csv
python src/10_hanze_flood_days.py --tag final2           # flood-day tests (needs the row-level files of the main run)
python src/11_make_figures.py                           # Fig1-Fig7 and FigS1 (PDF, EPS, PNG) in ../figures
```

Copying `results/*.csv` into `$FLOOD_DATA_DIR/processed/` lets `11_make_figures.py` regenerate all figures without re-running the experiments.

## Reproducibility notes

- Seeds: basin sampling uses seed 42 (`06`); masks, subsampling and the held-out basins use the `--seeds` value.
  Results for methods that exist in two runs were identical, so runs are deterministic for a given environment.
- Calibration/test split is temporal: train 1981-1999, calibrate 2000-2007, test 2008-2017 (CAMELS-GB records mostly end in 2015).
- Missingness is simulated by masking inputs; observed gauge and forcing data are essentially complete in Caravan.
- Flood-day tests use HANZE events in River, Flash and River/Coastal categories, matched to gauges by NUTS3 region and to days
  from one day before the start to seven days after it (capped at the event end). Only 34 events have data, so intervals are wide.

## Citation

See `CITATION.cff`. Please also cite the source datasets listed in `DATA_LICENCES.md`.
