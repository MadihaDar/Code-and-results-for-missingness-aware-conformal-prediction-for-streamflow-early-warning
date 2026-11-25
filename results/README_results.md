# Result files

All files are produced by `src/` (see the top-level README). `source` is `all`, `lamah` or `camelsgb`.

| File | Produced by | Content |
|---|---|---|
| `trackA_runs_final2.csv` | `08_trackA_runs.py --tag final2` | Main run (300 basins, leads 1 and 3): coverage, width, forecast-high and outcome-high coverage by model, scenario, method |
| `trackA_basinlevel_final2.csv` | same, `--per-basin` | Per-basin counts (rows, covered rows, forecast-high rows, covered forecast-high rows, summed width) |
| `trackA_bootstrap_summary.csv` | `09_bootstrap_basins.py` | Basin-level cluster bootstrap: 95% intervals for coverage, forecast-high coverage and width; per-basin coverage spread (5th-95th percentile, share within 0.85-0.95); coverage by region |
| `trackA_warnings_final2.csv` | `08_trackA_runs.py` | High-flow warning recall and false-alarm rate with conformally calibrated margins (`alpha_w` 0.05 and 0.1; `kind`: U = quantile upper bound, P = point forecast; `_bins` = per missingness bin, `_complete` = calibrated on complete data) |
| `trackA_floodday_final2.csv` | `10_hanze_flood_days.py` | Coverage on documented HANZE flood days (95% intervals bootstrapped over events), share of misses above the upper bound |
| `trackA_floodday_warnings_final2.csv` | same | Recall of high-flow flood days and false-alarm rate on non-flood low-flow days for the calibrated warnings |
| `trackA_runs_holdout.csv`, `trackA_basinlevel_holdout.csv`, `trackA_warnings_holdout.csv`, `trackA_bootstrap_summary_holdout.csv` | `08_..` with `--holdout 0.3`, then `09_..` | Same quantities on 30% of basins unseen in training and calibration |
| `trackA_runs_lamah_to_gb.csv`, `trackA_runs_lamah_to_gb_cal.csv` | `08_..` transfer runs | Train on LamaH, test on CAMELS-GB; calibrated on LamaH or on CAMELS-GB (3 seeds, random 40-basin subsets per source) |
| `earlier_runs/` | earlier code versions | `trackA_runs_main.csv` (5 seeds, 80 basins), `trackA_runs_mda80.csv` (3 seeds, 80 basins, includes exact-mask and nested baselines), `trackA_runs_final.csv` (3 seeds, 300 basins; seeds share the same basins, so their spread understates uncertainty) |

Definitions: `coverage_predhigh` conditions on the forecast being above the basin's 95th percentile (training period);
`coverage_highflow` conditions on the observed flow being above it (selection on the outcome, below nominal by construction);
`width_log` is the mean interval width in log(q + 0.01) units.
