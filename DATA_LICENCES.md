# Data sources and licences

| Source | Used for | Licence (where it was checked) | Redistributed here? |
|---|---|---|---|
| Caravan (netCDF release) | Streamflow, ERA5-Land-derived forcing, basin attributes | CC BY 4.0 for the Caravan compilation/release; individual embedded/source datasets retain their own licences, documented in licenses/. | No |
| LamaH-CE (Caravan source `lamah`) | 150 basins | CC BY-SA 4.0 for the LamaH-CE dataset files. | No (but share-alike applies to derived data) |
| CAMELS-GB (Caravan source `camelsgb`) | 150 basins | Open Government Licence v3.0 (OGL v3.0), published by NERC EDS Environmental Information Data Centre.  | No |
| ERA5-Land (inside Caravan) | Forcing variables | Copernicus licence; Caravan describes it as permissive and allowing free distribution. | No |
| HANZE v3.0.2 beta (https://zenodo.org/records/23670886) | Flood events (dates, NUTS3 regions) and region polygons for the flood-day test | CC BY 4.0 International, explicitly listed under Rights on the Zenodo record. | No (derived event tables are rebuilt by `src/01_prepare_tables.py`) |
| FLODIS, Pakistan flood events | Exploratory only, not used for paper results | FLODIS: CC BY 4.0; Pakistan dataset: CC BY 4.0 | No |

What this repository does redistribute: the code (MIT), aggregate result tables (coverage, width, warning rates, bootstrap intervals,
per-basin coverage counts) and the list of basin identifiers. The result tables contain no streamflow or forcing values. If you are
unsure whether aggregate statistics derived from CC BY-SA data fall under share-alike, ask your institution's data steward; the
conservative option is to license `results/` as CC BY-SA 4.0.

Please cite the source datasets using the citation instructions of their own records: Caravan and its source datasets (LamaH-CE,
CAMELS-GB), ERA5-Land, and HANZE.
