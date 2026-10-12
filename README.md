# West African Dust AOD Machine Learning

Python code for the study **Machine-learning prediction of dust aerosol optical depth at 550 nm over West Africa** (2003–2024).

## Status — important

**Private, staging repository — not yet independently reproducible or ready to certify for AGU.**
The original notebooks were supplied by the author. They contain experimental versions and repeated training/plotting cells; this package separates the main workflow into `.py` scripts and retains a conversion of the original notebook as reference. We performed a syntax check only; the input datasets were not supplied, so we have **not executed or numerically validated** the code.

### Core scripts (run in order from the repository root)

1. `python scripts/01_prepare_gridded_data.py` — harmonize CAMS/ERA5/CHIRPS, derive wind speed and RH, regrid predictors, save ML-ready NetCDF.
2. `python scripts/02_build_ml_table.py` — create lagged predictors, drop missing rows, save ML table (Parquet).
3. `python scripts/03_train_and_evaluate.py` — compare linear regression, random forest and LightGBM; save validation predictions, refit LightGBM for 2003–2020, evaluate 2021–2024.
4. `python scripts/04_high_dust_events.py` — overall CAMS P90 event verification and gridded event frequencies.
5. `python scripts/05_feature_importance_shap.py` — save feature-importance table and a sampled SHAP plot.
6. `python scripts/06_daily_som.py` — original 4×4 CAMS SOM and model-frequency plots (requires Cartopy and MiniSom).
7. `python scripts/07_seasonal_bias_som.py` — 3×3 model-season bias SOM with node assignments and composite files.

`legacy_reference/ML_full_notebook_reference.py` is a **reference export** retaining additional SOM and plotting code from the original notebook. It includes alternative modelling and plotting versions and **should not** be treated as a tested, stand-alone executable script. `legacy_reference/seasonal_bias_SOM_inputs_reference.py` records original 12-field bias-preparation code. The complete set of 16 final figure outputs has not been consolidated and numerically verified. The SOM scripts have not yet been run on the actual input datasets.

## Local input data

The scripts expect these **daily** NetCDF inputs in the repository working directory (or you can edit paths locally):

- `dust_aod_daily.nc` — CAMS dust optical depth variable `duaod550`.
- `ERA5_daily.nc` — variables `u10`, `v10`, `t2m`, `d2m`, `swvl1`, `blh`.
- `CHIRPS_daily.nc` — precipitation variable `precip`.

These input files are **not included**. Do not commit large datasets, personal file paths, credentials, or author-only outputs. See the paper's data-availability statement for source datasets.

## Requirements

`python -m pip install -r requirements.txt`

Versions were not recoverable from the supplied notebooks; freeze a tested environment after confirmation.

## Before this repository is used as publication evidence

- **Reconcile final LightGBM trees:** One original cell refits with 2,000 trees, another with 1,251. The canonical script implements validation early stopping (maximum 2,000) then refits using the selected best iteration. Compare reported validation and test metrics with the manuscript before claiming reproduction.
- **Verify preprocessing:** Confirm the three input datasets are already daily and are on the expected grids/units. The notebook is not a raw-data downloader.
- **Reconcile SOM methods and plots:** The notebook contains both 3×3 seasonal bias and 4×4 daily CAMS SOM analyses, plus historical figure drafts. Verify the 3×3 seasonal-bias node ordering/assignments and 4×4 daily CAMS SOM against reported Figures 14–16; the newly organised 3×3 calculation is not yet confirmed to reproduce their numeric labels.
- **Provide inputs or documented preparation:** The paper's underlying public CAMS/ERA5/CHIRPS datasets must be referenced, and processed inputs or exact instructions supplied to reviewers.
- Run the entire workflow against actual data, reproduce the reported figures and numerical metrics, and save software versions before changing repository visibility or making an AGU availability declaration.

## Licence

No open-source licence has been selected. All code remains under the author's control. Repository visibility is determined by its GitHub settings.
