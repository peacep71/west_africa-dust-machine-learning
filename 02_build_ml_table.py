# -*- coding: utf-8 -*-
"""Dust aerosol optical-depth machine-learning research — code from supplied notebooks.

Execute from the repository root, with documented input data present locally.
"""

# ---- Original notebook cell 1 ----
import xarray as xr

# --------------------------------------------------
# Open ML-ready dataset that already contains u10 and v10
# --------------------------------------------------

input_file = "ML_ready_dust_AOD_ERA5_CHIRPS_2003_2024.nc"  # change to your actual filename

ds = xr.open_dataset(input_file)

print("Variables before adding lags:")
print(list(ds.data_vars))

# --------------------------------------------------
# Add lag variables
# --------------------------------------------------

ds["rain_lag1"] = ds["rainfall"].shift(time=1)
ds["rain_3day_sum"] = ds["rainfall"].shift(time=1).rolling(time=3, min_periods=3).sum()

ds["soil_moisture_lag1"] = ds["swvl1"].shift(time=1)
ds["wind_lag1"] = ds["wind_speed"].shift(time=1)
ds["blh_lag1"] = ds["blh"].shift(time=1)

# Optional, but I would NOT add these yet unless you want another sensitivity test
# ds["u10_lag1"] = ds["u10"].shift(time=1)
# ds["v10_lag1"] = ds["v10"].shift(time=1)

# --------------------------------------------------
# Add metadata
# --------------------------------------------------

ds["rain_lag1"].attrs["long_name"] = "Previous-day rainfall"
ds["rain_3day_sum"].attrs["long_name"] = "Previous 3-day rainfall sum"
ds["soil_moisture_lag1"].attrs["long_name"] = "Previous-day soil moisture layer 1"
ds["wind_lag1"].attrs["long_name"] = "Previous-day 10 m wind speed"
ds["blh_lag1"].attrs["long_name"] = "Previous-day boundary layer height"

# --------------------------------------------------
# Save new file with u10, v10, and lag variables
# --------------------------------------------------

encoding = {
    var: {
        "zlib": True,
        "complevel": 4,
        "dtype": "float32"
    }
    for var in ds.data_vars
}

output_file = "ML_ready_dust_AOD_ERA5_CHIRPS_lags_2003_2024.nc"

ds.to_netcdf(
    output_file,
    encoding=encoding
)

print("\nSaved:", output_file)

print("\nVariables now in dataset:")
print(list(ds.data_vars))

print("\nFinal dataset:")
print(ds)

# ---- Original notebook cell 2 ----
import xarray as xr
import pandas as pd

# Open lagged ML-ready dataset
ds = xr.open_dataset("ML_ready_dust_AOD_ERA5_CHIRPS_lags_2003_2024.nc")

# Define predictors and target
predictor_vars = [
    "u10",
    "v10",
    "wind_speed",
    "rh",
    "t2m",
    "swvl1",
    "blh",
    "rainfall",
    "rain_lag1",
    "rain_3day_sum",
    "soil_moisture_lag1",
    "wind_lag1",
    "blh_lag1"
]

target_var = "dust_aod"

# Keep only variables needed for ML
ml_ds = ds[predictor_vars + [target_var]]

# Convert from NetCDF grid format to ML table format
ml_table = ml_ds.to_dataframe().reset_index()

# Remove rows with missing predictors or missing target
ml_table = ml_table.dropna(subset=predictor_vars + [target_var])

# Add useful time columns
ml_table["year"] = ml_table["time"].dt.year
ml_table["month"] = ml_table["time"].dt.month

# Add season column
def get_season(month):
    if month in [12, 1, 2]:
        return "DJF"
    elif month in [3, 4, 5]:
        return "MAM"
    elif month in [6, 7, 8]:
        return "JJA"
    else:
        return "SON"

ml_table["season"] = ml_table["month"].apply(get_season)

# Check table
print(ml_table.head())
print(ml_table.shape)
print(ml_table.columns)
print(ml_table["year"].min(), ml_table["year"].max())
ml_table.to_parquet(
    "ML_table_dust_AOD_ERA5_CHIRPS_lags_2003_2024.parquet",
    index=False
)
