# -*- coding: utf-8 -*-
"""Dust aerosol optical-depth machine-learning research — code from supplied notebooks.

Execute from the repository root, with documented input data present locally.
"""

# ---- Original notebook cell 1 ----
import xarray as xr

# Open files
dust = xr.open_dataset("dust_aod_daily.nc")
era5 = xr.open_dataset("ERA5_daily.nc")
chirps = xr.open_dataset("CHIRPS_daily.nc")

# Rename coordinates for consistency
dust = dust.rename({"valid_time": "time"})
era5 = era5.rename({"lat": "latitude", "lon": "longitude"})

# Select common study period
start_date = "2003-01-01"
end_date = "2024-12-31"

dust = dust.sel(time=slice(start_date, end_date))
era5 = era5.sel(time=slice(start_date, end_date))
chirps = chirps.sel(time=slice(start_date, end_date))

# Remove unnecessary bound variables from ERA5 if present
era5 = era5.drop_vars(["time_bnds", "depth_bnds"], errors="ignore")

# Remove depth dimension from swvl1
if "depth" in era5["swvl1"].dims:
    era5["swvl1"] = era5["swvl1"].squeeze("depth", drop=True)

# Check final time sizes
print("Dust:", dust.time.values[0], dust.time.values[-1], dust.sizes["time"])
print("ERA5:", era5.time.values[0], era5.time.values[-1], era5.sizes["time"])
print("CHIRPS:", chirps.time.values[0], chirps.time.values[-1], chirps.sizes["time"])

# ---- Original notebook cell 2 ----
# Make all time coordinates daily dates at 00:00
dust = dust.assign_coords(time=dust.time.dt.floor("D"))
era5 = era5.assign_coords(time=era5.time.dt.floor("D"))
chirps = chirps.assign_coords(time=chirps.time.dt.floor("D"))

# Confirm time coordinates now match exactly
import numpy as np

print("Dust vs ERA5 time match:", np.array_equal(dust.time.values, era5.time.values))
print("Dust vs CHIRPS time match:", np.array_equal(dust.time.values, chirps.time.values))

print(dust.time.values[0], dust.time.values[-1], dust.sizes["time"])
print(era5.time.values[0], era5.time.values[-1], era5.sizes["time"])
print(chirps.time.values[0], chirps.time.values[-1], chirps.sizes["time"])

# ---- Original notebook cell 3 ----
import xarray as xr
import numpy as np

# --------------------------------------------------
# 1. Define dust AOD target grid
# --------------------------------------------------
target_lat = dust["latitude"]
target_lon = dust["longitude"]

print("Dust grid:")
print("lat:", float(target_lat.min()), "to", float(target_lat.max()), "n =", target_lat.size)
print("lon:", float(target_lon.min()), "to", float(target_lon.max()), "n =", target_lon.size)

# --------------------------------------------------
# 2. Regrid ERA5 to dust grid
# --------------------------------------------------
era5_regrid = era5.interp(
    latitude=target_lat,
    longitude=target_lon,
    method="linear"
)

# --------------------------------------------------
# 3. Regrid CHIRPS to dust grid
# --------------------------------------------------
# Use nearest for CHIRPS to avoid edge NaNs near latitude = 0.0
chirps_regrid = chirps.interp(
    latitude=target_lat,
    longitude=target_lon,
    method="nearest"
)

# --------------------------------------------------
# 4. Check results
# --------------------------------------------------
print("\nERA5 regridded:")
print(era5_regrid)

print("\nCHIRPS regridded:")
print(chirps_regrid)

print("\nGrid check:")
print("Dust lat size:", dust.sizes["latitude"])
print("ERA5 lat size:", era5_regrid.sizes["latitude"])
print("CHIRPS lat size:", chirps_regrid.sizes["latitude"])

print("Dust lon size:", dust.sizes["longitude"])
print("ERA5 lon size:", era5_regrid.sizes["longitude"])
print("CHIRPS lon size:", chirps_regrid.sizes["longitude"])

print("\nTime check:")
print("Dust vs ERA5:", np.array_equal(dust.time.values, era5_regrid.time.values))
print("Dust vs CHIRPS:", np.array_equal(dust.time.values, chirps_regrid.time.values))

# ---- Original notebook cell 4 ----
import numpy as np
import xarray as xr

# --------------------------------------------------
# 1. Clean leftover ERA5 depth coordinate
# --------------------------------------------------
era5_regrid = era5_regrid.squeeze(drop=True)
era5_regrid = era5_regrid.drop_vars("depth", errors="ignore")

# --------------------------------------------------
# 2. Calculate wind speed
# --------------------------------------------------
wind_speed = np.sqrt(era5_regrid["u10"]**2 + era5_regrid["v10"]**2)
wind_speed.name = "wind_speed"
wind_speed.attrs["units"] = "m s-1"
wind_speed.attrs["long_name"] = "10 m wind speed"

# --------------------------------------------------
# 3. Calculate relative humidity from t2m and d2m
#    ERA5 t2m and d2m are in Kelvin
# --------------------------------------------------
t_c = era5_regrid["t2m"] - 273.15
td_c = era5_regrid["d2m"] - 273.15

rh = 100 * (
    np.exp((17.625 * td_c) / (243.04 + td_c)) /
    np.exp((17.625 * t_c) / (243.04 + t_c))
)

rh = rh.clip(min=0, max=100)
rh.name = "rh"
rh.attrs["units"] = "%"
rh.attrs["long_name"] = "2 m relative humidity"

# --------------------------------------------------
# 4. Build clean predictor dataset
# --------------------------------------------------
predictors = xr.Dataset(
    {
        "u10": era5_regrid["u10"],
        "v10": era5_regrid["v10"],
        "wind_speed": wind_speed,
        "rh": rh,
        "t2m": era5_regrid["t2m"],
        "swvl1": era5_regrid["swvl1"],
        "blh": era5_regrid["blh"],
        "rainfall": chirps_regrid["precip"],
    }
)

# Keep memory lighter
predictors = predictors.astype("float32")

# Target
target = dust["duaod550"].astype("float32")
target.name = "dust_aod"

# --------------------------------------------------
# 5. Check result
# --------------------------------------------------
print(predictors)
print(target)

print("Predictor variables:", list(predictors.data_vars))
print("Target variable:", target.name)

# ---- Original notebook cell 5 ----
# Combine predictors and target into one clean dataset
ml_data = predictors.copy()
ml_data["dust_aod"] = target

# Add basic metadata
ml_data.attrs["description"] = (
    "Daily ML-ready dataset for dust AOD prediction over West Africa. "
    "Predictors from ERA5 and CHIRPS regridded to CAMS dust AOD grid."
)
ml_data.attrs["period"] = "2003-01-01 to 2024-12-31"
ml_data.attrs["grid"] = "CAMS dust AOD grid, 0.75 degree"

# Compression settings
encoding = {
    var: {
        "zlib": True,
        "complevel": 4,
        "dtype": "float32"
    }
    for var in ml_data.data_vars
}

# Save final processed file
ml_data.to_netcdf(
    "ML_ready_dust_AOD_ERA5_CHIRPS_2003_2024.nc",
    encoding=encoding
)
