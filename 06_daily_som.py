# -*- coding: utf-8 -*-
"""Original 4x4 daily CAMS SOM workflow (Figure 15 and 16 equivalents).

The code is adapted directly from the submitted notebook; field values and
published figures still require independent reproduction against input data.
"""
import os
import pandas as pd
import xarray as xr
from IPython.display import display

val_df = pd.read_parquet("outputs/validation_predictions_2019_2020.parquet")
val_xr = (val_df.set_index(["time", "latitude", "longitude"])
          .to_xarray().sortby("time").sortby("latitude").sortby("longitude"))
ds = xr.open_dataset("ML_ready_dust_AOD_ERA5_CHIRPS_lags_2003_2024.nc")

# ============================================================
# FIGURE 15
# 16-NODE SOM CLASSIFICATION OF DAILY CAMS DUST AOD
# 2003–2024
# ============================================================

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

try:
    from minisom import MiniSom
except ImportError:
    raise ImportError("Install MiniSom first: pip install minisom")

# ============================================================
# 1. SETTINGS
# ============================================================

outdir16 = "SOM_16node_daily_CAMS"
os.makedirs(outdir16, exist_ok=True)

SOM_ROWS = 4
SOM_COLS = 4
N_ITERATIONS = 30000
RANDOM_SEED = 42

regime_order16 = [f"R{i}" for i in range(1, 17)]

FS_TITLE = 22
FS_NODE = 17
FS_VALUE = 17
FS_TICK = 15
FS_CBAR_LABEL = 20
FS_CBAR_TICK = 17

FIG_WIDTH = 16.0
FIG_HEIGHT = 8.7

LEFT = 0.050
RIGHT = 0.900
BOTTOM = 0.070
TOP = 0.905

WSPACE = 0.025
HSPACE = 0.035

CBAR_LEFT = 0.915
CBAR_BOTTOM = 0.130
CBAR_WIDTH = 0.016
CBAR_HEIGHT = 0.710

# ============================================================
# 2. PLACE FULL CAMS DATA ON THE VALIDATION GRID
# ============================================================

cams_full = ds["dust_aod"].interp(
    latitude=val_xr["latitude"],
    longitude=val_xr["longitude"],
    method="nearest"
)

# Cells available for CAMS and all three model predictions
validation_fields = [
    val_xr["dust_aod"],
    val_xr["pred_lr"],
    val_xr["pred_rf"],
    val_xr["pred_lgbm"]
]

validation_mask = xr.concat(
    [field.notnull().all(dim="time") for field in validation_fields],
    dim="field"
).all(dim="field")

# Retain cells with CAMS data for at least 99% of 2003–2024
full_cams_mask = cams_full.notnull().mean(dim="time") >= 0.99

common_mask16 = validation_mask & full_cams_mask

n_cells = int(common_mask16.sum().values)

print("Full CAMS shape:", cams_full.shape)
print("Number of SOM grid cells:", n_cells)

if n_cells == 0:
    raise ValueError("No common valid grid cells were found.")

# ============================================================
# 3. BUILD DAILY SOM INPUT MATRIX
# ============================================================

mask_np = common_mask16.values

X_cams_raw = cams_full.values[:, mask_np].astype(np.float32)

# Fill occasional missing CAMS values with the cell temporal mean
cell_means = np.nanmean(X_cams_raw, axis=0)

missing_rows, missing_cols = np.where(~np.isfinite(X_cams_raw))

if len(missing_rows) > 0:
    X_cams_raw[missing_rows, missing_cols] = cell_means[missing_cols]

if not np.isfinite(X_cams_raw).all():
    raise ValueError("The SOM matrix still contains missing values.")

# One global transformation; this does not alter relative spatial distances
som_global_mean16 = float(X_cams_raw.mean())
som_global_std16 = float(X_cams_raw.std())

X_cams_som16 = (
    X_cams_raw - som_global_mean16
) / som_global_std16

print("SOM input matrix:", X_cams_som16.shape)
print("Daily maps:", X_cams_som16.shape[0])
print("Spatial cells:", X_cams_som16.shape[1])

# ============================================================
# 4. TRAIN THE 4 × 4 SOM
# ============================================================

som16 = MiniSom(
    x=SOM_COLS,
    y=SOM_ROWS,
    input_len=X_cams_som16.shape[1],
    sigma=1.2,
    learning_rate=0.5,
    neighborhood_function="gaussian",
    random_seed=RANDOM_SEED
)

som16.random_weights_init(X_cams_som16)

som16.train_random(
    X_cams_som16,
    num_iteration=N_ITERATIONS,
    verbose=True
)

# ============================================================
# 5. ASSIGN EACH DAILY CAMS MAP TO A SOM NODE
# ============================================================

winner_coordinates16 = np.array([
    som16.winner(sample)
    for sample in X_cams_som16
])

# MiniSom returns coordinates as x, y
node_assignments16 = (
    winner_coordinates16[:, 1] * SOM_COLS
    + winner_coordinates16[:, 0]
    + 1
)

som_daily16 = pd.DataFrame({
    "time": pd.to_datetime(cams_full["time"].values),
    "SOM Node": node_assignments16,
    "som_regime": [
        f"R{node}" for node in node_assignments16
    ]
})

som_daily16.to_csv(
    f"{outdir16}/daily_CAMS_SOM16_assignments.csv",
    index=False
)

# ============================================================
# 6. QUANTIZATION AND TOPOGRAPHIC ERRORS
# ============================================================

quantization_error16 = som16.quantization_error(X_cams_som16)
topographic_error16 = som16.topographic_error(X_cams_som16)

print("\nSOM quality measures")
print("Quantization error:", round(quantization_error16, 4))
print("Topographic error:", round(topographic_error16, 4))

# ============================================================
# 7. BUILD NODE COMPOSITES AND SUMMARY
# ============================================================

composite_maps16 = {}
summary_rows16 = []

total_days16 = cams_full.sizes["time"]

for node in range(1, 17):

    regime = f"R{node}"
    day_indices = np.where(node_assignments16 == node)[0]
    n_days = len(day_indices)

    if n_days == 0:
        composite_maps16[regime] = None

        summary_rows16.append({
            "som_regime": regime,
            "SOM Node": node,
            "n_days": 0,
            "percent_contribution": 0.0,
            "mean_dust_aod": np.nan
        })

        continue

    composite = cams_full.isel(
        time=day_indices
    ).mean(
        dim="time",
        skipna=True
    ).where(common_mask16)

    composite_maps16[regime] = composite

    summary_rows16.append({
        "som_regime": regime,
        "SOM Node": node,
        "n_days": n_days,
        "percent_contribution": 100.0 * n_days / total_days16,
        "mean_dust_aod": float(
            composite.mean(skipna=True).values
        )
    })

regime_summary16 = pd.DataFrame(summary_rows16)

regime_summary16.to_csv(
    f"{outdir16}/CAMS_SOM16_node_summary.csv",
    index=False
)

print("\nNode summary")
display(regime_summary16)

# Save composite maps as NetCDF
composite_list16 = []

for regime in regime_order16:

    if composite_maps16[regime] is not None:
        da = composite_maps16[regime]
    else:
        da = xr.full_like(
            cams_full.isel(time=0),
            np.nan
        )

    da = da.expand_dims(som_regime=[regime])
    composite_list16.append(da)

composite_ds16 = xr.concat(
    composite_list16,
    dim="som_regime"
).to_dataset(name="dust_aod_composite")

composite_ds16.to_netcdf(
    f"{outdir16}/CAMS_SOM16_composite_maps.nc"
)

# ============================================================
# 8. MAP HELPERS
# ============================================================

lon_min, lon_max = -20, 20
lat_min, lat_max = 0, 20

ocean_feature = cfeature.OCEAN.with_scale("50m")
coast_feature = cfeature.COASTLINE.with_scale("50m")
borders_feature = cfeature.BORDERS.with_scale("50m")


def style_map16(ax):

    ax.set_extent(
        [lon_min, lon_max, lat_min, lat_max],
        crs=ccrs.PlateCarree()
    )

    ax.add_feature(
        ocean_feature,
        facecolor="white",
        edgecolor="none",
        zorder=3
    )

    ax.add_feature(
        coast_feature,
        linewidth=0.60,
        edgecolor="black",
        zorder=4
    )

    ax.add_feature(
        borders_feature,
        linewidth=0.35,
        edgecolor="black",
        zorder=4
    )

    ax.set_xticks(
        [-15, -5, 5, 15],
        crs=ccrs.PlateCarree()
    )

    ax.set_yticks(
        [5, 10, 15],
        crs=ccrs.PlateCarree()
    )

    ax.set_xticklabels(
        ["15°W", "5°W", "5°E", "15°E"],
        fontsize=FS_TICK
    )

    ax.set_yticklabels(
        ["5°N", "10°N", "15°N"],
        fontsize=FS_TICK
    )

    ax.tick_params(
        axis="both",
        labelsize=FS_TICK,
        length=2,
        pad=1
    )


def annotate_panel16(
    ax,
    node,
    mean_aod,
    contribution
):

    ax.text(
        0.03, 0.96,
        f"{node}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FS_NODE,
        fontweight="bold",
        zorder=10
    )

    if np.isfinite(mean_aod):
        left_text = f"{mean_aod:.2f}"
    else:
        left_text = "Empty"

    ax.text(
        0.03, 0.03,
        left_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FS_VALUE,
        fontweight="bold",
        zorder=10
    )

    ax.text(
        0.97, 0.03,
        f"{contribution:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=FS_VALUE,
        fontweight="bold",
        zorder=10
    )

# ============================================================
# 9. COMMON COLOUR SCALE
# ============================================================

all_composite_values16 = []

for regime in regime_order16:

    da = composite_maps16[regime]

    if da is not None:
        values = da.values.ravel()
        values = values[np.isfinite(values)]
        all_composite_values16.append(values)

all_composite_values16 = np.concatenate(
    all_composite_values16
)

cams_vmin16 = 0.05
cams_vmax16 = float(
    np.nanpercentile(all_composite_values16, 98)
)

# ============================================================
# 10. PLOT FIGURE 15
# ============================================================

fig, axes = plt.subplots(
    4, 4,
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    subplot_kw={
        "projection": ccrs.PlateCarree()
    }
)

pcm = None

for i in range(4):
    for j in range(4):

        ax = axes[i, j]

        node = i * 4 + j + 1
        regime = f"R{node}"

        row = regime_summary16.loc[
            regime_summary16["som_regime"] == regime
        ].iloc[0]

        composite = composite_maps16[regime]

        if composite is not None:

            pcm = ax.pcolormesh(
                composite["longitude"],
                composite["latitude"],
                composite,
                transform=ccrs.PlateCarree(),
                shading="auto",
                cmap="YlOrBr",
                vmin=cams_vmin16,
                vmax=cams_vmax16,
                zorder=1
            )

        style_map16(ax)

        if i < 3:
            ax.set_xticklabels([])

        if j > 0:
            ax.set_yticklabels([])

        annotate_panel16(
            ax=ax,
            node=node,
            mean_aod=float(row["mean_dust_aod"]),
            contribution=float(
                row["percent_contribution"]
            )
        )

fig.subplots_adjust(
    left=LEFT,
    right=RIGHT,
    bottom=BOTTOM,
    top=TOP,
    wspace=WSPACE,
    hspace=HSPACE
)

cax = fig.add_axes([
    CBAR_LEFT,
    CBAR_BOTTOM,
    CBAR_WIDTH,
    CBAR_HEIGHT
])

cbar = fig.colorbar(
    pcm,
    cax=cax,
    orientation="vertical"
)

cbar.set_label(
    "Mean CAMS dust AOD",
    fontsize=FS_CBAR_LABEL
)

cbar.ax.tick_params(
    labelsize=FS_CBAR_TICK
)



fig15_path = (
    f"{outdir16}/"
    "Figure15_CAMS_daily_dust_AOD_SOM_16nodes.png"
)

plt.savefig(
    fig15_path,
    dpi=1000,
    bbox_inches="tight"
)

plt.show()

print("\nSaved:", fig15_path)

# ============================================================
# FIGURE 16
# PERCENT CONTRIBUTION TO THE 16 CAMS-DEFINED SOM NODES
# CAMS vs LR vs RF vs LightGBM
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. ASSIGN A DAILY FIELD TO THE TRAINED CAMS SOM
# ------------------------------------------------------------

mask_np = common_mask16.values

def assign_to_som16(da, label):

    X = da.values[:, mask_np].astype(np.float32)

    if not np.isfinite(X).all():
        raise ValueError(f"{label} contains missing values on the SOM grid.")

    # Use exactly the same transformation used to train the SOM
    X_scaled = (
        X - som_global_mean16
    ) / som_global_std16

    winners = np.array([
        som16.winner(sample)
        for sample in X_scaled
    ])

    nodes = (
        winners[:, 1] * SOM_COLS
        + winners[:, 0]
        + 1
    )

    return nodes


# ------------------------------------------------------------
# 2. NODE ASSIGNMENTS
# ------------------------------------------------------------

cams_nodes16 = assign_to_som16(
    val_xr["dust_aod"],
    "CAMS"
)

lr_nodes16 = assign_to_som16(
    val_xr["pred_lr"],
    "Linear Regression"
)

rf_nodes16 = assign_to_som16(
    val_xr["pred_rf"],
    "Random Forest"
)

lgbm_nodes16 = assign_to_som16(
    val_xr["pred_lgbm"],
    "LightGBM"
)


# ------------------------------------------------------------
# 3. PERCENT CONTRIBUTIONS
# ------------------------------------------------------------

def node_percentages16(nodes):

    counts = np.bincount(
        nodes,
        minlength=17
    )[1:17]

    return 100.0 * counts / len(nodes)


cams_values16 = node_percentages16(cams_nodes16)
lr_values16 = node_percentages16(lr_nodes16)
rf_values16 = node_percentages16(rf_nodes16)
lgbm_values16 = node_percentages16(lgbm_nodes16)


cams_pct16 = {
    f"R{i + 1}": cams_values16[i]
    for i in range(16)
}

lr_pct16 = {
    f"R{i + 1}": lr_values16[i]
    for i in range(16)
}

rf_pct16 = {
    f"R{i + 1}": rf_values16[i]
    for i in range(16)
}

lgbm_pct16 = {
    f"R{i + 1}": lgbm_values16[i]
    for i in range(16)
}


# ------------------------------------------------------------
# 4. SAVE DAILY ASSIGNMENTS
# ------------------------------------------------------------

assignment_df16 = pd.DataFrame({
    "time": pd.to_datetime(val_xr["time"].values),
    "CAMS_node": cams_nodes16,
    "LR_node": lr_nodes16,
    "RF_node": rf_nodes16,
    "LGBM_node": lgbm_nodes16
})

assignment_df16.to_csv(
    f"{outdir16}/validation_SOM16_daily_assignments.csv",
    index=False
)


# ------------------------------------------------------------
# 5. SAVE CONTRIBUTION TABLE
# ------------------------------------------------------------

contribution_df16 = pd.DataFrame({
    "SOM Node": np.arange(1, 17),
    "CAMS": cams_values16,
    "Linear Regression": lr_values16,
    "Random Forest": rf_values16,
    "LightGBM": lgbm_values16
})

contribution_df16.to_csv(
    f"{outdir16}/validation_SOM16_percent_contributions.csv",
    index=False
)

display(contribution_df16.round(2))


# ------------------------------------------------------------
# 6. PLOT SETTINGS
# ------------------------------------------------------------

FIG_TITLE = (
    "Percentage contribution of CAMS and model-predicted "
    "daily dust-AOD fields to the 16 CAMS SOM nodes"
)

FS_TITLE = 22
FS_NODE = 19
FS_XTICK = 17
FS_YTICK = 17
FS_YLABEL = 20

FIG_WIDTH = 16
FIG_HEIGHT = 11

BAR_LABELS = ["CAMS", "LR", "RF", "LGBM"]

BAR_COLORS = [
    "#1f77b4",
    "#00FF33",
    "#00CC00",
    "#006400"
]

BAR_WIDTH = 0.98

all_values16 = np.concatenate([
    cams_values16,
    lr_values16,
    rf_values16,
    lgbm_values16
])

ymax16 = max(10, np.nanmax(all_values16) * 1.15)


# ------------------------------------------------------------
# 7. PLOT 4 × 4 PANELS
# ------------------------------------------------------------

fig, axes = plt.subplots(
    4, 4,
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    sharey=True
)

bar_x = np.arange(4)

for i in range(4):
    for j in range(4):

        ax = axes[i, j]

        node = i * 4 + j + 1
        idx = node - 1

        values = [
            cams_values16[idx],
            lr_values16[idx],
            rf_values16[idx],
            lgbm_values16[idx]
        ]

        ax.bar(
            bar_x,
            values,
            width=BAR_WIDTH,
            color=BAR_COLORS,
            edgecolor="black",
            linewidth=0.6
        )

        ax.text(
            0.04,
            0.93,
            f"{node}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FS_NODE,
            fontweight="bold"
        )

        ax.set_xticks(bar_x)
        ax.set_xticklabels(
            BAR_LABELS,
            fontsize=FS_XTICK
        )

        ax.set_ylim(0, ymax16)
        ax.margins(x=0.01)

        ax.tick_params(
            axis="y",
            labelsize=FS_YTICK
        )

        if j > 0:
            ax.tick_params(labelleft=False)

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)


# ------------------------------------------------------------
# 8. LABELS AND SPACING
# ------------------------------------------------------------

fig.text(
    0.025,
    0.50,
    "Percent contribution (%)",
    rotation="vertical",
    va="center",
    fontsize=FS_YLABEL,
    fontweight="bold"
)



fig.subplots_adjust(
    left=0.070,
    right=0.985,
    bottom=0.070,
    top=0.925,
    wspace=0.060,
    hspace=0.145
)


# ------------------------------------------------------------
# 9. SAVE
# ------------------------------------------------------------

fig16_path = (
    f"{outdir16}/"
    "Figure16_percent_contribution_CAMS_models_SOM16.png"
)

plt.savefig(
    fig16_path,
    dpi=1000,
    bbox_inches="tight"
)

plt.show()

print("Saved:", fig16_path)
