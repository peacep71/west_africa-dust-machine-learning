# -*- coding: utf-8 -*-
"""Global 90th-percentile contingency statistics as in the notebook."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("outputs")

def main():
    df = pd.read_parquet(OUT / "LightGBM_test_predictions_2021_2024.parquet")
    threshold = float(np.nanpercentile(df["dust_aod"], 90))
    df["obs_high_dust"] = df["dust_aod"] >= threshold
    df["pred_high_dust"] = df["dust_aod_pred"] >= threshold
    df["hit"] = df.obs_high_dust & df.pred_high_dust
    df["miss"] = df.obs_high_dust & ~df.pred_high_dust
    df["false_alarm"] = ~df.obs_high_dust & df.pred_high_dust
    h, m, f = df.hit.mean(), df.miss.mean(), df.false_alarm.mean()
    oh, ph = df.obs_high_dust.mean(), df.pred_high_dust.mean()
    result = {"threshold":threshold, "CAMS_HDF":oh, "pred_HDF":ph,
              "HF":h, "MF":m, "FAF":f,
              "POD": h / oh if oh else np.nan,
              "FAR": f / ph if ph else np.nan,
              "CSI": h / (h+m+f) if h+m+f else np.nan,
              "FB": ph / oh if oh else np.nan}
    pd.DataFrame([result]).to_csv(OUT / "high_dust_metrics.csv", index=False)
    df.to_parquet(OUT / "high_dust_events_2021_2024.parquet", index=False)
    freqs = df.groupby(["latitude", "longitude"], as_index=False)[
        ["obs_high_dust", "pred_high_dust", "hit", "miss", "false_alarm"]
    ].mean()
    freqs.to_csv(OUT / "high_dust_spatial_frequencies.csv", index=False)
    print(result)

if __name__ == "__main__":
    main()
