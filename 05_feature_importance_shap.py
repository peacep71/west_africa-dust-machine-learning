# -*- coding: utf-8 -*-
"""Gain importance and SHAP sampling for the final LightGBM model."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

OUT = Path("outputs")
PREDICTORS = ["u10", "v10", "wind_speed", "rh", "t2m", "swvl1", "blh",
              "rainfall", "rain_lag1", "rain_3day_sum", "soil_moisture_lag1",
              "wind_lag1", "blh_lag1"]

def main():
    OUT.mkdir(exist_ok=True)
    model = joblib.load(OUT / "final_LightGBM_model.joblib")
    table = pd.read_parquet("ML_table_dust_AOD_ERA5_CHIRPS_lags_2003_2024.parquet")
    test = table[table.year.between(2021, 2024)]
    X_test = test[PREDICTORS].astype("float32")
    gains = pd.DataFrame({"predictor": PREDICTORS,
                          "gain": model.booster_.feature_importance(importance_type="gain"),
                          "split_count": model.booster_.feature_importance(importance_type="split")})
    gains.sort_values("gain", ascending=False).to_csv(
        OUT / "feature_importance.csv", index=False)
    # Sampling is only for SHAP interpretation; it does not alter the fitted model.
    X_sample = X_test.sample(n=min(10000, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_sample)
    shap.summary_plot(explanation.values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(OUT / "SHAP_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    np.savez_compressed(OUT / "SHAP_sample_values.npz",
                        shap_values=np.asarray(explanation.values),
                        input_values=X_sample.to_numpy())
    print("Saved feature importance and sampled SHAP summary to", OUT)

if __name__ == "__main__":
    main()
