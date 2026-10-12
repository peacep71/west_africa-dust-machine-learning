# -*- coding: utf-8 -*-
"""Train LR, RF and LightGBM with the manuscript's chronological split.

Manuscript periods: training 2003–2018, validation 2019–2020,
final LightGBM refit 2003–2020, independent test 2021–2024.

NOTE: The supplied notebook contains both a 2,000-tree final refit and a
1,251-tree final refit. This file uses the selected validation best iteration
for the final refit, as the paper's Methods describe. Verify the resulting
numbers against the submitted manuscript using the original input datasets.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

TABLE = Path("ML_table_dust_AOD_ERA5_CHIRPS_lags_2003_2024.parquet")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
PREDICTORS = ["u10", "v10", "wind_speed", "rh", "t2m", "swvl1", "blh",
              "rainfall", "rain_lag1", "rain_3day_sum", "soil_moisture_lag1",
              "wind_lag1", "blh_lag1"]
TARGET = "dust_aod"


def metrics(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    return {"R2": r2_score(obs, pred), "r": np.corrcoef(obs, pred)[0, 1],
            "RMSE": np.sqrt(mean_squared_error(obs, pred)),
            "MAE": mean_absolute_error(obs, pred), "MB": np.mean(pred - obs)}


def xy(df):
    return df[PREDICTORS].astype("float32"), df[TARGET].astype("float32")


def lightgbm_model(n_estimators):
    return lgb.LGBMRegressor(objective="regression", n_estimators=n_estimators,
        learning_rate=0.03, num_leaves=63, max_depth=-1,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=-1)


def main():
    table = pd.read_parquet(TABLE)
    train = table[table.year.between(2003, 2018)].copy()
    valid = table[table.year.between(2019, 2020)].copy()
    test = table[table.year.between(2021, 2024)].copy()
    trainval = table[table.year.between(2003, 2020)].copy()
    X_tr, y_tr = xy(train)
    X_va, y_va = xy(valid)
    X_tv, y_tv = xy(trainval)
    X_te, y_te = xy(test)

    lr = make_pipeline(StandardScaler(), LinearRegression())
    rf = RandomForestRegressor(n_estimators=100, max_depth=20,
                              min_samples_leaf=5, random_state=42, n_jobs=-1)
    candidate = lightgbm_model(2000)
    lr.fit(X_tr, y_tr)
    rf.fit(X_tr, y_tr)
    candidate.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="rmse",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    valid_pred = valid[["time", "latitude", "longitude", TARGET]].copy()
    outcomes = []
    for name, model, column in [
        ("Linear Regression", lr, "pred_lr"),
        ("Random Forest", rf, "pred_rf"),
        ("LightGBM", candidate, "pred_lgbm")]:
        p = model.predict(X_va)
        valid_pred[column] = p
        outcomes.append({"model": name, **metrics(y_va, p)})
    valid_pred.to_parquet(OUT / "validation_predictions_2019_2020.parquet", index=False)
    pd.DataFrame(outcomes).to_csv(OUT / "validation_metrics.csv", index=False)

    best_iteration = int(candidate.best_iteration_ or 2000)
    final = lightgbm_model(best_iteration)
    final.fit(X_tv, y_tv)
    test_predictions = final.predict(X_te)
    test_pred = test[["time", "latitude", "longitude", TARGET]].copy()
    test_pred["dust_aod_pred"] = test_predictions
    test_pred["error"] = test_pred["dust_aod_pred"] - test_pred[TARGET]
    test_pred.to_parquet(OUT / "LightGBM_test_predictions_2021_2024.parquet", index=False)
    pd.DataFrame([{"model": "LightGBM", "best_validation_iteration": best_iteration,
                   **metrics(y_te, test_predictions)}]).to_csv(
        OUT / "test_metrics_2021_2024.csv", index=False)
    joblib.dump(final, OUT / "final_LightGBM_model.joblib")
    print(pd.DataFrame(outcomes).to_string(index=False))
    print("Selected best iteration:", best_iteration)
    print("Independent test:", metrics(y_te, test_predictions))


if __name__ == "__main__":
    main()
