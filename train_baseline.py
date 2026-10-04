"""Baselines: same-hour-last-week vs LightGBM A (no events) vs B (+ official event features).
Time split: train 2018-19 + 2023, valid 2024-25, test 2026."""
import numpy as np
import pandas as pd
import lightgbm as lgb

df = pd.read_parquet("master.parquet").reset_index()
df = df[df.has_record].copy()
y = df.datetime.dt.year
split = np.select([y.isin([2018, 2019, 2023]), y.isin([2024, 2025])], ["train", "valid"], "test")
df["split"] = split

BASE = ["hour", "dow", "month", "is_holiday", "is_makeup_workday", "is_day_before_holiday",
        "is_day_after_holiday", "temperature_2m_act", "apparent_temperature_act",
        "relative_humidity_2m_act", "precipitation_act", "weather_code_act", "wind_speed_10m_act",
        "lag_7d", "lag_14d", "lag_21d", "lag_364d", "lag_mean_4w"]
EVENT = ["n_events", "n_sessions", "first_start_h", "last_start_h", "last_end_h", "end_imputed",
         "is_concert", "is_sport", "is_other_event", "event_today", "has_event_time",
         "rel_end_h", "rel_start_h", "is_end_hour", "is_post_end_hour"]

PREV = ["prev_event_win", "prev3_event_win", "prev_type_win"]
PRICE = ["price_median", "price_min", "price_mean", "n_price_tiers"]

def mk(cols):
    X = df[cols].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    return X

def fit_predict(cols):
    X = mk(cols)
    tr, va = df.split == "train", df.split == "valid"
    p = dict(objective="regression_l1", learning_rate=0.03, num_leaves=31, min_child_samples=20,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_estimators=3000)
    m = lgb.LGBMRegressor(**p)
    m.fit(X[tr], df.entries[tr], eval_set=[(X[va], df.entries[va])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    print(f"  best_iter={m.best_iteration_}")
    return np.clip(m.predict(X), 0, None), m

preds = {"lag_7d": df.lag_7d.fillna(df.lag_mean_4w).fillna(0).values}
preds["LGBM_A"], _ = fit_predict(BASE)
preds["LGBM_B"], mB = fit_predict(BASE + EVENT)
preds["LGBM_C"], mC = fit_predict(BASE + EVENT + PRICE)
preds["LGBM_D"], mD = fit_predict(BASE + EVENT + PRICE + PREV)

masks = {"all": df.entries == df.entries,
         "event_day": df.event_today,
         "non_event_day": ~df.event_today,
         "end_hour(+1)": df.is_end_hour | df.is_post_end_hour}
rows = []
for s in ["valid", "test"]:
    for mn, mk_ in masks.items():
        sel = (df.split == s) & mk_
        for name, p in preds.items():
            e = p[sel.values] - df.entries[sel].values
            rows.append((s, mn, name, int(sel.sum()), np.abs(e).mean(), np.sqrt((e ** 2).mean())))
res = pd.DataFrame(rows, columns=["split", "subset", "model", "n", "MAE", "RMSE"]).round(1)
print(res.to_string(index=False))
res.to_csv("baseline_results.csv", index=False, encoding="utf-8-sig")
imp = pd.Series(mD.feature_importances_, index=mD.feature_name_).sort_values(ascending=False)
print(imp.head(12).to_string())
