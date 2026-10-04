"""MAPE / WAPE of the main models. MAPE is undefined at entries=0 and explodes for tiny entries, so several
denominator floors are reported; WAPE = sum|err| / sum(actual) is the stable one."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
exec(open("train_baseline.py", encoding="utf-8").read().split("def fit_predict")[0])
X = mk(BASE + EVENT + PRICE); y = df.entries.values
yr = df.datetime.dt.year.values
tr = (df.split == "train").values

def fit(obj):
    p = dict(objective=obj, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, verbose=-1)
    if obj == "regression_l1": p["metric"] = "l1"
    itr, iva = tr & (yr <= 2024), tr & (yr == 2025)   # inner hold-out only to pick the number of trees
    m0 = lgb.LGBMRegressor(n_estimators=4000, **p).fit(X[itr], y[itr], eval_set=[(X[iva], y[iva])],
                                                     callbacks=[lgb.early_stopping(100, verbose=False)])
    m = lgb.LGBMRegressor(n_estimators=int(m0.best_iteration_ * 1.1), **p).fit(X[tr], y[tr])
    return np.clip(m.predict(X), 0, None)

preds = {"lag_7d": df.lag_7d.fillna(df.lag_mean_4w).fillna(0).values, "L1": fit("regression_l1"), "Poisson": fit("poisson")}
preds["blend"] = (preds["L1"] + preds["Poisson"]) / 2
masks = {"all": np.ones(len(df), bool), "event_day": df.event_today.values, "non_event_day": ~df.event_today.values,
         "end_hour(+1)": (df.is_end_hour | df.is_post_end_hour).values, "daytime 07-22h": df.hour.between(7, 22).values}
rows = []
for s in ["test"]:
    for mn, m in masks.items():
        sel = (df.split == s).values & m
        for name, p in preds.items():
            a, f = y[sel], p[sel]; e = np.abs(f - a)
            r = dict(split=s, subset=mn, model=name, n=int(sel.sum()), WAPE=100 * e.sum() / a.sum())
            for fl in [1, 100, 500]:
                k = a >= fl; r[f"MAPE(y>={fl})"] = 100 * (e[k] / a[k]).mean(); r[f"n(y>={fl})"] = int(k.sum())
            r["sMAPE"] = 100 * (2 * e / np.maximum(a + f, 1)).mean()
            rows.append(r)
res = pd.DataFrame(rows).round(1)
res.to_csv("mape_results.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(res[res.split == "test"].drop(columns="split").to_string(index=False))
