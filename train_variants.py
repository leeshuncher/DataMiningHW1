"""Variants on feature set C (BASE+EVENT+PRICE): L1 vs Poisson loss, global vs per-event-type models.
Same split as train_baseline.py (train 2018-19+2023, valid 2024-25, test 2026)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
exec(open("train_baseline.py", encoding="utf-8").read().split("def fit_predict")[0])  # df, split, BASE/EVENT/PRICE, mk

COLS = BASE + EVENT + PRICE
X = mk(COLS)
# segment by what the official schedule says for that day (priority concert > sport > other)
seg = np.select([~df.event_today, df.is_concert, df.is_sport], ["none", "concert", "sport"], "other")
df["seg"] = seg
print(df[df.split == "train"].seg.value_counts().to_dict())

def fit(mask, obj):
    tr, va = mask & (df.split == "train"), mask & (df.split == "valid")
    p = dict(objective=obj, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_estimators=3000)
    if obj == "regression_l1": p["metric"] = "l1"
    m = lgb.LGBMRegressor(**p)
    m.fit(X[tr], df.entries[tr], eval_set=[(X[va], df.entries[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m

def global_pred(obj):
    m = fit(df.entries == df.entries, obj)
    return np.clip(m.predict(X), 0, None)

def seg_pred(obj):
    out = np.zeros(len(df))
    for s in ["none", "concert", "sport", "other"]:
        mk_ = (df.seg == s).values
        m = fit(pd.Series(mk_, index=df.index), obj)
        out[mk_] = np.clip(m.predict(X[mk_]), 0, None)
    return out

preds = {"C_l1": global_pred("regression_l1"), "C_poisson": global_pred("poisson"),
         "seg_l1": seg_pred("regression_l1"), "seg_poisson": seg_pred("poisson")}
preds["avg(l1,poisson)"] = (preds["C_l1"] + preds["C_poisson"]) / 2

masks = {"all": df.entries == df.entries, "non_event_day": ~df.event_today,
         "concert": df.seg == "concert", "sport": df.seg == "sport", "other_event": df.seg == "other",
         "end_hour(+1)": df.is_end_hour | df.is_post_end_hour}
rows = []
for s in ["valid", "test"]:
    for mn, m_ in masks.items():
        sel = ((df.split == s) & m_).values
        for name, p in preds.items():
            e = p[sel] - df.entries.values[sel]
            rows.append((s, mn, name, int(sel.sum()), np.abs(e).mean(), np.sqrt((e ** 2).mean()),
                         e.mean()))
res = pd.DataFrame(rows, columns=["split", "subset", "model", "n", "MAE", "RMSE", "bias"]).round(1)
res.to_csv("variant_results.csv", index=False, encoding="utf-8-sig")
for s in ["valid", "test"]:
    print(s); print(res[res.split == s].pivot(index="subset", columns="model", values="MAE").to_string())
