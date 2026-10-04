import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
exec(open("train_baseline.py", encoding="utf-8").read().split("def fit_predict")[0])
df["split"] = np.select([df.datetime.dt.year.isin([2018, 2019, 2023]), df.datetime.dt.year.isin([2024, 2025])], ["train", "valid"], "test")  # legacy 3-way split
X = mk(BASE + EVENT + PRICE); y = df.entries.values
tr, va = (df.split == "train").values, (df.split == "valid").values
p = dict(objective="regression_l1", metric="l1", learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8,
         subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_estimators=4000)
m = lgb.LGBMRegressor(**p).fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
df["pred"] = np.clip(m.predict(X), 0, None)
df["date"] = df.datetime.dt.date
ev = df[df.has_event_time & df.event_today]
rows = []
for (d, split), g in ev.groupby(["date", "split"]):
    if split == "train": continue
    g = g.set_index("hour")
    # event window hours around the end
    e = int(g.last_end_h.iloc[0] // 1); hrs = [h for h in range(e - 1, e + 3) if h in g.index and h < 24]
    sub = g.loc[hrs]
    rows.append(dict(date=d, split=split, end_h=g.last_end_h.iloc[0], imputed=bool(g.end_imputed.iloc[0]),
                     n_sess=int(g.n_sessions.iloc[0]), concert=bool(g.is_concert.iloc[0]), sport=bool(g.is_sport.iloc[0]),
                     act_peak=int(sub.entries.max()), act_peak_h=int(sub.entries.idxmax()),
                     pred_peak=int(sub.pred.max()), pred_peak_h=int(sub.pred.idxmax()),
                     act_sum=int(sub.entries.sum()), pred_sum=int(sub.pred.sum()),
                     lag7_sum=int(sub.lag_7d.fillna(0).sum()), price=g.price_median.iloc[0]))
r = pd.DataFrame(rows); r["err_pct"] = 100 * (r.pred_sum - r.act_sum) / r.act_sum
r["peak_shift"] = r.pred_peak_h - r.act_peak_h
r.to_csv("end_hour_diag.csv", index=False, encoding="utf-8-sig")
t = r[r.split == "test"]
print("test event days:", len(t), " valid:", (r.split == "valid").sum())
print("median abs err% of window sum:", t.err_pct.abs().median().round(1), " mean bias%:", t.err_pct.mean().round(1))
print("peak hour: pred==actual", (t.peak_shift == 0).mean().round(2), " shifted", t.peak_shift.value_counts().to_dict())
print("corr(act_sum, pred_sum)", np.corrcoef(t.act_sum, t.pred_sum)[0, 1].round(2))
print("by type: "); print(t.assign(type=np.where(t.concert, "concert", np.where(t.sport, "sport", "other"))).groupby("type").agg(
    n=("date", "size"), act=("act_sum", "mean"), pred=("pred_sum", "mean"), abs_err=("err_pct", lambda s: s.abs().median())).round(0))
print("imputed end:", t.groupby("imputed").err_pct.agg(lambda s: s.abs().median()).round(1).to_dict())
print("window actual sum quantiles", t.act_sum.quantile([.1, .5, .9]).to_dict())
print(t.sort_values("err_pct")[["date","end_h","n_sess","act_sum","pred_sum","err_pct","act_peak_h","pred_peak_h","price"]].head(8).to_string())
print(t.sort_values("err_pct")[["date","end_h","n_sess","act_sum","pred_sum","err_pct","act_peak_h","pred_peak_h","price"]].tail(8).to_string())
