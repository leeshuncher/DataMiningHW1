"""Go / no-go check (proposal section 'Go / no-go check by Wed 10/7').
Train 2024, test 2025 (strictly by time; alpha chosen with TimeSeriesSplit inside 2024). Hourly (60 min) slots.
Models: same slot last week (lag 7 d), train mean per hour x weekday, Ridge on calendar/holiday features only,
Ridge on calendar + lags (same slot last week, mean of the last 4 same weekdays), each on minutes and on log(minutes).
Rows with missing target or missing lag are dropped for every model, so all models are scored on the same rows.
Go rule: MAE on holiday days >= 20% lower than same-slot-last-week, and ordinary days not worse.
Output: go_nogo_results.csv. Also: decision metric (see bottom). SLOT=30 python go_nogo.py runs the 30-minute version (files get a _30min suffix).
Usage: python go_nogo.py"""
import os
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit

SLOT = int(os.environ.get("SLOT", 60)); SUF = "" if SLOT == 60 else f"_{SLOT}min"; PER_DAY = 1440 // SLOT
t = pd.read_csv(f"data/travel_{SLOT}min.csv", parse_dates=["depart"]).set_index("depart")
cal = pd.read_csv("data/calendar.csv", parse_dates=["date"])
d = t[["minutes"]].copy(); d["date"] = d.index.normalize(); d["hour"] = (d.index.hour * 60 + d.index.minute) // SLOT   # slot index within the day
d = d.merge(cal[["date", "daytype", "dow", "block_len"]], left_on="date", right_on="date", how="left").set_index(t.index)
s = d.minutes
d["lag7"] = s.shift(7 * PER_DAY)
d["lag4mean"] = pd.concat([s.shift(k * 7 * PER_DAY) for k in (1, 2, 3, 4)], axis=1).mean(axis=1, skipna=False)
long_ = d.daytype.str.startswith("lw_")
d["len_b"] = np.where(long_, np.where(d.block_len >= 5, "5+", d.block_len.astype(str)), "-")
d["dtype_len"] = d.daytype + "_" + d.len_b
d["hour_s"] = d.hour.astype(str); d["dow_s"] = d.dow.astype(str)
d["h_dtype"] = d.hour_s + "|" + d.daytype; d["h_dow"] = d.hour_s + "|" + d.dow_s; d["h_dtl"] = d.hour_s + "|" + d.dtype_len
CAT = ["hour_s", "dow_s", "daytype", "dtype_len", "h_dtype", "h_dow", "h_dtl"]
d = d.dropna(subset=["minutes", "lag7", "lag4mean"])
train, test = d[d.index.year == 2024], d[d.index.year == 2025]
print("train rows", len(train), "test rows", len(test))

def fit_ridge(cols_cat, cols_num, log):
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True).fit(train[cols_cat])
    def X(df):
        import scipy.sparse as sp
        parts = [enc.transform(df[cols_cat])]
        if cols_num: parts.append(sp.csr_matrix(df[cols_num].values / 30.0))
        return sp.hstack(parts).tocsr()
    y = np.log(train.minutes) if log else train.minutes
    best = None
    for a in [0.3, 1, 3, 10, 30, 100]:
        errs = []
        for tr_i, va_i in TimeSeriesSplit(n_splits=4).split(train):
            m = Ridge(alpha=a).fit(X(train.iloc[tr_i]), y.iloc[tr_i]); p = m.predict(X(train.iloc[va_i]))
            p = np.exp(p) if log else p; errs.append(np.abs(p - train.minutes.iloc[va_i]).mean())
        if best is None or np.mean(errs) < best[0]: best = (np.mean(errs), a)
    m = Ridge(alpha=best[1]).fit(X(train), y); p = m.predict(X(test))
    return (np.exp(p) if log else p), best[1]

P = {"last week (lag 7d)": test.lag7.values,
     "mean per hour x weekday": test[["hour", "dow"]].merge(train.groupby(["hour", "dow"]).minutes.mean().rename("m").reset_index(), on=["hour", "dow"], how="left").m.values}
alphas = {}
for log in (False, True):
    tag = " (log)" if log else ""
    P["Ridge calendar" + tag], alphas["Ridge calendar" + tag] = fit_ridge(CAT, [], log)
    P["Ridge calendar+lags" + tag], alphas["Ridge calendar+lags" + tag] = fit_ridge(CAT, ["lag7", "lag4mean"], log)

y = test.minutes.values; dt = test.daytype.values
groups = {"all": np.ones(len(test), bool),
          "ordinary (weekday+sat+sun)": np.isin(dt, ["weekday", "sat", "sun"]),
          "holiday days (long weekend + eve)": np.isin(dt, ["lw_first", "lw_mid", "lw_last", "eve_of_long"]),
          "long weekend days only": np.isin(dt, ["lw_first", "lw_mid", "lw_last"]),
          **{k: dt == k for k in ["weekday", "sat", "sun", "eve_of_long", "lw_first", "lw_mid", "lw_last", "single_holiday", "makeup_workday"]}}
rows = []
for g, m in groups.items():
    if m.sum() == 0: continue
    r = {"slice": g, "n": int(m.sum())}
    for k, p in P.items(): r[k + " MAE"] = round(float(np.abs(p[m] - y[m]).mean()), 2)
    base = r["last week (lag 7d) MAE"]
    r["best Ridge vs last week %"] = round(100 * (1 - min(v for kk, v in r.items() if kk.startswith("Ridge")) / base), 1)
    rows.append(r)
res = pd.DataFrame(rows); res.to_csv(f"archive/results/go_nogo_results{SUF}.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(res.to_string(index=False)); print("alphas", alphas)
rm = pd.DataFrame({k: [float(np.sqrt(((p - y) ** 2).mean()))] for k, p in P.items()}, index=["RMSE all"]); print(rm.round(2).to_string())

# ---- day-level bootstrap: how sure are we about the improvement on a slice? (resample whole DAYS) ----
rng = np.random.default_rng(0); day_id = test.index.normalize().values
brow = []
for g in ["ordinary (weekday+sat+sun)", "holiday days (long weekend + eve)", "long weekend days only", "eve_of_long", "lw_first", "sat"]:
    m = groups[g]; days = np.unique(day_id[m])
    ae = {k: pd.Series(np.abs(p[m] - y[m])).groupby(day_id[m]).sum() for k, p in P.items()}
    cnt = pd.Series(1, index=range(m.sum())).groupby(day_id[m]).sum()
    for cand in ["Ridge calendar (log)", "Ridge calendar+lags (log)"]:
        for base in ["last week (lag 7d)", "mean per hour x weekday"]:
            imp = []
            for _ in range(2000):
                pick = rng.choice(days, len(days)); n = cnt.loc[pick].sum()
                imp.append(100 * (1 - ae[cand].loc[pick].sum() / ae[base].loc[pick].sum()))
            brow.append({"slice": g, "days": len(days), "candidate": cand, "baseline": base,
                         "improvement_%": round(100 * (1 - ae[cand].sum() / ae[base].sum()), 1),
                         "ci95_low": round(float(np.percentile(imp, 2.5)), 1), "ci95_high": round(float(np.percentile(imp, 97.5)), 1)})
bt = pd.DataFrame(brow); bt.to_csv(f"archive/results/go_nogo_bootstrap{SUF}.csv", index=False, encoding="utf-8-sig")
print(bt[bt.candidate == "Ridge calendar (log)"].to_string(index=False))


# ---- decision metric: leave in the slot the model recommends vs the truly fastest slot in the same window ----
# For each test day and each 4-hour window the driver can choose any slot in it. The recommended slot is the argmin of the
# predicted travel time (exact ties -> earliest). Regret = actual time at the recommended slot - actual minimum in the window.
# "leave at window start" is the no-model default. Windows with a missing slot are skipped for every model.
WINDOWS = [(6, 10), (10, 14), (14, 18), (18, 22)]
P["leave at window start"] = None
drows = []
test_idx = test.index; slot = test.hour.values
for date in np.unique(day_id):
    mday = day_id == date
    for (h0, h1) in WINDOWS:
        lo, hi = h0 * 60 // SLOT, h1 * 60 // SLOT
        m = mday & (slot >= lo) & (slot < hi)
        if m.sum() != hi - lo: continue
        act = y[m]; rec = {"oracle": 0.0, "leave at window start": act[0] - act.min()}
        for k, p in P.items():
            if p is None: continue
            pv = p[m] + 1e-6 * np.arange(m.sum()); rec[k] = act[int(np.argmin(pv))] - act.min()
        drows.append({"date": pd.Timestamp(date), "daytype": dt[m][0], "window": f"{h0:02d}-{h1:02d}", "range": act.max() - act.min(), **rec})
dd = pd.DataFrame(drows)
dgroups = {"ordinary (weekday+sat+sun)": ["weekday", "sat", "sun"], "holiday days (long weekend + eve)": ["lw_first", "lw_mid", "lw_last", "eve_of_long"],
           "long weekend days only": ["lw_first", "lw_mid", "lw_last"], "sat": ["sat"], "lw_first": ["lw_first"]}
models = ["leave at window start", "last week (lag 7d)", "mean per hour x weekday", "Ridge calendar", "Ridge calendar (log)", "Ridge calendar+lags (log)", "oracle"]
out, boot = [], []
for g, types in dgroups.items():
    for congested in (False, True):
        sub = dd[dd.daytype.isin(types)]
        if congested: sub = sub[sub["range"] >= 5]
        if sub.empty: continue
        r = {"group": g, "windows": "range >= 5 min" if congested else "all", "n_windows": len(sub), "n_days": sub.date.nunique(), "mean_range_min": round(sub["range"].mean(), 2)}
        for k in models: r[k] = round(sub[k].mean(), 2)
        out.append(r)
        dayg = sub.groupby("date")
        sums = {k: dayg[k].sum() for k in models}; cnts = dayg.size(); days = sums["oracle"].index.values
        for cand in ["Ridge calendar (log)"]:
            for base in ["leave at window start", "mean per hour x weekday", "last week (lag 7d)"]:
                diffs = []
                for _ in range(2000):
                    pick = rng.choice(days, len(days)); n = cnts.loc[pick].sum()
                    diffs.append((sums[base].loc[pick].sum() - sums[cand].loc[pick].sum()) / n)
                boot.append({"group": g, "windows": r["windows"], "candidate": cand, "baseline": base,
                             "minutes_saved_per_window": round((sums[base].sum() - sums[cand].sum()) / cnts.sum(), 2),
                             "ci95_low": round(float(np.percentile(diffs, 2.5)), 2), "ci95_high": round(float(np.percentile(diffs, 97.5)), 2)})
dec = pd.DataFrame(out); dec.to_csv(f"archive/results/decision_results{SUF}.csv", index=False, encoding="utf-8-sig")
bdec = pd.DataFrame(boot); bdec.to_csv(f"archive/results/decision_bootstrap{SUF}.csv", index=False, encoding="utf-8-sig")
print("\nDecision regret = minutes lost vs the fastest slot in the 4-hour window (mean per window)")
print(dec.to_string(index=False)); print(bdec.to_string(index=False))
