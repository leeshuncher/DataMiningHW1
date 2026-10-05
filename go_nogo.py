"""Go / no-go check (proposal section 'Go / no-go check by Wed 10/7').
Train 2024, test 2025 (strictly by time; alpha chosen with TimeSeriesSplit inside 2024). Hourly (60 min) slots.
Models: same slot last week (lag 7 d), train mean per hour x weekday, Ridge on calendar/holiday features only,
Ridge on calendar + lags (same slot last week, mean of the last 4 same weekdays), each on minutes and on log(minutes).
Rows with missing target or missing lag are dropped for every model, so all models are scored on the same rows.
Go rule: MAE on holiday days >= 20% lower than same-slot-last-week, and ordinary days not worse.
Output: go_nogo_results.csv. Usage: python go_nogo.py"""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit

t = pd.read_csv("data/travel_60min.csv", parse_dates=["depart"]).set_index("depart")
cal = pd.read_csv("data/calendar.csv", parse_dates=["date"])
d = t[["minutes"]].copy(); d["date"] = d.index.normalize(); d["hour"] = d.index.hour
d = d.merge(cal[["date", "daytype", "dow", "block_len"]], left_on="date", right_on="date", how="left").set_index(t.index)
s = d.minutes
d["lag7"] = s.shift(7 * 24)
d["lag4mean"] = pd.concat([s.shift(k * 7 * 24) for k in (1, 2, 3, 4)], axis=1).mean(axis=1, skipna=False)
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
res = pd.DataFrame(rows); res.to_csv("go_nogo_results.csv", index=False, encoding="utf-8-sig")
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
bt = pd.DataFrame(brow); bt.to_csv("go_nogo_bootstrap.csv", index=False, encoding="utf-8-sig")
print(bt[bt.candidate == "Ridge calendar (log)"].to_string(index=False))
