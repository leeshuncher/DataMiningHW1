"""Shared helpers for the Freeway 5 experiments: data loading with calendar features, Ridge / LightGBM fitting with
TimeSeriesSplit tuning inside the training years, slice tables and the departure-slot decision metric."""
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

def load_data(slot=60):
    per_day = 1440 // slot
    t = pd.read_csv(f"data/travel_{slot}min.csv", parse_dates=["depart"]).set_index("depart")
    cal = pd.read_csv("data/calendar.csv", parse_dates=["date"])
    nxt = cal.assign(date=cal.date - pd.Timedelta(days=1))[["date", "block_len"]].rename(columns={"block_len": "next_len"})
    cal = cal.merge(nxt, on="date", how="left")
    cal["eve_len"] = np.where(cal.daytype == "eve_of_long", cal.next_len.fillna(0), 0)
    d = t[["minutes"]].copy(); d["date"] = d.index.normalize()
    d = d.merge(cal[["date", "daytype", "dow", "block_len", "block_pos", "is_makeup_workday", "eve_len"]], on="date", how="left").set_index(t.index)
    d["slot"] = (d.index.hour * 60 + d.index.minute) // slot
    d["hod"] = d.slot * slot / 60.0
    s = d.minutes
    d["lag7"] = s.shift(7 * per_day)
    d["lag4mean"] = pd.concat([s.shift(k * 7 * per_day) for k in (1, 2, 3, 4)], axis=1).mean(axis=1, skipna=False)
    return d.dropna(subset=["minutes", "lag7", "lag4mean"])

CATS = ["c_slot", "c_dow", "c_dtype", "c_dtl", "c_s_dtype", "c_s_dow", "c_s_dtl", "c_eve"]

def add_cats(d, eve_mode):
    """eve_mode: separate = 'eve_of_long' is its own day type (old approach, hour x type cells);
    none = a workday before a long weekend is treated as an ordinary weekday;
    shared = none + one effect per time-of-day bin for ALL eves; by_dow = none + one effect per bin, separately for Friday and other weekdays."""
    d = d.copy(); eve = d.daytype.eq("eve_of_long")
    dtype = d.daytype if eve_mode == "separate" else d.daytype.where(~eve, "weekday")
    long_ = dtype.str.startswith("lw_")
    len_b = np.where(long_, np.where(d.block_len >= 5, "5+", d.block_len.astype(int).astype(str)), "-")
    sl, dw = d.slot.astype(str), d.dow.astype(str)
    d["c_slot"], d["c_dow"], d["c_dtype"] = sl, dw, dtype
    d["c_dtl"] = dtype + "_" + len_b
    d["c_s_dtype"], d["c_s_dow"], d["c_s_dtl"] = sl + "|" + dtype, sl + "|" + dw, sl + "|" + d.c_dtl
    bins = pd.cut(d.hod, [-1, 11.99, 15.99, 19.99, 24], labels=["am", "pm", "eve1", "eve2"]).astype(str)
    if eve_mode == "shared": d["c_eve"] = np.where(eve, "eve|" + bins, "no")
    elif eve_mode == "by_dow": d["c_eve"] = np.where(eve, "eve|" + np.where(d.dow == 4, "fri", "nonfri") + "|" + bins, "no")
    else: d["c_eve"] = "no"
    return d

def ridge_fit(train, test, cats, nums, log):
    enc = OneHotEncoder(handle_unknown="ignore").fit(train[cats])
    X = lambda df: sp.hstack([enc.transform(df[cats])] + ([sp.csr_matrix(df[nums].values / 30.0)] if nums else [])).tocsr()
    y = np.log(train.minutes) if log else train.minutes
    best = None
    for a in [0.3, 1, 3, 10, 30, 100]:
        errs = []
        for tr_i, va_i in TimeSeriesSplit(n_splits=4).split(train):
            p = Ridge(alpha=a).fit(X(train.iloc[tr_i]), y.iloc[tr_i]).predict(X(train.iloc[va_i]))
            errs.append(np.abs((np.exp(p) if log else p) - train.minutes.iloc[va_i]).mean())
        if best is None or np.mean(errs) < best[0]: best = (np.mean(errs), a)
    p = Ridge(alpha=best[1]).fit(X(train), y).predict(X(test))
    return (np.exp(p) if log else p), best[1]

GBM_CAL = ["slot", "dow", "dtype_code", "block_len", "block_pos", "eve", "eve_len", "makeup"]
def gbm_frame(d, codes):
    f = pd.DataFrame({"slot": d.slot.values, "dow": d.dow.values, "dtype_code": pd.Categorical(d.daytype.values, categories=codes),
                      "block_len": d.block_len.values, "block_pos": d.block_pos.values, "eve": (d.daytype == "eve_of_long").astype(int).values,
                      "eve_len": d.eve_len.values, "makeup": d.is_makeup_workday.astype(int).values}, index=d.index)
    f["lag7"], f["lag4mean"] = d.lag7.values, d.lag4mean.values
    return f

def gbm_fit(train, test, lags, log, objective):
    codes = sorted(set(train.daytype) | set(test.daytype)); cols = GBM_CAL + (["lag7", "lag4mean"] if lags else [])
    Xtr, Xte = gbm_frame(train, codes)[cols], gbm_frame(test, codes)[cols]
    y = np.log(train.minutes.values) if log else train.minutes.values
    best = None
    for leaves in (4, 8, 16):
        for n in (100, 300):
            for mcs in (10, 40):
                errs = []
                for tr_i, va_i in TimeSeriesSplit(n_splits=4).split(Xtr):
                    m = lgb.LGBMRegressor(objective=objective, n_estimators=n, learning_rate=0.05, num_leaves=leaves, min_child_samples=mcs,
                                          subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1, random_state=0, n_jobs=4)
                    p = m.fit(Xtr.iloc[tr_i], y[tr_i]).predict(Xtr.iloc[va_i]); p = np.exp(p) if log else p
                    errs.append(np.abs(p - train.minutes.values[va_i]).mean())
                if best is None or np.mean(errs) < best[0]: best = (np.mean(errs), dict(num_leaves=leaves, n_estimators=n, min_child_samples=mcs))
    m = lgb.LGBMRegressor(objective=objective, learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1,
                          random_state=0, n_jobs=4, **best[1]).fit(Xtr, y)
    p = m.predict(Xte)
    return (np.exp(p) if log else p), best[1]

def decision_regret(test, P, slot, windows=((6, 10), (10, 14), (14, 18), (18, 22))):
    """Per test day and 4-hour window: minutes lost by leaving in the slot with the lowest predicted time vs the truly fastest slot."""
    y = test.minutes.values; day = test.index.normalize().values; sl = test.slot.values; dt = test.daytype.values; rows = []
    for date in np.unique(day):
        md = day == date
        for h0, h1 in windows:
            lo, hi = h0 * 60 // slot, h1 * 60 // slot
            m = md & (sl >= lo) & (sl < hi)
            if m.sum() != hi - lo: continue
            act = y[m]; r = {"date": pd.Timestamp(date), "daytype": dt[m][0], "window": f"{h0:02d}-{h1:02d}", "range": act.max() - act.min(),
                             "leave at window start": act[0] - act.min()}
            for k, p in P.items(): r[k] = act[int(np.argmin(p[m] + 1e-6 * np.arange(m.sum())))] - act.min()
            rows.append(r)
    return pd.DataFrame(rows)


def ridge_l1(train, test, cats, nums, iters=15):
    """Median (L1) regression with the same design: iteratively reweighted ridge on minutes."""
    enc = OneHotEncoder(handle_unknown="ignore").fit(train[cats])
    X = lambda df: sp.hstack([enc.transform(df[cats])] + ([sp.csr_matrix(df[nums].values / 30.0)] if nums else [])).tocsr()
    def fit(Xa, ya, a):
        w = np.ones(len(ya)); m = Ridge(alpha=a)
        for _ in range(iters):
            m.fit(Xa, ya, sample_weight=w); w = 1.0 / np.maximum(np.abs(ya - m.predict(Xa)), 0.25)
        return m
    best = None
    for a in [0.1, 0.3, 1, 3, 10]:
        errs = []
        for tr_i, va_i in TimeSeriesSplit(n_splits=4).split(train):
            m = fit(X(train.iloc[tr_i]), train.minutes.values[tr_i], a)
            errs.append(np.abs(m.predict(X(train.iloc[va_i])) - train.minutes.values[va_i]).mean())
        if best is None or np.mean(errs) < best[0]: best = (np.mean(errs), a)
    return fit(X(train), train.minutes.values, best[1]).predict(X(test)), best[1]

def ridge_log_residuals(train, test, cats, nums):
    """Ridge on log(minutes). Returns test log-prediction, alpha, in-sample log residuals and out-of-fold (TimeSeriesSplit) log residuals
    (NaN for the first fold) of the training rows, used for smearing corrections."""
    enc = OneHotEncoder(handle_unknown="ignore").fit(train[cats])
    X = lambda df: sp.hstack([enc.transform(df[cats])] + ([sp.csr_matrix(df[nums].values / 30.0)] if nums else [])).tocsr()
    y = np.log(train.minutes.values); best = None
    for a in [0.3, 1, 3, 10, 30, 100]:
        errs = []
        for tr_i, va_i in TimeSeriesSplit(n_splits=4).split(train):
            p = Ridge(alpha=a).fit(X(train.iloc[tr_i]), y[tr_i]).predict(X(train.iloc[va_i]))
            errs.append(np.abs(np.exp(p) - train.minutes.values[va_i]).mean())
        if best is None or np.mean(errs) < best[0]: best = (np.mean(errs), a)
    a = best[1]; m = Ridge(alpha=a).fit(X(train), y)
    ins = y - m.predict(X(train)); oof = np.full(len(y), np.nan)
    for tr_i, va_i in TimeSeriesSplit(n_splits=4).split(train):
        oof[va_i] = y[va_i] - Ridge(alpha=a).fit(X(train.iloc[tr_i]), y[tr_i]).predict(X(train.iloc[va_i]))
    return m.predict(X(test)), a, ins, oof
