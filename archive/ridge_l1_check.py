"""Is the GBM L1 advantage over Ridge (log, L2) due to non-linearity or just to the loss? Fit the SAME linear design with an L1 loss
(median regression via iteratively reweighted ridge, 15 iterations) on minutes, and compare with GBM L1 and Ridge (log).
Train 2024, test 2025, eve modelled as 'shared' evening-bin effect. Output: ridge_l1_check.csv. Usage: python ridge_l1_check.py"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from forecast_common import *

def ridge_l1(train, test, cats, nums, iters=15):
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

d = load_data(60); train, test = d[d.index.year == 2024], d[d.index.year == 2025]
tr, te = add_cats(train, "shared"), add_cats(test, "shared"); P = {}
P["Ridge log (L2)"], _ = ridge_fit(tr, te, CATS, [], True)
P["Ridge L1"], a1 = ridge_l1(tr, te, CATS, [])
P["GBM log (L2)"], _ = gbm_fit(train, test, False, True, "regression")
P["GBM L1"], _ = gbm_fit(train, test, False, False, "regression_l1")
P["Ridge log (L2) +lags"], _ = ridge_fit(tr, te, CATS, ["lag7", "lag4mean"], True)
P["Ridge L1 +lags"], _ = ridge_l1(tr, te, CATS, ["lag7", "lag4mean"])
P["GBM L1 +lags"], _ = gbm_fit(train, test, True, False, "regression_l1")
y, dt = test.minutes.values, test.daytype.values
groups = {"all": np.ones(len(test), bool), "ordinary (weekday+sat+sun)": np.isin(dt, ["weekday", "sat", "sun"]),
          "holiday days (long weekend + eve)": np.isin(dt, ["lw_first", "lw_mid", "lw_last", "eve_of_long"]),
          "long weekend days only": np.isin(dt, ["lw_first", "lw_mid", "lw_last"]), "lw_first": dt == "lw_first", "sat": dt == "sat"}
res = pd.DataFrame([{"slice": g, "n": int(m.sum()), **{k: round(float(np.abs(p[m] - y[m]).mean()), 2) for k, p in P.items()}} for g, m in groups.items()])
res.to_csv("archive/results/ridge_l1_check.csv", index=False, encoding="utf-8-sig"); pd.set_option("display.width", 250); print(res.to_string(index=False))
