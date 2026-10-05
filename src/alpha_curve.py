"""Tuning curve of the ridge penalty alpha for the final compact model: the same inner 4-fold TimeSeriesSplit on 2023-2024 that
compact_model.fit_predict uses to pick alpha, with every alpha's validation MAE recorded. Usage: python src/alpha_curve.py"""
import numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from protocol import data, drift_weights
import compact_model as cm

NAME, LOSS, W23 = "compact8h", "l1", 0.1
d = data(); train = d[d.index.year <= 2024]
e, l = cm.blocks(NAME); X = cm.compact_X(train, e, l, hours=True); X = X[X.columns[(X != 0).any()]].values
y = train.minutes.values; w = drift_weights(train, W23)
def fit(Xa, ya, wa, a, iters=8):
    m = Ridge(alpha=a); ww = wa.copy()
    for _ in range(iters): m.fit(Xa, ya, sample_weight=ww); ww = wa / np.maximum(np.abs(ya - m.predict(Xa)), 0.25)
    return m
rows = []
for a in [0.001, 0.01, 0.1, 1, 3, 10, 30, 100]:
    errs = [np.abs(fit(X[tr], y[tr], w[tr], a).predict(X[va]) - y[va]).mean() for tr, va in TimeSeriesSplit(n_splits=4).split(X)]
    rows.append({"alpha": a, "in_selection_grid": a in [0.01, 0.1, 1, 3, 10], "cv_mae": np.mean(errs), "cv_mae_sd": np.std(errs, ddof=1)})
out = pd.DataFrame(rows); out.to_csv("results/final2_alpha_curve.csv", index=False, encoding="utf-8-sig"); print(out.round(4).to_string(index=False))
