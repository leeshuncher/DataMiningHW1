"""Model-selection protocol. EVERYTHING that is chosen (loss, how the eve of a long weekend is handled, down-weighting of 2023, night handling,
last-year-holiday feature, alpha, top-k) is chosen on training years only: rolling-origin CV where the validation blocks are the four quarters of 2024
and each block is predicted from all earlier data (2023 and earlier 2024). The test year 2025 is touched once, in final_eval.py.
Pre-specified selection score = 0.5 * MAE on ordinary days + 0.5 * MAE on holiday days (long weekend + eve), pooled over the four validation quarters."""
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from forecast_common import load_data, add_cats, CATS, gbm_fit

ORD = ["weekday", "sat", "sun"]; HOL = ["lw_first", "lw_mid", "lw_last", "eve_of_long"]
QUARTERS = [("2024-01-01", "2024-04-01"), ("2024-04-01", "2024-07-01"), ("2024-07-01", "2024-10-01"), ("2024-10-01", "2025-01-01")]
ALPHAS_L2 = [0.3, 1, 3, 10, 30, 100]; ALPHAS_L1 = [0.3, 1, 3]; KS = [20, 50, 100, 200, 400]

def data():
    d = load_data(60); d["ly_f"] = d.ly.fillna(d.lag4mean); d["ly_ok"] = d.ly.notna().astype(float) * 30.0   # x30: numeric columns are divided by 30
    return d

def drift_weights(df, w):
    """2023 ORDINARY days get weight w, everything else 1."""
    out = np.ones(len(df)); out[(df.index.year == 2023) & df.daytype.isin(ORD).values] = w; return out

SETS = {"C": CATS, "A": ["c_slot", "c_dow", "c_lw"],
        "C-interactions": [c for c in CATS if c not in ("c_s_dtype", "c_s_dtl")],
        "C-holiday structure": [c for c in CATS if c not in ("c_dtype", "c_dtl", "c_s_dtype", "c_s_dtl")],
        "C-weekly rhythm (slot x weekday)": [c for c in CATS if c != "c_s_dow"], "C-eve effect": [c for c in CATS if c != "c_eve"]}

def _topk(X, yc, k):
    n = X.shape[0]; mean = np.asarray(X.mean(axis=0)).ravel(); sd = np.sqrt(np.maximum(mean * (1 - mean), 0))
    cov = np.asarray(X.T @ yc).ravel() / n - mean * yc.mean() * 0   # yc is centred already
    corr = np.abs(cov) / (sd * yc.std() + 1e-12); corr[sd == 0] = 0
    return np.argsort(-corr)[:k]

def _fit_predict(Xtr, ytr, Xte, w, loss, alpha, iters=8):
    m = Ridge(alpha=alpha)
    if loss == "log": return m.fit(Xtr, ytr, sample_weight=w).predict(Xte)
    ww = w.copy()
    for _ in range(iters):
        m.fit(Xtr, ytr, sample_weight=ww); ww = w / np.maximum(np.abs(ytr - m.predict(Xtr)), 0.25)
    return m.predict(Xte)

def ridge_cfg(train, test, cats, nums, loss, weights, topk=False):
    """Ridge on log(minutes) (loss='log') or median regression on minutes (loss='l1'); alpha (and k when topk) by TimeSeriesSplit inside `train`."""
    enc = OneHotEncoder(handle_unknown="ignore").fit(train[cats])
    def X(df): return sp.hstack([enc.transform(df[cats])] + ([sp.csr_matrix(df[nums].values / 30.0)] if nums else [])).tocsr()
    Xtr, Xte = X(train), X(test); y = np.log(train.minutes.values) if loss == "log" else train.minutes.values
    back = (lambda p: np.exp(p)) if loss == "log" else (lambda p: p); w = np.asarray(weights, float)
    alphas = ALPHAS_L2 if loss == "log" else ALPHAS_L1; ks = KS if topk else [None]; best = None
    for k in ks:
        for a in alphas:
            errs = []
            for tr_i, va_i in TimeSeriesSplit(n_splits=4).split(Xtr):
                cols = slice(None) if k is None else _topk(Xtr[tr_i], y[tr_i] - y[tr_i].mean(), k)
                p = back(_fit_predict(Xtr[tr_i][:, cols], y[tr_i], Xtr[va_i][:, cols], w[tr_i], loss, a))
                errs.append(np.abs(p - train.minutes.values[va_i]).mean())
            if best is None or np.mean(errs) < best[0]: best = (np.mean(errs), a, k)
    _, a, k = best; cols = slice(None) if k is None else _topk(Xtr, y - y.mean(), k)
    return back(_fit_predict(Xtr[:, cols], y, Xte[:, cols], w, loss, a)), {"alpha": a, "k": k}

def predict_cfg(train, test, cfg, feature_set="C"):
    """cfg = dict(loss, eve, ly, w, night). Returns predictions for test."""
    tr, te = add_cats(train, cfg["eve"], night=cfg["night"]), add_cats(test, cfg["eve"], night=cfg["night"])
    cats = SETS[feature_set if feature_set in SETS else "C"]; nums = ["ly_f", "ly_ok"] if (cfg["ly"] and feature_set != "C-ly" and feature_set != "A") else []
    return ridge_cfg(tr, te, cats, nums, cfg["loss"], drift_weights(train, cfg["w"]), topk=(feature_set == "B"))[0] if feature_set != "B" else \
        ridge_cfg(tr, te, SETS["C"], nums, cfg["loss"], drift_weights(train, cfg["w"]), topk=True)[0]

def fold_split(d, start, end):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return d[(d.index < s)], d[(d.index >= s) & (d.index < e)]
