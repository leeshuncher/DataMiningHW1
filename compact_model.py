"""Compact, fully reportable linear model: log or median regression on ~50-90 indicator terms (time-of-day blocks x day types, Friday terms, long-holiday term).
Compared with the large one-hot Ridge using the SAME protocol (rolling-origin CV on the four 2024 quarters; test 2025 used once).
Pre-specified rule: the compact model becomes the final model if its CV score (0.5*MAE ordinary + 0.5*MAE holiday) is within 5% of the best large model.
Outputs compact_cv.csv, compact_test.csv. Usage: python compact_model.py"""
import json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from protocol import data, drift_weights, fold_split, QUARTERS, ORD, HOL

def compact_X(d, edges, labels, hours=False):
    blk = pd.cut(d.hod, edges, labels=labels, include_lowest=True, ordered=False).astype(str); X = pd.DataFrame(index=d.index)
    inner = [l for l in labels if l != "night"]
    if hours:   # hourly main effects replace the block main effects (the day-type x block interactions stay)
        hr = d.slot.astype(int)
        for h in range(1, 24): X[f"hr_{h}"] = (hr == h).astype(float)
    else:
        for b in inner: X["blk_" + b] = (blk == b).astype(float)
    fri = (d.daytype == "weekday") & (d.dow == 4)
    for b in inner[len(inner) // 2:]: X["fri_" + b] = (fri & (blk == b)).astype(float)
    types = {"sat": d.daytype == "sat", "sun": d.daytype == "sun", "lwf": d.daytype == "lw_first", "lwm": d.daytype == "lw_mid", "lwl": d.daytype == "lw_last",
             "eveF": (d.daytype == "eve_of_long") & (d.dow == 4), "eveO": (d.daytype == "eve_of_long") & (d.dow != 4), "single": d.daytype == "single_holiday", "makeup": d.daytype == "makeup_workday"}
    for k, m in types.items():
        X[k] = m.astype(float)
        for b in inner: X[f"{k}_{b}"] = (m & (blk == b)).astype(float)
    pm = [l for l in inner if l in ("pm", "13-16", "14-16", "16-18", "12-14", "pm1", "pm2")]
    X["lwf_pm_long4"] = ((d.daytype == "lw_first") & (d.block_len >= 4) & blk.isin(pm)).astype(float)
    return X

BLOCKS = {"compact5": ([0, 5.99, 9.99, 13.99, 17.99, 21.99, 24], ["night", "am", "mid", "pm", "ev", "night"]),
          "compact8": ([0, 5.99, 7.99, 9.99, 11.99, 13.99, 15.99, 17.99, 19.99, 21.99, 24], ["night", "6-8", "8-10", "10-12", "12-14", "14-16", "16-18", "18-20", "20-22", "night"])}
BLOCKS["compact8h"] = BLOCKS["compact8"]
def blocks(name):
    e, l = BLOCKS[name]; return e, l

def fit_predict(train, test, name, loss, w2023, ly=False, drop=(), keep_cols=None, topk=None, custom_X=None):
    e, l = blocks(name)
    def mk(df):
        X = custom_X(df) if custom_X is not None else compact_X(df, e, l, hours=name.endswith("h"))
        if ly: X["ly_f"] = df.ly.fillna(df.lag4mean).values / 30.0; X["ly_ok"] = df.ly.notna().astype(float).values
        return X.drop(columns=[c for c in X.columns if any(c.startswith(p) or c == p for p in drop)])
    Xtr_df, Xte_df = mk(train), mk(test); keep = Xtr_df.columns[(Xtr_df != 0).any()]
    if keep_cols is not None: keep = [c for c in keep if c in keep_cols]
    Xtr, Xte = Xtr_df[keep].values, Xte_df.reindex(columns=keep, fill_value=0).values
    y = np.log(train.minutes.values) if loss == "log" else train.minutes.values; w = drift_weights(train, w2023); back = (lambda p: np.exp(p)) if loss == "log" else (lambda p: p)
    def fit(Xa, ya, wa, a, iters=8):
        m = Ridge(alpha=a)
        if loss == "log": return m.fit(Xa, ya, sample_weight=wa)
        ww = wa.copy()
        for _ in range(iters): m.fit(Xa, ya, sample_weight=ww); ww = wa / np.maximum(np.abs(ya - m.predict(Xa)), 0.25)
        return m
    best = None
    for a in [0.01, 0.1, 1, 3, 10]:
        errs = []
        for tr_i, va_i in TimeSeriesSplit(n_splits=4).split(Xtr):
            p = back(fit(Xtr[tr_i], y[tr_i], w[tr_i], a).predict(Xtr[va_i])); errs.append(np.abs(p - train.minutes.values[va_i]).mean())
        if best is None or np.mean(errs) < best[0]: best = (np.mean(errs), a)
    return back(fit(Xtr, y, w, best[1]).predict(Xte)), best[1], list(keep)

if __name__ == "__main__":
    d = data(); rows = []
    for name in BLOCKS:
        for loss in ["log", "l1"]:
            for w2023 in [1.0, 0.1]:
                eo, eh, ea, folds = [], [], [], []
                for s, e in QUARTERS:
                    tr, va = fold_split(d, s, e); p, a, _ = fit_predict(tr, va, name, loss, w2023); ae = np.abs(p - va.minutes.values)
                    eo += list(ae[va.daytype.isin(ORD).values]); eh += list(ae[va.daytype.isin(HOL).values]); ea += list(ae); folds.append(ae.mean())
                rows.append({"model": name, "loss": loss, "w2023": w2023, "mae_ordinary": np.mean(eo), "mae_holiday": np.mean(eh), "mae_all": np.mean(ea), "fold_mean": np.mean(folds), "fold_sd": np.std(folds, ddof=1), "score": 0.5 * np.mean(eo) + 0.5 * np.mean(eh)})
    cvt = pd.DataFrame(rows).sort_values("score"); cvt.to_csv("compact_cv.csv", index=False, encoding="utf-8-sig"); pd.set_option("display.width", 200); print(cvt.round(3).to_string(index=False))
    best_large = pd.read_csv("cv_grid.csv").score.min(); print("best large-model CV score:", round(best_large, 3), "| 5% rule threshold:", round(best_large * 1.05, 3))
