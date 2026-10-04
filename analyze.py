"""Slice analysis + decision-level metrics on data/modeling_table{TAG}.parquet.
Models for dHR = HR(t+60) - HR(t): persistence (0), linear extrapolation (c * hr_slope60 * 60), Ridge on raw columns only,
Ridge on all engineered features, LightGBM on all (reference ceiling, NOT the final model).
Outputs: slice_results{TAG}.csv, decision_results{TAG}.csv. Test set is used once for reporting; thresholds (margins) are
chosen on an inner validation split of TRAIN workouts (workout_id % 4 == 0)."""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

TAG = os.environ.get("TABLE_TAG", "")
d = pd.read_parquet(f"data/modeling_table{TAG}.parquet")
RAW = ["hr", "speed", "grade", "elapsed_s"]
ENG = ["hr_slope30", "hr_slope60", "dspeed30", "dspeed60", "speed_ema", "dgrade30", "dgrade60", "speed_x_grade", "elapsed_x_speed", "hr_gap"]
ALL = RAW + ENG
MODELS = ["persistence", "extrapolation", "ridge_raw", "ridge_all", "lgbm_all"]

def fit_predict(tr, ev, seed=0):
    out = {"persistence": np.zeros(len(ev))}
    x, y = tr.hr_slope60 * 60, tr.dhr
    c = float((x * y).sum() / (x * x).sum()); out["extrapolation"] = c * ev.hr_slope60.values * 60
    for name, cols in [("ridge_raw", RAW), ("ridge_all", ALL)]:
        sc = StandardScaler().fit(tr[cols]); m = Ridge(alpha=10).fit(sc.transform(tr[cols]), tr.dhr)
        out[name] = m.predict(sc.transform(ev[cols]))
    sub = tr.sample(min(len(tr), 1_200_000), random_state=seed)
    g = lgb.LGBMRegressor(objective="regression_l1", n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=100,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=seed)
    out["lgbm_all"] = g.fit(sub[ALL], sub.dhr).predict(ev[ALL])
    return out

train = d[d.split == "train"]; test = d[d.split == "test"].reset_index(drop=True)
inner_val_mask = (train.workout_id % 4 == 0)
inner_tr, inner_ev = train[~inner_val_mask].reset_index(drop=True), train[inner_val_mask].reset_index(drop=True)
P_test = fit_predict(train, test)
P_inner = fit_predict(inner_tr, inner_ev)

# ---------------- 1. slices ----------------
mae = lambda e: float(np.abs(e).mean())
a, ag = test.dspeed60.abs(), test.grade
slices = {
    "all": np.ones(len(test), bool),
    "|dspeed60| < 0.1": a < 0.1, "0.1-0.3": (a >= 0.1) & (a < 0.3), "0.3-0.5": (a >= 0.3) & (a < 0.5),
    "0.5-1.0": (a >= 0.5) & (a < 1.0), ">= 1.0": a >= 1.0,
    "accelerating (dspeed60 > +0.5)": test.dspeed60 > 0.5, "decelerating (dspeed60 < -0.5)": test.dspeed60 < -0.5,
    "|dgrade60| > 0.05": test.dgrade60.abs() > 0.05,
    "uphill (grade > 0.05)": ag > 0.05, "downhill (grade < -0.05)": ag < -0.05, "flat (|grade| < 0.02)": ag.abs() < 0.02,
    "transition (|dspeed60|>0.5 or |dgrade60|>0.05)": (a > 0.5) | (test.dgrade60.abs() > 0.05),
    "steady (|dspeed60|<0.1 and |dgrade60|<0.02)": (a < 0.1) & (test.dgrade60.abs() < 0.02),
    "HR rising fast (hr_slope60 > 0.1)": test.hr_slope60 > 0.1,
}
rows = []
for name, m in slices.items():
    m = np.asarray(m); e = {k: mae(test.dhr.values[m] - P_test[k][m]) for k in MODELS}
    rows.append({"slice": name, "n": int(m.sum()), "share_%": round(100 * m.mean(), 1), "std_dhr": round(float(test.dhr[m].std()), 2),
                 **{k + "_MAE": round(v, 2) for k, v in e.items()},
                 "ridge_all_vs_persist_%": round(100 * (1 - e["ridge_all"] / e["persistence"]), 1)})
sl = pd.DataFrame(rows); sl.to_csv(f"slice_results{TAG}.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(sl.to_string(index=False))

# ---------------- 3. decision level ----------------
def zone(lo_q, hi_q):
    z = train.groupby("userId").hr.quantile([lo_q, hi_q]).unstack(); z.columns = ["lo", "hi"]; return z
def prf(y, pred):
    tp = int((y & pred).sum()); fp = int((~y & pred).sum()); fn = int((y & ~pred).sum())
    p = tp / (tp + fp) if tp + fp else float("nan"); r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if tp else 0.0; return p, r, f
GRID = np.arange(-15, 15.01, 0.5)
rows, cls_rows = [], []
for lo_q, hi_q in [(0.25, 0.75), (0.35, 0.65), (0.15, 0.85)]:
    z = zone(lo_q, hi_q); zinfo = f"q{int(lo_q*100)}-q{int(hi_q*100)}"
    width = float((z.hi - z.lo).median())
    def prep(df):
        zz = z.reindex(df.userId.values); lo, hi = zz.lo.values, zz.hi.values
        inz = (df.hr.values >= lo) & (df.hr.values <= hi)
        return lo, hi, inz, df.hr_future.values > hi, df.hr_future.values < lo
    lo_i, hi_i, inz_i, up_i, dn_i = prep(inner_ev); lo_t, hi_t, inz_t, up_t, dn_t = prep(test)
    for kind in ["up", "down"]:
        for mname in MODELS:
            def score(df, P, lo, hi):
                fut = df.hr.values + P[mname]
                return (fut - hi) if kind == "up" else (lo - fut)
            s_i = score(inner_ev, P_inner, lo_i, hi_i); s_t = score(test, P_test, lo_t, hi_t)
            y_i = (up_i if kind == "up" else dn_i)[inz_i]; y_t = (up_t if kind == "up" else dn_t)[inz_t]
            best = max(GRID, key=lambda dl: prf(y_i, s_i[inz_i] > -dl)[2])
            for variant, dl in ([("strict (margin 0)", 0.0)] if mname == "persistence" else []) + [("tuned margin", best)]:
                p, r, f = prf(y_t, s_t[inz_t] > -dl)
                try: auc = roc_auc_score(y_t, s_t[inz_t])
                except ValueError: auc = float("nan")
                rows.append({"zone": zinfo, "zone_width_median_BPM": round(width, 1), "event": kind + "-exit", "model": mname, "threshold": variant,
                             "margin_BPM": dl, "n_in_zone": int(inz_t.sum()), "base_rate_%": round(100 * y_t.mean(), 1),
                             "precision": round(p, 3), "recall": round(r, 3), "F1": round(f, 3), "AUC": round(auc, 3)})
    # 3-class decision accuracy over all test rows (reduce / hold / increase), margin 0
    truth = np.where(test.hr_future.values > hi_t, 2, np.where(test.hr_future.values < lo_t, 0, 1))
    for mname in MODELS:
        fut = test.hr.values + P_test[mname]
        pred = np.where(fut > hi_t, 2, np.where(fut < lo_t, 0, 1))
        cls_rows.append({"zone": zinfo, "model": mname, "3class_accuracy": round(float((pred == truth).mean()), 4),
                         "accuracy_on_changed_state": round(float((pred == truth)[truth != np.where(test.hr.values > hi_t, 2, np.where(test.hr.values < lo_t, 0, 1))].mean()), 4)})
dec = pd.DataFrame(rows); dec.to_csv(f"decision_results{TAG}.csv", index=False, encoding="utf-8-sig")
cls = pd.DataFrame(cls_rows); cls.to_csv(f"decision_3class{TAG}.csv", index=False, encoding="utf-8-sig")
print(dec[dec.zone == "q25-q75"].to_string(index=False)); print(cls.to_string(index=False))
