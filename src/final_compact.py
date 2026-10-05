"""FINAL evaluation of the compact linear model chosen by the pre-specified rule (compact_model.py): 8 two-hour blocks x day types, median (L1) regression,
2023 ordinary days weight 0.1. Train 2023-2024, test 2025 used once. All methods share data, folds and weights.
Outputs final2_*.csv. Usage: python src/final_compact.py"""
import json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from protocol import *
from forecast_common import decision_regret, gbm_fit
import compact_model as cm

NAME, LOSS, W23 = "compact8h", "l1", 0.1; FINAL = "Ours: compact linear (median regression)"
d = data(); train, test = d[d.index.year <= 2024], d[d.index.year == 2025]; W = drift_weights(train, W23); y = test.minutes.values
cfg_large = json.load(open("results/cv_selected.json"))

def wmean(tr, te, w):
    g = tr.assign(w=w, wy=w * tr.minutes.values).groupby(["slot", "dow"])[["w", "wy"]].sum(); m = (g.wy / g.w).rename("m").reset_index()
    return te[["slot", "dow"]].merge(m, on=["slot", "dow"], how="left").m.values
def sols(tr, te, w):
    A = np.c_[np.ones(len(tr)), tr.lag7.values] * np.sqrt(w)[:, None]; b = np.linalg.lstsq(A, tr.minutes.values * np.sqrt(w), rcond=None)[0]; return b[0] + b[1] * te.lag7.values
def baselines(tr, te, w):
    return {"Trivial: training mean": np.full(len(te), np.average(tr.minutes.values, weights=w)), "Persistence-type: last week same slot": te.lag7.values,
            "Same holiday last time (else last week)": np.where(te.ly.notna(), te.ly, te.lag7), "Seasonal mean: slot x weekday": wmean(tr, te, w), "Simple: OLS on last week only": sols(tr, te, w)}
def models(tr, te, w, with_large=True):
    P = baselines(tr, te, w); P[FINAL] = cm.fit_predict(tr, te, NAME, LOSS, W23)[0]
    P["Strong: GBM (L1)"], _ = gbm_fit(tr, te, False, False, "regression_l1", weights=w); P["Strong: GBM (log)"], _ = gbm_fit(tr, te, False, True, "regression", weights=w)
    if with_large: P["Flexible linear: large one-hot Ridge"] = predict_cfg(tr, te, cfg_large)
    return P

# ---------------- test predictions ----------------
P = models(train, test, W); P["Exploratory: ours + last-year feature"] = cm.fit_predict(train, test, NAME, LOSS, W23, ly=True)[0]
pd.DataFrame({"depart": test.index, "daytype": test.daytype.values, "actual": y, **P}).to_csv("results/final2_predictions.csv", index=False, encoding="utf-8-sig")
dt, hod, dow = test.daytype.values, test.hod.values, test.dow.values; lw = np.isin(dt, ["lw_first", "lw_mid", "lw_last"]); eve = dt == "eve_of_long"; has_ly = test.ly.notna().values
groups = {"all slots": np.ones(len(test), bool), "ordinary (weekday+sat+sun)": np.isin(dt, ORD), "Saturday 08-17h": (dt == "sat") & (hod >= 8) & (hod < 17), "holiday days (long weekend + eve)": lw | eve,
          "long weekend, all hours": lw, "long weekend 08-18h": lw & (hod >= 8) & (hod < 18), "long weekend morning 07-12h": lw & (hod >= 7) & (hod < 12), "lw_first 10-18h": (dt == "lw_first") & (hod >= 10) & (hod < 18),
          "eve 16-23h": eve & (hod >= 16), "long weekend night 0-6h + 20-24h": lw & ((hod < 6) | (hod >= 20)), "holiday days WITH last-year value": (lw | eve) & has_ly, "long weekend 08-18h WITH last-year value": lw & (hod >= 8) & (hod < 18) & has_ly}
rows = []
for g, m in groups.items():
    r = {"slice": g, "n": int(m.sum()), "days": int(pd.Series(test.index.normalize()[m]).nunique()), "mean actual": round(float(y[m].mean()), 1)}
    for k, p in P.items(): r[k] = round(float(np.abs(p[m] - y[m]).mean()), 2)
    rows.append(r)
res = pd.DataFrame(rows); res.to_csv("results/final2_mae.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 330); pd.set_option("display.max_columns", 40)
print("TEST MAE"); print(res.to_string(index=False))

# ---------------- main table: CV mean +- sd (RMSE too) + test MAE/RMSE/R2 ----------------
cvr = []
for s, e in QUARTERS:
    tr, va = fold_split(d, s, e); w = drift_weights(tr, W23); Pm = models(tr, va, w); yv = va.minutes.values; h = va.daytype.isin(HOL).values
    for k, p in Pm.items(): cvr.append({"fold": s, "model": k, "mae": np.abs(p - yv).mean(), "rmse": np.sqrt(((p - yv) ** 2).mean()), "mae_holiday": np.abs(p[h] - yv[h]).mean() if h.any() else np.nan})
cv = pd.DataFrame(cvr); cv.to_csv("results/final2_cv_folds.csv", index=False, encoding="utf-8-sig")
r2 = lambda p: 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
tab = cv.groupby("model").agg(cv_mae_mean=("mae", "mean"), cv_mae_sd=("mae", lambda s: s.std(ddof=1)), cv_rmse_mean=("rmse", "mean"), cv_rmse_sd=("rmse", lambda s: s.std(ddof=1)), cv_hol_mae_mean=("mae_holiday", "mean"), cv_hol_mae_sd=("mae_holiday", lambda s: s.std(ddof=1)))
tab["test_mae"] = {k: np.abs(p - y).mean() for k, p in P.items()}; tab["test_rmse"] = {k: np.sqrt(((p - y) ** 2).mean()) for k, p in P.items()}; tab["test_r2"] = {k: r2(p) for k, p in P.items()}
tab["test_mae_holiday"] = {k: np.abs(p[lw | eve] - y[lw | eve]).mean() for k, p in P.items()}; tab = tab.loc[[k for k in P if k in tab.index]].round(3)
tab.to_csv("results/final2_main_table.csv", encoding="utf-8-sig"); print("\nMAIN TABLE"); print(tab.to_string())
strongest = cv.groupby("model").mae.mean().drop([FINAL, "Flexible linear: large one-hot Ridge"], errors="ignore").idxmin(); print("strongest baseline by CV MAE:", strongest)
a, b = cv[cv.model == FINAL].sort_values("fold").mae.values, cv[cv.model == strongest].sort_values("fold").mae.values; tt = stats.ttest_rel(a, b)
print(f"paired t-test over 4 folds, ours vs {strongest}: t={tt.statistic:.2f}, p={tt.pvalue:.3f}; fold diffs {np.round(a - b, 3)}")

# ---------------- day-block bootstrap on test ----------------
rng = np.random.default_rng(0); day_id = test.index.normalize().values; out = []
for g, m in groups.items():
    if m.sum() == 0: continue
    days = np.unique(day_id[m]); ae = {k: pd.Series(np.abs(p[m] - y[m])).groupby(day_id[m]).sum() for k, p in P.items()}
    for base in [k for k in P if k not in (FINAL,)]:
        imp = [100 * (1 - ae[FINAL].loc[pk].sum() / ae[base].loc[pk].sum()) for pk in (rng.choice(days, len(days)) for _ in range(2000))]
        out.append({"slice": g, "days": len(days), "baseline": base, "improvement_%": round(100 * (1 - ae[FINAL].sum() / ae[base].sum()), 1), "ci95_low": round(float(np.percentile(imp, 2.5)), 1), "ci95_high": round(float(np.percentile(imp, 97.5)), 1),
                    "gap_minutes": round(float((ae[base].sum() - ae[FINAL].sum()) / m.sum()), 3)})
bt = pd.DataFrame(out); bt.to_csv("results/final2_bootstrap.csv", index=False, encoding="utf-8-sig")
print("\nBOOTSTRAP vs strongest and key baselines"); print(bt[bt.baseline.isin([strongest, "Persistence-type: last week same slot", "Same holiday last time (else last week)", "Seasonal mean: slot x weekday"])].to_string(index=False))

# ---------------- decision metric with minute thresholds ----------------
DM = {k: P[k] for k in P if not k.startswith("Exploratory")}
dr = decision_regret(test, DM, 60); dr.to_csv("results/final2_decision_windows.csv", index=False, encoding="utf-8-sig")
dg = {"all": None, "ordinary (weekday+sat+sun)": ORD, "long weekend days only": ["lw_first", "lw_mid", "lw_last"], "lw_first": ["lw_first"], "eve_of_long": ["eve_of_long"], "holiday days (long weekend + eve)": HOL}
dd_, db_ = [], []; names = ["leave at window start"] + list(DM)
for g, types in dg.items():
    sub = dr if types is None else dr[dr.daytype.isin(types)]
    dd_.append({"group": g, "n_windows": len(sub), "n_days": sub.date.nunique(), **{k: round(sub[k].mean(), 2) for k in names}, "share regret<=1min (ours)": round(float((sub[FINAL] <= 1).mean()), 3)})
    ps = sub.groupby("date")[names].sum(); cnt = sub.groupby("date").size(); days = ps.index.values
    for base in names:
        if base == FINAL: continue
        dif = [(ps[base].loc[pk].sum() - ps[FINAL].loc[pk].sum()) / cnt.loc[pk].sum() for pk in (rng.choice(days, len(days)) for _ in range(2000))]
        db_.append({"group": g, "baseline": base, "minutes_saved_per_window": round((ps[base].sum() - ps[FINAL].sum()) / cnt.sum(), 2), "ci95_low": round(float(np.percentile(dif, 2.5)), 2), "ci95_high": round(float(np.percentile(dif, 97.5)), 2)})
pd.DataFrame(dd_).to_csv("results/final2_decision.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(db_).to_csv("results/final2_decision_bootstrap.csv", index=False, encoding="utf-8-sig")
print("\nDECISION"); print(pd.DataFrame(dd_).to_string(index=False)); print(pd.DataFrame(db_)[lambda x: x.group.isin(["long weekend days only", "holiday days (long weekend + eve)", "ordinary (weekday+sat+sun)"])].to_string(index=False))
