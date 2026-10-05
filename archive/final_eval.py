"""Final evaluation. The configuration comes from cv_select.py (chosen on 2024 quarterly validation only). Train = 2023+2024, test = 2025, used once.
Contents: baselines (last week, same holiday last time else last week, weighted mean per slot x weekday, trivial training mean, single-feature OLS on last week),
the selected Ridge, GBM with the SAME training weights, CV mean +- sd on the four 2024 validation quarters, day-level bootstrap, decision metric with a
minute threshold, last-year-holiday comparison restricted to rows where it exists, ex-post/bias slices, ablation Set A / B / C and drop-one-group.
Outputs final_*.csv. Usage: python final_eval.py (after cv_select.py)"""
import json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from protocol import *
from forecast_common import decision_regret

cfg = json.load(open("results/cv_selected.json")); print("selected config:", cfg)
d = data(); train, test = d[d.index.year <= 2024], d[d.index.year == 2025]
W = drift_weights(train, cfg["w"]); y = test.minutes.values

def wmean_slot_dow(tr, te, w):
    g = (tr.assign(w=w, wy=w * tr.minutes.values).groupby(["slot", "dow"])[["w", "wy"]].sum()); m = (g.wy / g.w).rename("m").reset_index()
    return te[["slot", "dow"]].merge(m, on=["slot", "dow"], how="left").m.values
def simple_ols(tr, te, w):
    x, yy = tr.lag7.values, tr.minutes.values; A = np.c_[np.ones(len(x)), x] * np.sqrt(w)[:, None]; beta = np.linalg.lstsq(A, yy * np.sqrt(w), rcond=None)[0]
    return beta[0] + beta[1] * te.lag7.values
def baselines(tr, te, w):
    return {"last week (lag 7d)": te.lag7.values, "same holiday last time, else last week": np.where(te.ly.notna(), te.ly, te.lag7),
            "mean per slot x weekday (weighted)": wmean_slot_dow(tr, te, w), "trivial: training mean": np.full(len(te), np.average(tr.minutes.values, weights=w)),
            "simple: OLS on last week only": simple_ols(tr, te, w)}

# ---------- test predictions ----------
P = baselines(train, test, W)
P["Ridge (selected)"] = predict_cfg(train, test, cfg)
for loss in ["log", "l1"]: P[f"Ridge {loss}, other settings as selected"] = predict_cfg(train, test, {**cfg, "loss": loss})
P["Ridge (selected) without last-year feature"] = predict_cfg(train, test, {**cfg, "ly": 0})
P["Ridge (selected) with last-year feature"] = predict_cfg(train, test, {**cfg, "ly": 1})
P["GBM log (same weights)"], _ = gbm_fit(train, test, False, True, "regression", weights=W, ly=bool(cfg["ly"]))
P["GBM L1 (same weights)"], _ = gbm_fit(train, test, False, False, "regression_l1", weights=W, ly=bool(cfg["ly"]))
pd.DataFrame({"depart": test.index, "daytype": test.daytype.values, "actual": y, **{k: v for k, v in P.items()}}).to_csv("archive/results/final_predictions.csv", index=False, encoding="utf-8-sig")

dt, hod, dow = test.daytype.values, test.hod.values, test.dow.values; lw = np.isin(dt, ["lw_first", "lw_mid", "lw_last"]); eve = dt == "eve_of_long"
has_ly = test.ly.notna().values
groups = {"all slots": np.ones(len(test), bool), "ordinary (weekday+sat+sun)": np.isin(dt, ORD), "Saturday": dt == "sat", "Saturday 08-17h": (dt == "sat") & (hod >= 8) & (hod < 17),
          "holiday days (long weekend + eve)": lw | eve, "long weekend, all hours": lw, "long weekend 08-18h": lw & (hod >= 8) & (hod < 18),
          "long weekend morning 07-12h": lw & (hod >= 7) & (hod < 12), "lw_first 10-18h": (dt == "lw_first") & (hod >= 10) & (hod < 18), "eve 16-23h": eve & (hod >= 16),
          "long weekend night 0-6h + 20-24h": lw & ((hod < 6) | (hod >= 20)), "holiday days WITH last-year value": (lw | eve) & has_ly,
          "holiday days WITHOUT last-year value": (lw | eve) & ~has_ly, "long weekend 08-18h WITH last-year value": lw & (hod >= 8) & (hod < 18) & has_ly}
FF = float(train[train.hod < 6].minutes.median())
groups["[ex post] actual >= 1.25 x free flow"] = y >= 1.25 * FF; groups["[ex post] actual >= 1.5 x free flow"] = y >= 1.5 * FF
rows, brows = [], []
for g, m in groups.items():
    r = {"slice": g, "n": int(m.sum()), "days": int(pd.Series(test.index.normalize()[m]).nunique()), "mean actual": round(float(y[m].mean()), 1)}; b = {"slice": g, "n": int(m.sum())}
    for k, p in P.items(): r[k] = round(float(np.abs(p[m] - y[m]).mean()), 2); b[k] = round(float((y[m] - p[m]).mean()), 2)
    rows.append(r); brows.append(b)
res = pd.DataFrame(rows); res.to_csv("archive/results/final_mae.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(brows).to_csv("archive/results/final_bias.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40)
SEL = "Ridge (selected)"
print("MAE (minutes), test 2025"); print(res[["slice", "n", "days", "mean actual"] + list(P)[:5] + [SEL, "Ridge log, other settings as selected", "Ridge l1, other settings as selected", "GBM log (same weights)", "GBM L1 (same weights)"]].to_string(index=False))
print("\nLast-year feature on/off"); print(res[["slice", "Ridge (selected) without last-year feature", "Ridge (selected) with last-year feature"]].to_string(index=False))

# ---------- day-level bootstrap ----------
rng = np.random.default_rng(0); day_id = test.index.normalize().values; out = []
pairs = [(SEL, b) for b in list(P)[:5]] + [(SEL, "GBM log (same weights)"), (SEL, "GBM L1 (same weights)"), ("Ridge (selected) with last-year feature", "Ridge (selected) without last-year feature")]
for g, m in groups.items():
    if m.sum() == 0: continue
    days = np.unique(day_id[m]); ae = {k: pd.Series(np.abs(p[m] - y[m])).groupby(day_id[m]).sum() for k, p in P.items()}
    for cand, base in pairs:
        imp = [100 * (1 - ae[cand].loc[pk].sum() / ae[base].loc[pk].sum()) for pk in (rng.choice(days, len(days)) for _ in range(2000))]
        out.append({"slice": g, "days": len(days), "candidate": cand, "baseline": base, "improvement_%": round(100 * (1 - ae[cand].sum() / ae[base].sum()), 1),
                    "ci95_low": round(float(np.percentile(imp, 2.5)), 1), "ci95_high": round(float(np.percentile(imp, 97.5)), 1)})
bt = pd.DataFrame(out); bt.to_csv("archive/results/final_bootstrap.csv", index=False, encoding="utf-8-sig")
print("\nBootstrap: selected Ridge vs baselines"); print(bt[(bt.candidate == SEL) & bt.slice.isin(["all slots", "ordinary (weekday+sat+sun)", "holiday days (long weekend + eve)", "long weekend, all hours", "long weekend 08-18h", "lw_first 10-18h", "eve 16-23h", "Saturday 08-17h", "holiday days WITH last-year value", "long weekend 08-18h WITH last-year value"])].to_string(index=False))

# ---------- decision metric and minute threshold ----------
DM = {k: P[k] for k in [*list(P)[:5], SEL, "GBM log (same weights)", "GBM L1 (same weights)"]}
dr = decision_regret(test, DM, 60); dr.to_csv("archive/results/final_decision_windows.csv", index=False, encoding="utf-8-sig")
dg = {"all": None, "ordinary (weekday+sat+sun)": ORD, "long weekend days only": ["lw_first", "lw_mid", "lw_last"], "lw_first": ["lw_first"], "eve_of_long": ["eve_of_long"], "holiday days (long weekend + eve)": HOL}
dd, db = [], []; names = ["leave at window start"] + list(DM)
for g, types in dg.items():
    sub = dr if types is None else dr[dr.daytype.isin(types)]
    dd.append({"group": g, "n_windows": len(sub), "n_days": sub.date.nunique(), **{k: round(sub[k].mean(), 2) for k in names}, "share windows regret <= 1 min (selected)": round(float((sub[SEL] <= 1).mean()), 3)})
    ps = sub.groupby("date")[names].sum(); cnt = sub.groupby("date").size(); days = ps.index.values
    for base in names:
        if base == SEL: continue
        dif = [(ps[base].loc[pk].sum() - ps[SEL].loc[pk].sum()) / cnt.loc[pk].sum() for pk in (rng.choice(days, len(days)) for _ in range(2000))]
        db.append({"group": g, "baseline": base, "minutes_saved_per_window": round((ps[base].sum() - ps[SEL].sum()) / cnt.sum(), 2), "ci95_low": round(float(np.percentile(dif, 2.5)), 2), "ci95_high": round(float(np.percentile(dif, 97.5)), 2)})
dd = pd.DataFrame(dd); dd.to_csv("archive/results/final_decision.csv", index=False, encoding="utf-8-sig"); db = pd.DataFrame(db); db.to_csv("archive/results/final_decision_bootstrap.csv", index=False, encoding="utf-8-sig")
print("\nDecision regret (minutes lost per 4-hour window)"); print(dd.to_string(index=False)); print(db[db.group.isin(["long weekend days only", "holiday days (long weekend + eve)", "ordinary (weekday+sat+sun)"])].to_string(index=False))
lwd = dd[dd.group == "long weekend days only"].iloc[0]
print(f"\nPRE-SPECIFIED THRESHOLDS: long-weekend mean regret <= 1.0 min -> {lwd[SEL]} ({'met' if lwd[SEL] <= 1 else 'NOT met'}); "
      f"saving vs leaving at window start >= 1.0 min -> see bootstrap row")

# ---------- CV mean +- sd on the four 2024 validation quarters (same weights scheme) ----------
cvr = []
for s, e in QUARTERS:
    tr, va = fold_split(d, s, e); w = drift_weights(tr, cfg["w"]); B = baselines(tr, va, w); B[SEL] = predict_cfg(tr, va, cfg)
    B["GBM log (same weights)"], _ = gbm_fit(tr, va, False, True, "regression", weights=w, ly=bool(cfg["ly"])); B["GBM L1 (same weights)"], _ = gbm_fit(tr, va, False, False, "regression_l1", weights=w, ly=bool(cfg["ly"]))
    h = va.daytype.isin(HOL).values
    for k, p in B.items(): cvr.append({"fold": s, "model": k, "mae_all": float(np.abs(p - va.minutes.values).mean()), "mae_holiday": float(np.abs(p[h] - va.minutes.values[h]).mean()) if h.any() else np.nan})
cv = pd.DataFrame(cvr); cv.to_csv("archive/results/final_cv_folds.csv", index=False, encoding="utf-8-sig")
cvs = cv.groupby("model").agg(mae_all_mean=("mae_all", "mean"), mae_all_sd=("mae_all", lambda s: s.std(ddof=1)), mae_holiday_mean=("mae_holiday", "mean"), mae_holiday_sd=("mae_holiday", lambda s: s.std(ddof=1))).round(3)
cvs.to_csv("archive/results/final_cv_summary.csv", encoding="utf-8-sig"); print("\nCV on 2024 quarters (mean, sd over 4 folds)"); print(cvs.to_string())

# ---------- ablation ----------
abl, brow = [], []
sets = ["A", "B", "C", "C-interactions", "C-holiday structure", "C-weekly rhythm (slot x weekday)", "C-eve effect", "C-ly"]; PA = {}
for s_ in sets:
    PA[s_] = predict_cfg(train, test, cfg if s_ != "C-ly" else {**cfg, "ly": 1}, feature_set=s_)
    cvv = []
    for s, e in QUARTERS:
        tr, va = fold_split(d, s, e); p = predict_cfg(tr, va, cfg if s_ != "C-ly" else {**cfg, "ly": 1}, feature_set=s_); cvv.append(float(np.abs(p - va.minutes.values).mean()))
    r = {"set": s_, "cv_mae_mean": round(np.mean(cvv), 3), "cv_mae_sd": round(np.std(cvv, ddof=1), 3)}
    for g in ["all slots", "ordinary (weekday+sat+sun)", "holiday days (long weekend + eve)", "long weekend 08-18h", "lw_first 10-18h", "eve 16-23h"]: r["test " + g] = round(float(np.abs(PA[s_][groups[g]] - y[groups[g]]).mean()), 2)
    abl.append(r)
ab = pd.DataFrame(abl); ab.to_csv("archive/results/final_ablation.csv", index=False, encoding="utf-8-sig"); print("\nAblation"); print(ab.to_string(index=False))
ob = []
for g in ["all slots", "holiday days (long weekend + eve)", "long weekend 08-18h", "lw_first 10-18h"]:
    m = groups[g]; days = np.unique(day_id[m]); ae = {k: pd.Series(np.abs(p[m] - y[m])).groupby(day_id[m]).sum() for k, p in PA.items()}
    for s_ in sets:
        if s_ == "C": continue
        imp = [100 * (1 - ae["C"].loc[pk].sum() / ae[s_].loc[pk].sum()) for pk in (rng.choice(days, len(days)) for _ in range(2000))]
        ob.append({"slice": g, "set": s_, "C better than set by %": round(100 * (1 - ae["C"].sum() / ae[s_].sum()), 1), "ci95_low": round(float(np.percentile(imp, 2.5)), 1), "ci95_high": round(float(np.percentile(imp, 97.5)), 1)})
pd.DataFrame(ob).to_csv("archive/results/final_ablation_bootstrap.csv", index=False, encoding="utf-8-sig"); print(pd.DataFrame(ob).to_string(index=False))
