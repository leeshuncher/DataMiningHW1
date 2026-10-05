"""Pre-registered blind test on 2026-01..08 (see docs/BLIND_TEST_2026.md). The design is frozen; nothing here is tuned on 2026.
Primary (P): refit on 2023-2025, predict 2026. Secondary (S): models fitted on 2023-2024 as in the paper, applied to 2026.
Outputs results/blind2026_{main,slices,bootstrap,decision}.csv. Usage: python src/blind_2026.py"""
import json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from protocol import data, drift_weights, predict_cfg, ORD, HOL
from forecast_common import decision_regret, gbm_fit
import compact_model as cm

NAME, LOSS, W23 = "compact8h", "l1", 0.1; FINAL = "Ours: compact median linear"; GBM = "Strong: GBM (L1)"
cfg_large = json.load(open("results/cv_selected.json"))
d = data(); test = d[(d.index >= "2026-01-01") & (d.index < "2026-09-01")]; y = test.minutes.values

def wmean(tr, te, w):
    g = tr.assign(w=w, wy=w * tr.minutes.values).groupby(["slot", "dow"])[["w", "wy"]].sum(); m = (g.wy / g.w).rename("m").reset_index()
    return te[["slot", "dow"]].merge(m, on=["slot", "dow"], how="left").m.values
def sols(tr, te, w):
    A = np.c_[np.ones(len(tr)), tr.lag7.values] * np.sqrt(w)[:, None]; b = np.linalg.lstsq(A, tr.minutes.values * np.sqrt(w), rcond=None)[0]; return b[0] + b[1] * te.lag7.values
def models(tr, te):
    w = drift_weights(tr, W23)
    P = {"Trivial: training mean": np.full(len(te), np.average(tr.minutes.values, weights=w)), "Last week, same hour": te.lag7.values,
         "Same holiday last time": np.where(te.ly.notna(), te.ly, te.lag7), "Seasonal mean: hour x weekday": wmean(tr, te, w), "Simple: OLS on last week": sols(tr, te, w)}
    P[FINAL] = cm.fit_predict(tr, te, NAME, LOSS, W23)[0]
    P[GBM], _ = gbm_fit(tr, te, False, False, "regression_l1", weights=w); P["Strong: GBM (log)"], _ = gbm_fit(tr, te, False, True, "regression", weights=w)
    P["Flexible linear: large one-hot ridge"] = predict_cfg(tr, te, cfg_large)
    return P

dt, hod = test.daytype.values, test.index.hour.values; lw = np.isin(dt, ["lw_first", "lw_mid", "lw_last"]); eve = dt == "eve_of_long"
SL = {"all": np.ones(len(test), bool), "ordinary days": np.isin(dt, ORD), "long-weekend days": lw, "holiday days (LW + eve)": lw | eve,
      "LW 08-18h": lw & (hod >= 8) & (hod < 18), "first day 10-18h": (dt == "lw_first") & (hod >= 10) & (hod < 18), "eve 16-23h": eve & (hod >= 16),
      "Saturday 08-17h": (dt == "sat") & (hod >= 8) & (hod < 17), "congested (actual > 40)": y > 40}
rng = np.random.default_rng(0); day_id = test.index.normalize().values
main, slices, boot, dec = [], [], [], []
for tag, train in [("P: train 2023-2025", d[d.index < "2026-01-01"]), ("S: train 2023-2024", d[d.index < "2025-01-01"])]:
    P = models(train, test)
    for k, p in P.items():
        main.append({"setting": tag, "model": k, "mae": np.abs(p - y).mean(), "rmse": np.sqrt(((p - y) ** 2).mean()), "r2": 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()})
        for s, m in SL.items(): slices.append({"setting": tag, "slice": s, "days": len(np.unique(day_id[m])), "rows": int(m.sum()), "model": k, "mae": np.abs(p[m] - y[m]).mean() if m.any() else np.nan})
    for s, m in SL.items():
        if not m.any(): continue
        days = np.unique(day_id[m]); ae = {k: pd.Series(np.abs(p[m] - y[m])).groupby(day_id[m]).sum() for k, p in P.items()}; n = pd.Series(1, index=day_id[m]).groupby(level=0).sum()
        for base in [k for k in P if k != FINAL]:
            res = [(1 - ae[FINAL].loc[pk].sum() / ae[base].loc[pk].sum(), (ae[FINAL].loc[pk].sum() - ae[base].loc[pk].sum()) / n.loc[pk].sum()) for pk in (rng.choice(days, len(days)) for _ in range(2000))]
            imp, gap = np.array(res).T
            boot.append({"setting": tag, "slice": s, "days": len(days), "baseline": base, "improvement_%": 100 * (1 - ae[FINAL].sum() / ae[base].sum()),
                         "imp_ci_lo": 100 * np.percentile(imp, 2.5), "imp_ci_hi": 100 * np.percentile(imp, 97.5), "ours_minus_base_min": (ae[FINAL].sum() - ae[base].sum()) / m.sum(),
                         "gap_ci_lo": np.percentile(gap, 2.5), "gap_ci_hi": np.percentile(gap, 97.5)})
    dr = decision_regret(test, P, 60); names = ["leave at window start"] + list(P)
    for g, types in {"all": None, "ordinary days": ORD, "long-weekend days": ["lw_first", "lw_mid", "lw_last"], "LW first day": ["lw_first"], "eve": ["eve_of_long"]}.items():
        sub = dr if types is None else dr[dr.daytype.isin(types)]
        dec.append({"setting": tag, "group": g, "windows": len(sub), "days": sub.date.nunique(), **{k: sub[k].mean() for k in names}, "share<=1min (ours)": (sub[FINAL] <= 1).mean()})
for nm, rows in [("main", main), ("slices", slices), ("bootstrap", boot), ("decision", dec)]:
    pd.DataFrame(rows).round(3).to_csv(f"results/blind2026_{nm}.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(pd.DataFrame(main).round(3).to_string(index=False))
print(pd.DataFrame(slices).pivot_table(index=["setting", "slice"], columns="model", values="mae").round(2).to_string())
b = pd.DataFrame(boot); print(b[b.baseline.isin(["Last week, same hour", "Seasonal mean: hour x weekday", GBM])].round(2).to_string(index=False))
print(pd.DataFrame(dec).round(2).to_string(index=False))
