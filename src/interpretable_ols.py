"""Compact interpretable model on TRAINING YEARS ONLY (2023-2024): WLS of log(travel time) on time-of-day blocks x day types, with day-clustered
(Newey-West-style robust to within-day correlation) standard errors, VIF, effects as % and as minutes, and formal tests of the proposal's hypotheses H1-H4.
Weights: 2023 ordinary days x w (w from cv_selected.json, same scheme as the forecasting model).
Outputs ols_coefficients.csv, ols_hypotheses.csv, ols_vif.csv. Usage: python src/interpretable_ols.py"""
import json, os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from protocol import data, drift_weights

w2023 = json.load(open("results/cv_selected.json"))["w"] if os.path.exists("results/cv_selected.json") else 0.1
d = data(); d = d[d.index.year <= 2024].copy(); d["date"] = d.index.normalize()
d["blk"] = pd.cut(d.hod, [-1, 5.99, 9.99, 13.99, 17.99, 21.99, 24], labels=["night", "am", "mid", "pm", "ev", "night2"]).astype(str).replace({"night2": "night"})
types = {"sat": d.daytype == "sat", "sun": d.daytype == "sun", "lwf": d.daytype == "lw_first", "lwm": d.daytype == "lw_mid", "lwl": d.daytype == "lw_last",
         "eveF": (d.daytype == "eve_of_long") & (d.dow == 4), "eveO": (d.daytype == "eve_of_long") & (d.dow != 4), "single": d.daytype == "single_holiday", "makeup": d.daytype == "makeup_workday"}
X = pd.DataFrame(index=d.index)
for b in ["am", "mid", "pm", "ev"]: X["blk_" + b] = (d.blk == b).astype(float)
fri = ((d.daytype == "weekday") & (d.dow == 4))
for b in ["pm", "ev"]: X["fri_" + b] = (fri & (d.blk == b)).astype(float)
for k, m in types.items():
    X[k] = m.astype(float)
    for b in ["am", "mid", "pm", "ev"]: X[f"{k}_{b}"] = (m & (d.blk == b)).astype(float)
long4 = (d.daytype == "lw_first") & (d.block_len >= 4) & (d.blk == "pm"); X["lwf_pm_long4"] = long4.astype(float)
X = sm.add_constant(X); X = X.loc[:, (X != 0).any(axis=0)]
y = np.log(d.minutes.values); w = drift_weights(d, w2023)
res = sm.WLS(y, X, weights=w).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.date)[0]})
ref = {b: float(d[(d.daytype == "weekday") & (d.blk == b)].minutes.mean()) for b in ["night", "am", "mid", "pm", "ev"]}   # ordinary weekday level in that block
co = pd.DataFrame({"beta": res.params, "se_cluster": res.bse, "p": res.pvalues, "ci_low": res.conf_int()[0], "ci_high": res.conf_int()[1]})
co["effect_%"] = (np.exp(co.beta) - 1) * 100; co["effect_low_%"] = (np.exp(co.ci_low) - 1) * 100; co["effect_high_%"] = (np.exp(co.ci_high) - 1) * 100
def blk_of(name): return name.split("_")[1] if "_" in name and name.split("_")[1] in ref else "night"
co["effect_minutes"] = [(np.exp(b) - 1) * ref.get(blk_of(n), ref["night"]) if n != "const" else np.nan for n, b in co.beta.items()]
vif = pd.Series([variance_inflation_factor(X.drop(columns="const").values, i) for i in range(X.shape[1] - 1)], index=X.drop(columns="const").columns, name="VIF")
co = co.join(vif); co.round(4).to_csv("results/ols_coefficients.csv", encoding="utf-8-sig")
print("rows", len(d), "days", d.date.nunique(), "terms", X.shape[1], "R2 (weighted)", round(res.rsquared, 3), "| 2023 ordinary weight", w2023, "| max VIF", round(vif.max(), 1))
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 100)
key = [n for n in co.index if n.endswith(("_pm", "_ev", "_am", "_mid")) and n.split("_")[0] in ("lwf", "lwm", "lwl", "eveF", "eveO", "sat")] + ["lwf_pm_long4", "fri_pm", "fri_ev"]
print(co.loc[key, ["effect_%", "effect_low_%", "effect_high_%", "effect_minutes", "p", "VIF"]].round(2).to_string())

H = []
def test(name, expr, claim, one_sided=None):
    t = res.t_test(expr); b = float(np.squeeze(t.effect)); p2 = float(np.squeeze(t.pvalue)); se = float(np.squeeze(t.sd))
    p1 = (p2 / 2 if b > 0 else 1 - p2 / 2) if one_sided == ">" else ((p2 / 2 if b < 0 else 1 - p2 / 2) if one_sided == "<" else np.nan)
    H.append({"hypothesis": name, "contrast": expr, "claim": claim, "beta_diff": round(b, 4), "effect_%": round((np.exp(b) - 1) * 100, 1), "se": round(se, 4), "p_two_sided": round(p2, 4), "p_one_sided": round(p1, 4) if not np.isnan(p1) else np.nan})
test("H1 first-day morning slower than ordinary Saturday morning", "lwf_am - sat_am = 0", "first-day morning (6-10h) vs Saturday morning; H1 predicts > 0", ">")
test("H1b first-day afternoon peak vs Saturday afternoon", "lwf_pm - sat_pm = 0", "14-18h", None)
for blk in ["am", "mid", "pm", "ev"]: test(f"H2 last day vs ordinary weekday, {blk}", f"lwl_{blk} = 0", "H2 predicts about 0")
w_all = res.wald_test("lwl_am = 0, lwl_mid = 0, lwl_pm = 0, lwl_ev = 0", scalar=True); H.append({"hypothesis": "H2 joint: last day = ordinary weekday in all blocks", "claim": "joint Wald", "p_two_sided": round(float(w_all.pvalue), 4)})
test("H3a eve (Friday) evening vs ordinary Friday evening", "eveF_ev = 0", "beyond ordinary Friday, 18-22h; H3 predicts > 0", ">")
test("H3b eve (other weekday) evening vs ordinary weekday evening", "eveO_ev = 0", "18-22h", ">")
test("H3c eve (other weekday) afternoon vs ordinary weekday", "eveO_pm = 0", "14-18h", ">")
test("H4 longer holidays (>=4 days) have a lower first-day afternoon peak", "lwf_pm_long4 = 0", "H4 predicts < 0", "<")
hy = pd.DataFrame(H); hy.to_csv("results/ols_hypotheses.csv", index=False, encoding="utf-8-sig"); print(hy.to_string(index=False))
