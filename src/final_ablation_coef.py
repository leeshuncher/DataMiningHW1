"""Feature ablation (Set A raw, Set B top-k correlated selected INSIDE each training fold, Set C ours, plus drop-one-group from C) and the coefficient table of the final
compact model (beta in MINUTES, day-block bootstrap 95% CI and p, VIF). Same model type, split, weights and tuning for every set; only the features change.
Outputs final2_ablation*.csv, final2_coefficients.csv. Usage: python src/final_ablation_coef.py (after final_compact.py)"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from statsmodels.stats.outliers_influence import variance_inflation_factor
from protocol import *
import compact_model as cm

NAME, LOSS, W23, K = "compact8h", "l1", 0.1, 30
d = data(); train, test = d[d.index.year <= 2024], d[d.index.year == 2025]; y = test.minutes.values
E, L = cm.blocks(NAME); HRS = NAME.endswith("h")
def setA(df):   # raw columns: hour of day (one-hot), weekday (one-hot), long-weekend flag
    X = pd.get_dummies(df.slot.astype(int), prefix="hour").astype(float).set_index(df.index)
    X = X.join(pd.get_dummies(df.dow.astype(int), prefix="dow").astype(float).set_index(df.index)); X["lw_flag"] = df.daytype.str.startswith("lw_").astype(float).values; return X
def topk_cols(tr, k):
    X = cm.compact_X(tr, E, L, hours=HRS); yy = tr.minutes.values; Xc = X.values - X.values.mean(0); yc = yy - yy.mean()
    corr = np.abs((Xc * yc[:, None]).sum(0)) / (np.sqrt((Xc ** 2).sum(0)) * np.sqrt((yc ** 2).sum()) + 1e-12); return list(X.columns[np.argsort(-corr)[:k]])
DROPS = {"C minus day-type x time-block interactions": ("sat_", "sun_", "lwf_", "lwm_", "lwl_", "eveF_", "eveO_", "single_", "makeup_"),
         "C minus Friday terms": ("fri_",), "C minus eve-of-long-weekend terms": ("eveF", "eveO"), "C minus long-weekend terms": ("lwf", "lwm", "lwl"),
         "C minus long-holiday (>=4 days) term": ("lwf_pm_long4",), "C minus Saturday terms": ("sat",)}
def predict(tr, te, setname):
    if setname == "A": return cm.fit_predict(tr, te, NAME, LOSS, W23, custom_X=setA)[0]
    if setname == "B": return cm.fit_predict(tr, te, NAME, LOSS, W23, keep_cols=topk_cols(tr, K))[0]
    if setname == "C": return cm.fit_predict(tr, te, NAME, LOSS, W23)[0]
    return cm.fit_predict(tr, te, NAME, LOSS, W23, drop=DROPS[setname])[0]
sets = ["A", "B", "C"] + list(DROPS); PA = {s: predict(train, test, s) for s in sets}
dt, hod = test.daytype.values, test.hod.values; lw = np.isin(dt, ["lw_first", "lw_mid", "lw_last"]); eve = dt == "eve_of_long"
groups = {"all slots": np.ones(len(test), bool), "ordinary (weekday+sat+sun)": np.isin(dt, ORD), "holiday days": lw | eve, "long weekend 08-18h": lw & (hod >= 8) & (hod < 18), "lw_first 10-18h": (dt == "lw_first") & (hod >= 10) & (hod < 18), "eve 16-23h": eve & (hod >= 16), "Saturday 08-17h": (dt == "sat") & (hod >= 8) & (hod < 17)}
rows = []
for s in sets:
    cvf = []
    for a, b in QUARTERS:
        tr, va = fold_split(d, a, b); cvf.append(float(np.abs(predict(tr, va, s) - va.minutes.values).mean()))
    r = {"set": s, "n_terms": None, "cv_mae_mean": round(np.mean(cvf), 3), "cv_mae_sd": round(np.std(cvf, ddof=1), 3)}
    for g, m in groups.items(): r["test " + g] = round(float(np.abs(PA[s][m] - y[m]).mean()), 2)
    rows.append(r)
ab = pd.DataFrame(rows); ab.to_csv("results/final2_ablation.csv", index=False, encoding="utf-8-sig"); pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); print(ab.to_string(index=False))
# Set B sensitivity to k (CV only)
sens = []
for k in [15, 30, 60]:
    cvf = [float(np.abs(cm.fit_predict(*(lambda tr, va: (tr, va))(*fold_split(d, a, b)), NAME, LOSS, W23, keep_cols=topk_cols(fold_split(d, a, b)[0], k))[0] - fold_split(d, a, b)[1].minutes.values).mean()) for a, b in QUARTERS]
    sens.append({"k": k, "cv_mae_mean": round(np.mean(cvf), 3), "cv_mae_sd": round(np.std(cvf, ddof=1), 3)})
pd.DataFrame(sens).to_csv("results/final2_ablation_setB_k.csv", index=False); print(pd.DataFrame(sens).to_string(index=False))
rng = np.random.default_rng(0); day_id = test.index.normalize().values; ob = []
for g, m in groups.items():
    days = np.unique(day_id[m]); ae = {k: pd.Series(np.abs(p[m] - y[m])).groupby(day_id[m]).sum() for k, p in PA.items()}
    for s in sets:
        if s == "C": continue
        imp = [100 * (1 - ae["C"].loc[pk].sum() / ae[s].loc[pk].sum()) for pk in (rng.choice(days, len(days)) for _ in range(2000))]
        ob.append({"slice": g, "set": s, "C better than set by %": round(100 * (1 - ae["C"].sum() / ae[s].sum()), 1), "ci95_low": round(float(np.percentile(imp, 2.5)), 1), "ci95_high": round(float(np.percentile(imp, 97.5)), 1)})
pd.DataFrame(ob).to_csv("results/final2_ablation_bootstrap.csv", index=False, encoding="utf-8-sig"); print(pd.DataFrame(ob)[lambda x: x.slice.isin(["all slots", "holiday days", "long weekend 08-18h"])].to_string(index=False))

# ---------------- coefficient table of the final model ----------------
_, alpha, cols = cm.fit_predict(train, test, NAME, LOSS, W23); Xdf = cm.compact_X(train, E, L, hours=HRS)[cols]; X = Xdf.values; yy = train.minutes.values; w = drift_weights(train, W23)
def irls(Xa, ya, wa, a, it=8):
    m = Ridge(alpha=a); ww = wa.copy()
    for _ in range(it): m.fit(Xa, ya, sample_weight=ww); ww = wa / np.maximum(np.abs(ya - m.predict(Xa)), 0.25)
    return m
m0 = irls(X, yy, w, alpha); beta0 = m0.coef_; dayg = pd.factorize(train.index.normalize())[0]; ndays = dayg.max() + 1; idx_by_day = [np.where(dayg == i)[0] for i in range(ndays)]
B = np.zeros((300, X.shape[1]))
for i in range(300):
    pick = rng.integers(0, ndays, ndays); ii = np.concatenate([idx_by_day[j] for j in pick]); B[i] = irls(X[ii], yy[ii], w[ii], alpha).coef_
vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
co = pd.DataFrame({"beta_minutes": beta0, "ci_low": np.percentile(B, 2.5, axis=0), "ci_high": np.percentile(B, 97.5, axis=0), "se_boot": B.std(0, ddof=1),
                   "p_boot": 2 * np.minimum((B > 0).mean(0), (B < 0).mean(0)).clip(1 / 300, 1), "VIF": vif, "n_rows_nonzero": (X != 0).sum(0)}, index=cols).round(3)
co.to_csv("results/final2_coefficients.csv", encoding="utf-8-sig"); print("alpha", alpha, "| terms", len(cols), "| intercept", round(m0.intercept_, 2), "| max VIF", round(max(vif), 1)); print(co.sort_values("beta_minutes", ascending=False).head(12).to_string()); print(co.sort_values("beta_minutes").head(6).to_string())
