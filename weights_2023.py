"""2023 differs from 2024/2025 (Saturdays much more congested), which hurts ordinary days but the extra long weekends help holiday days.
Idea: down-weight 2023 ORDINARY days (weekday/sat/sun) with weight w, keep full weight for holiday-structure days (long weekend, eve, single holiday, make-up day).
w is chosen on a validation block that does NOT include the test year: train = 2023 + 2024 H1, validate on 2024 H2 (ordinary days, MAE).
Then the chosen w is applied to train 2023+2024 and scored on 2025 (once). Usage: python weights_2023.py. Output: weights_2023_results.csv, weights_2023_bootstrap.csv"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from forecast_common import *

d = load_data(60); d["ly_f"] = d.ly.fillna(d.lag4mean); d["ly_ok"] = d.ly.notna().astype(float) * 30.0
ORD = ["weekday", "sat", "sun"]
def wts(df, w, scope):
    base = np.ones(len(df)); old = (df.index.year == 2023)
    if scope == "ordinary": base[old & df.daytype.isin(ORD).values] = w
    elif scope == "all": base[old] = w
    return base

# ---- choose w on 2024 H2 ----
tr_v = d[(d.index.year == 2023) | ((d.index.year == 2024) & (d.index.month <= 6))]; va = d[(d.index.year == 2024) & (d.index.month >= 7)]
va_ord = va.daytype.isin(ORD).values; trv, vav = add_cats(tr_v, "shared"), add_cats(va, "shared"); sel = []
for scope in ["ordinary", "all"]:
    for w in [1.0, 0.5, 0.25, 0.1, 0.03]:
        p, _ = ridge_fit(trv, vav, CATS, [], True, weights=wts(tr_v, w, scope)); e = np.abs(p - va.minutes.values)
        sel.append({"scope": scope, "w2023": w, "val MAE ordinary 2024H2": round(float(e[va_ord].mean()), 3), "val MAE all 2024H2": round(float(e.mean()), 3)})
sel = pd.DataFrame(sel); print(sel.to_string(index=False))
best_w = float(sel[sel.scope == "ordinary"].sort_values("val MAE ordinary 2024H2").iloc[0].w2023); print("chosen w (ordinary scope):", best_w)

# ---- final: train 2023+2024, test 2025 ----
test = d[d.index.year == 2025]; train = d[d.index.year.isin([2023, 2024])]; t24 = d[d.index.year == 2024]
tr, te = add_cats(train, "shared"), add_cats(test, "shared"); t24c = add_cats(t24, "shared")
P = {"last week (lag 7d)": test.lag7.values,
     "same holiday last time, else last week": np.where(test.ly.notna(), test.ly, test.lag7),
     "mean per slot x weekday (2024)": test[["slot", "dow"]].merge(t24.groupby(["slot", "dow"]).minutes.mean().rename("m").reset_index(), on=["slot", "dow"], how="left").m.values}
P["Ridge log, train 2024"], _ = ridge_fit(t24c, te, CATS, [], True)
P["Ridge log, 2023+2024 uniform"], _ = ridge_fit(tr, te, CATS, [], True)
P[f"Ridge log, 2023 ordinary x{best_w}"], _ = ridge_fit(tr, te, CATS, [], True, weights=wts(train, best_w, "ordinary"))
P["Ridge log +ly, train 2024"], _ = ridge_fit(t24c, te, CATS, ["ly_f", "ly_ok"], True)
P["Ridge log +ly, 2023+2024 uniform"], _ = ridge_fit(tr, te, CATS, ["ly_f", "ly_ok"], True)
P[f"Ridge log +ly, 2023 ordinary x{best_w}"], _ = ridge_fit(tr, te, CATS, ["ly_f", "ly_ok"], True, weights=wts(train, best_w, "ordinary"))
y, dt, hod, dow = test.minutes.values, test.daytype.values, test.hod.values, test.dow.values
lw = np.isin(dt, ["lw_first", "lw_mid", "lw_last"]); eve = dt == "eve_of_long"
groups = {"all slots": np.ones(len(test), bool), "ordinary (weekday+sat+sun)": np.isin(dt, ORD), "weekday": dt == "weekday", "sat": dt == "sat", "sun": dt == "sun",
          "holiday days (long weekend + eve)": lw | eve, "long weekend, all hours": lw, "long weekend 08-18h": lw & (hod >= 8) & (hod < 18),
          "lw_first 10-18h": (dt == "lw_first") & (hod >= 10) & (hod < 18), "eve 16-23h": eve & (hod >= 16), "Saturday 08-17h": (dt == "sat") & (hod >= 8) & (hod < 17)}
res = pd.DataFrame([{"slice": g, "n": int(m.sum()), "days": int(pd.Series(test.index.normalize()[m]).nunique()),
                     **{k: round(float(np.abs(p[m] - y[m]).mean()), 2) for k, p in P.items()}} for g, m in groups.items()])
res.to_csv("weights_2023_results.csv", index=False, encoding="utf-8-sig"); pd.set_option("display.width", 320); pd.set_option("display.max_columns", 30); print(res.to_string(index=False))
rng = np.random.default_rng(0); day_id = test.index.normalize().values; out = []
W = f"Ridge log, 2023 ordinary x{best_w}"; WL = f"Ridge log +ly, 2023 ordinary x{best_w}"
for g, m in groups.items():
    days = np.unique(day_id[m]); ae = {k: pd.Series(np.abs(p[m] - y[m])).groupby(day_id[m]).sum() for k, p in P.items()}
    for cand, base in [(W, "Ridge log, 2023+2024 uniform"), (W, "Ridge log, train 2024"), (W, "last week (lag 7d)"), (W, "mean per slot x weekday (2024)"),
                       (WL, W), (WL, "same holiday last time, else last week")]:
        imp = [100 * (1 - ae[cand].loc[pk].sum() / ae[base].loc[pk].sum()) for pk in (rng.choice(days, len(days)) for _ in range(2000))]
        out.append({"slice": g, "days": len(days), "candidate": cand, "baseline": base, "improvement_%": round(100 * (1 - ae[cand].sum() / ae[base].sum()), 1),
                    "ci95_low": round(float(np.percentile(imp, 2.5)), 1), "ci95_high": round(float(np.percentile(imp, 97.5)), 1)})
bt = pd.DataFrame(out); bt.to_csv("weights_2023_bootstrap.csv", index=False, encoding="utf-8-sig")
print(bt[bt.slice.isin(["all slots", "ordinary (weekday+sat+sun)", "long weekend, all hours", "long weekend 08-18h", "lw_first 10-18h", "eve 16-23h", "holiday days (long weekend + eve)"])].to_string(index=False))

# ---- decision metric for the weighted models ----
DM = {"last week (lag 7d)": P["last week (lag 7d)"], "same holiday last time, else last week": P["same holiday last time, else last week"],
      "mean per slot x weekday (2024)": P["mean per slot x weekday (2024)"], "Ridge log, train 2024": P["Ridge log, train 2024"],
      "Ridge log, 2023+2024 uniform": P["Ridge log, 2023+2024 uniform"], W: P[W], WL: P[WL]}
dr = decision_regret(test, DM, 60)
dg = {"all": None, "ordinary (weekday+sat+sun)": ["weekday", "sat", "sun"], "long weekend days only": ["lw_first", "lw_mid", "lw_last"], "lw_first": ["lw_first"], "eve_of_long": ["eve_of_long"],
      "holiday days (long weekend + eve)": ["lw_first", "lw_mid", "lw_last", "eve_of_long"]}
rows, brow = [], []
for g, types in dg.items():
    sub = dr if types is None else dr[dr.daytype.isin(types)]
    rows.append({"group": g, "n_windows": len(sub), "n_days": sub.date.nunique(), "leave at window start": round(sub["leave at window start"].mean(), 2), **{k: round(sub[k].mean(), 2) for k in DM}})
    ps = sub.groupby("date")[["leave at window start"] + list(DM)].sum(); cnt = sub.groupby("date").size(); days = ps.index.values
    for cand in [W, WL]:
        for base in ["leave at window start", "last week (lag 7d)", "mean per slot x weekday (2024)", "same holiday last time, else last week", "Ridge log, train 2024"]:
            dif = [(ps[base].loc[pk].sum() - ps[cand].loc[pk].sum()) / cnt.loc[pk].sum() for pk in (rng.choice(days, len(days)) for _ in range(2000))]
            brow.append({"group": g, "candidate": cand, "baseline": base, "minutes_saved_per_window": round((ps[base].sum() - ps[cand].sum()) / cnt.sum(), 2),
                         "ci95_low": round(float(np.percentile(dif, 2.5)), 2), "ci95_high": round(float(np.percentile(dif, 97.5)), 2)})
dd = pd.DataFrame(rows); dd.to_csv("weights_2023_decision.csv", index=False, encoding="utf-8-sig")
db = pd.DataFrame(brow); db.to_csv("weights_2023_decision_bootstrap.csv", index=False, encoding="utf-8-sig")
print("\nDecision: minutes lost per 4-hour window vs the fastest slot"); print(dd.to_string(index=False))
print(db[db.candidate == W].to_string(index=False))
