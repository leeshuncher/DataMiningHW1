"""(4) Eve of a long weekend: three ways to model it, and (5) a gradient-boosting baseline next to the linear model.
Forward check (the real one): train 2024, test 2025. Reverse check (NOT chronological, only more evidence for the eve days): train 2025, test 2024.
All models see the same rows. Ridge alpha / LightGBM settings are chosen with TimeSeriesSplit inside the training year only.
Outputs: eve_gbm_results[_reverse].csv (MAE by slice), eve_days_detail.csv (per eve day), eve_gbm_decision.csv, eve_gbm_bootstrap.csv.
Usage: python eve_and_gbm.py"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from forecast_common import *

SLOT = 60
def run(train_year, test_year, tag):
    d = load_data(SLOT)
    train, test = d[d.index.year == train_year], d[d.index.year == test_year]
    P, info = {}, {}
    P["last week (lag 7d)"] = test.lag7.values
    P["mean per slot x weekday"] = test[["slot", "dow"]].merge(train.groupby(["slot", "dow"]).minutes.mean().rename("m").reset_index(), on=["slot", "dow"], how="left").m.values
    for mode, label in [("separate", "Ridge eve=own type (old)"), ("none", "Ridge eve=ignored"), ("shared", "Ridge eve x evening bin"), ("by_dow", "Ridge eve x bin x Fri/other")]:
        tr, te = add_cats(train, mode), add_cats(test, mode)
        P[label], info[label] = ridge_fit(tr, te, CATS, [], True)
        P[label + " +lags"], info[label + " +lags"] = ridge_fit(tr, te, CATS, ["lag7", "lag4mean"], True)
    for lags in (False, True):
        sfx = " +lags" if lags else ""
        P["GBM L1" + sfx], info["GBM L1" + sfx] = gbm_fit(train, test, lags, False, "regression_l1")
        P["GBM log" + sfx], info["GBM log" + sfx] = gbm_fit(train, test, lags, True, "regression")
    y, dt, dow, hod = test.minutes.values, test.daytype.values, test.dow.values, test.hod.values
    eve = dt == "eve_of_long"
    groups = {"all": np.ones(len(test), bool), "ordinary (weekday+sat+sun)": np.isin(dt, ["weekday", "sat", "sun"]),
              "holiday days (long weekend + eve)": np.isin(dt, ["lw_first", "lw_mid", "lw_last", "eve_of_long"]),
              "long weekend days only": np.isin(dt, ["lw_first", "lw_mid", "lw_last"]), "eve_of_long (all hours)": eve,
              "eve_of_long 16-23h": eve & (hod >= 16), "eve, Friday": eve & (dow == 4), "eve, other weekday": eve & (dow != 4),
              "eve, Friday 16-23h": eve & (dow == 4) & (hod >= 16), "eve, other weekday 16-23h": eve & (dow != 4) & (hod >= 16),
              "lw_first": dt == "lw_first", "sat": dt == "sat", "weekday": dt == "weekday"}
    rows = []
    for g, m in groups.items():
        r = {"slice": g, "n": int(m.sum()), "days": int(pd.Series(test.index.normalize()[m]).nunique())}
        for k, p in P.items(): r[k] = round(float(np.abs(p[m] - y[m]).mean()), 2)
        rows.append(r)
    res = pd.DataFrame(rows); res.to_csv(f"archive/results/eve_gbm_results{tag}.csv", index=False, encoding="utf-8-sig")
    return train, test, P, info, groups, res

train, test, P, info, groups, res = run(2024, 2025, "")
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
print("train", len(train), "test", len(test), "eve days train/test:", train[train.daytype == "eve_of_long"].index.normalize().nunique(), test[test.daytype == "eve_of_long"].index.normalize().nunique())
cal_cols = ["slice", "n", "days", "last week (lag 7d)", "mean per slot x weekday", "Ridge eve=own type (old)", "Ridge eve=ignored", "Ridge eve x evening bin", "Ridge eve x bin x Fri/other", "GBM L1", "GBM log"]
print(res[cal_cols].to_string(index=False))
lag_cols = ["slice", "Ridge eve=own type (old) +lags", "Ridge eve=ignored +lags", "Ridge eve x evening bin +lags", "Ridge eve x bin x Fri/other +lags", "GBM L1 +lags", "GBM log +lags"]
print(res[lag_cols].to_string(index=False)); print({k: v for k, v in info.items() if k.startswith("GBM")})
_, _, Pr, _, _, resr = run(2025, 2024, "_reverse")
print("REVERSE (train 2025, test 2024; not chronological)"); print(resr[cal_cols].to_string(index=False))

# per-eve-day detail for the error analysis
y = test.minutes.values; day = test.index.normalize()
det = []
for date in sorted(set(day[test.daytype.values == "eve_of_long"])):
    m = (day == date) & (test.hod.values >= 16)
    r = {"date": date.date(), "weekday": "一二三四五六日"[date.dayofweek], "eve_of": test.eve_len[m].iloc[0], "actual 16-23h mean": y[m].mean()}
    for k in ["last week (lag 7d)", "mean per slot x weekday", "Ridge eve=own type (old)", "Ridge eve=ignored", "Ridge eve x evening bin", "Ridge eve x bin x Fri/other", "GBM log"]:
        r[k] = P[k][m].mean()
    det.append(r)
det = pd.DataFrame(det).round(1); det.to_csv("archive/results/eve_days_detail.csv", index=False, encoding="utf-8-sig"); print(det.to_string(index=False))

# decision metric and bootstrap
DM = ["last week (lag 7d)", "mean per slot x weekday", "Ridge eve=own type (old)", "Ridge eve x evening bin", "Ridge eve x bin x Fri/other", "GBM L1", "GBM log"]
dr = decision_regret(test, {k: P[k] for k in DM}, SLOT)
dg = {"all": None, "ordinary (weekday+sat+sun)": ["weekday", "sat", "sun"], "holiday days (long weekend + eve)": ["lw_first", "lw_mid", "lw_last", "eve_of_long"],
      "long weekend days only": ["lw_first", "lw_mid", "lw_last"], "eve_of_long": ["eve_of_long"]}
out = []
for g, types in dg.items():
    sub = dr if types is None else dr[dr.daytype.isin(types)]
    out.append({"group": g, "n_windows": len(sub), "n_days": sub.date.nunique(), "leave at window start": round(sub["leave at window start"].mean(), 2), **{k: round(sub[k].mean(), 2) for k in DM}})
dec = pd.DataFrame(out); dec.to_csv("archive/results/eve_gbm_decision.csv", index=False, encoding="utf-8-sig"); print(dec.to_string(index=False))
rng = np.random.default_rng(0); b = []
for g, types in dg.items():
    sub = dr if types is None else dr[dr.daytype.isin(types)]; days = sub.date.unique()
    ps = sub.groupby("date")[DM].sum(); cnt = sub.groupby("date").size()
    for cand, base in [("GBM log", "Ridge eve x evening bin"), ("GBM L1", "Ridge eve x evening bin"), ("Ridge eve x evening bin", "Ridge eve=own type (old)")]:
        diff = [(ps[base].loc[pk].sum() - ps[cand].loc[pk].sum()) / cnt.loc[pk].sum() for pk in (rng.choice(days, len(days)) for _ in range(2000))]
        b.append({"group": g, "candidate": cand, "baseline": base, "minutes_saved_per_window": round((ps[base].sum() - ps[cand].sum()) / cnt.sum(), 3),
                  "ci95_low": round(float(np.percentile(diff, 2.5)), 3), "ci95_high": round(float(np.percentile(diff, 97.5)), 3)})
bt = pd.DataFrame(b); bt.to_csv("archive/results/eve_gbm_bootstrap.csv", index=False, encoding="utf-8-sig"); print(bt.to_string(index=False))
