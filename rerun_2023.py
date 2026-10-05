"""Re-run the main experiments after adding 2023 data. Test year stays 2025 so results are comparable with GO_NOGO.md.
Scenarios: T24 = train 2024 only (earlier setting), T2324 = train 2023 + 2024. Alpha / GBM settings are chosen with TimeSeriesSplit inside the training years.
Models: baselines (last week, mean per slot x weekday, same holiday last time else last week), Ridge on log (eve = shared evening-bin effect, and the
other eve modes), Ridge + last-year-holiday feature, Ridge L1, GBM L1 / log (+ last-year feature).
Outputs: rerun2023_results.csv (MAE by slice), rerun2023_bias.csv (actual - predicted), rerun2023_bootstrap.csv, rerun2023_decision.csv, rerun2023_eve_days.csv.
Usage: python rerun_2023.py"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from forecast_common import *

SLOT = 60
d = load_data(SLOT); test = d[d.index.year == 2025]
print("rows by year:", d.groupby(d.index.year).size().to_dict())
d["ly_f"] = d.ly.fillna(d.lag4mean); d["ly_ok"] = d.ly.notna().astype(float) * 30.0   # x30 because ridge_fit divides numeric columns by 30
test = d[d.index.year == 2025]

def models(train_years, tag):
    train = d[d.index.year.isin(train_years)]
    P = {}
    P["last week (lag 7d)"] = test.lag7.values
    P["mean per slot x weekday"] = test[["slot", "dow"]].merge(train.groupby(["slot", "dow"]).minutes.mean().rename("m").reset_index(), on=["slot", "dow"], how="left").m.values
    P["same holiday last time, else last week"] = np.where(test.ly.notna(), test.ly, test.lag7)
    for mode, label in [("shared", "Ridge log"), ("separate", "Ridge log [eve=own type]"), ("none", "Ridge log [eve ignored]"), ("by_dow", "Ridge log [eve x bin x Fri/other]")]:
        tr, te = add_cats(train, mode), add_cats(test, mode)
        P[f"{tag} {label}"], _ = ridge_fit(tr, te, CATS, [], True)
        if mode == "shared":
            P[f"{tag} Ridge log +ly"], _ = ridge_fit(tr, te, CATS, ["ly_f", "ly_ok"], True)
            P[f"{tag} Ridge L1"], _ = ridge_l1(tr, te, CATS, [])
    P[f"{tag} GBM log"], _ = gbm_fit(train, test, False, True, "regression")
    P[f"{tag} GBM L1"], _ = gbm_fit(train, test, False, False, "regression_l1")
    return {k if k in ("last week (lag 7d)", "mean per slot x weekday", "same holiday last time, else last week") else k: v for k, v in P.items()}

P = {}
P.update(models([2024], "T24")); P24 = dict(P)
Pn = models([2023, 2024], "T2324"); P.update(Pn)
for k in ["last week (lag 7d)"]: P[k] = P24[k]
# 'mean per slot x weekday' differs by training set: keep the T2324 version under its own name
tr2 = d[d.index.year.isin([2023, 2024])]
P["mean per slot x weekday (2023+24)"] = test[["slot", "dow"]].merge(tr2.groupby(["slot", "dow"]).minutes.mean().rename("m").reset_index(), on=["slot", "dow"], how="left").m.values
P["mean per slot x weekday"] = P24["mean per slot x weekday"]

y, dt, hod, dow = test.minutes.values, test.daytype.values, test.hod.values, test.dow.values
lw = np.isin(dt, ["lw_first", "lw_mid", "lw_last"]); eve = dt == "eve_of_long"
groups = {"all slots": np.ones(len(test), bool), "ordinary (weekday+sat+sun)": np.isin(dt, ["weekday", "sat", "sun"]),
          "holiday days (long weekend + eve)": lw | eve, "long weekend, all hours": lw,
          "long weekend 08-18h": lw & (hod >= 8) & (hod < 18), "long weekend morning 07-12h": lw & (hod >= 7) & (hod < 12),
          "lw_first, all hours": dt == "lw_first", "lw_first 10-18h": (dt == "lw_first") & (hod >= 10) & (hod < 18),
          "lw_mid": dt == "lw_mid", "lw_last": dt == "lw_last", "eve, all hours": eve, "eve 16-23h": eve & (hod >= 16),
          "eve, Friday 16-23h": eve & (dow == 4) & (hod >= 16), "eve, other weekday 16-23h": eve & (dow != 4) & (hod >= 16),
          "Saturday 08-17h": (dt == "sat") & (hod >= 8) & (hod < 17), "long weekend night 0-6h + 20-24h": lw & ((hod < 6) | (hod >= 20)),
          "holiday days WITH last-year value": (lw | eve) & test.ly.notna().values, "holiday days WITHOUT last-year value": (lw | eve) & test.ly.isna().values}
rows, brows = [], []
for g, m in groups.items():
    r = {"slice": g, "n": int(m.sum()), "days": int(pd.Series(test.index.normalize()[m]).nunique()), "mean actual": round(float(y[m].mean()), 1)}
    b = {"slice": g, "n": int(m.sum())}
    for k, p in P.items(): r[k] = round(float(np.abs(p[m] - y[m]).mean()), 2); b[k] = round(float((y[m] - p[m]).mean()), 2)
    rows.append(r); brows.append(b)
res = pd.DataFrame(rows); res.to_csv("rerun2023_results.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(brows).to_csv("rerun2023_bias.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 60)
cmp = ["slice", "n", "days", "mean actual", "last week (lag 7d)", "mean per slot x weekday (2023+24)", "same holiday last time, else last week",
       "T24 Ridge log", "T2324 Ridge log", "T2324 Ridge log +ly", "T2324 Ridge L1", "T2324 GBM log", "T2324 GBM L1"]
print("MAE, test 2025 (T24 = train 2024, T2324 = train 2023+2024)"); print(res[cmp].to_string(index=False))
ev = ["slice", "n", "T24 Ridge log", "T24 Ridge log [eve=own type]", "T24 Ridge log [eve ignored]", "T24 Ridge log [eve x bin x Fri/other]",
      "T2324 Ridge log", "T2324 Ridge log [eve=own type]", "T2324 Ridge log [eve ignored]", "T2324 Ridge log [eve x bin x Fri/other]"]
print("\nEve modes"); print(res[res.slice.isin(["eve, all hours", "eve 16-23h", "eve, Friday 16-23h", "eve, other weekday 16-23h", "long weekend, all hours", "ordinary (weekday+sat+sun)"])][ev].to_string(index=False))
print("\nMean residual actual - predicted (positive = model too low)")
bb = pd.DataFrame(brows); print(bb[bb.slice.isin(["all slots", "long weekend, all hours", "long weekend 08-18h", "lw_first 10-18h", "eve 16-23h", "Saturday 08-17h"])][["slice", "n", "T24 Ridge log", "T2324 Ridge log", "T2324 Ridge log +ly", "T2324 Ridge L1", "T2324 GBM log", "T2324 GBM L1"]].to_string(index=False))

# day-level bootstrap
rng = np.random.default_rng(0); day_id = test.index.normalize().values; out = []
pairs = [("T2324 Ridge log", "last week (lag 7d)"), ("T2324 Ridge log", "mean per slot x weekday (2023+24)"), ("T2324 Ridge log", "same holiday last time, else last week"),
         ("T2324 Ridge log", "T24 Ridge log"), ("T2324 Ridge log +ly", "T2324 Ridge log"), ("T2324 GBM L1", "T2324 Ridge log"), ("T2324 GBM log", "T2324 Ridge log")]
for g, m in groups.items():
    if m.sum() == 0: continue
    days = np.unique(day_id[m]); ae = {k: pd.Series(np.abs(p[m] - y[m])).groupby(day_id[m]).sum() for k, p in P.items()}
    for cand, base in pairs:
        imp = [100 * (1 - ae[cand].loc[pk].sum() / ae[base].loc[pk].sum()) for pk in (rng.choice(days, len(days)) for _ in range(2000))]
        out.append({"slice": g, "days": len(days), "candidate": cand, "baseline": base, "improvement_%": round(100 * (1 - ae[cand].sum() / ae[base].sum()), 1),
                    "ci95_low": round(float(np.percentile(imp, 2.5)), 1), "ci95_high": round(float(np.percentile(imp, 97.5)), 1)})
bt = pd.DataFrame(out); bt.to_csv("rerun2023_bootstrap.csv", index=False, encoding="utf-8-sig")
print("\nBootstrap (whole days resampled)")
print(bt[bt.slice.isin(["long weekend, all hours", "long weekend 08-18h", "lw_first 10-18h", "eve 16-23h", "ordinary (weekday+sat+sun)", "holiday days (long weekend + eve)", "all slots"])].to_string(index=False))

# decision metric
DM = ["last week (lag 7d)", "mean per slot x weekday (2023+24)", "same holiday last time, else last week", "T24 Ridge log", "T2324 Ridge log", "T2324 Ridge log +ly", "T2324 Ridge L1", "T2324 GBM log", "T2324 GBM L1"]
dr = decision_regret(test, {k: P[k] for k in DM}, SLOT)
dg = {"all": None, "ordinary (weekday+sat+sun)": ["weekday", "sat", "sun"], "holiday days (long weekend + eve)": ["lw_first", "lw_mid", "lw_last", "eve_of_long"],
      "long weekend days only": ["lw_first", "lw_mid", "lw_last"], "lw_first": ["lw_first"], "eve_of_long": ["eve_of_long"]}
dd = []
for g, types in dg.items():
    sub = dr if types is None else dr[dr.daytype.isin(types)]
    dd.append({"group": g, "n_windows": len(sub), "n_days": sub.date.nunique(), "leave at window start": round(sub["leave at window start"].mean(), 2), **{k: round(sub[k].mean(), 2) for k in DM}})
dd = pd.DataFrame(dd); dd.to_csv("rerun2023_decision.csv", index=False, encoding="utf-8-sig")
print("\nDecision: minutes lost per 4-hour window vs the fastest slot"); print(dd.to_string(index=False))
# per-eve-day detail
det = []
for date in sorted(set(test.index.normalize()[eve])):
    m = (test.index.normalize() == date) & (hod >= 16)
    det.append({"date": date.date(), "weekday": "一二三四五六日"[date.dayofweek], "actual 16-23h": y[m].mean(), "T24 Ridge log": P["T24 Ridge log"][m].mean(),
                "T2324 Ridge log": P["T2324 Ridge log"][m].mean(), "T2324 Ridge log +ly": P["T2324 Ridge log +ly"][m].mean(), "T2324 eve ignored": P["T2324 Ridge log [eve ignored]"][m].mean(),
                "same holiday last time": np.nanmean(np.where(test.ly.notna().values[m], test.ly.values[m], np.nan)) if test.ly.notna().values[m].any() else np.nan})
pd.DataFrame(det).round(1).to_csv("rerun2023_eve_days.csv", index=False, encoding="utf-8-sig"); print("\n", pd.DataFrame(det).round(1).to_string(index=False))
