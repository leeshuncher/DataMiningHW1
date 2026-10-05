"""(1) Peak-hour slices, (2) day-level bootstrap CIs for them, (3) bias of the log model after exp() and smearing corrections.
Train 2024, test 2025, hourly slots, eve modelled as the shared evening-bin effect. Usage: python peak_and_bias.py
Outputs: peak_results.csv (MAE / bias by slice), peak_bootstrap.csv (improvement % with 95% CI, whole DAYS resampled), bias_results.csv.
Free flow = median travel time of the night slots (00-05h) in the training year. Two kinds of peak slices:
 - ex ante (defined by calendar only, fair for evaluation): long-weekend daytime 08-18h, long-weekend morning 07-12h, lw_first 10-18h, Saturday 08-17h
 - ex post (defined by the actual travel time, so every model is penalised; shown as a stress test, not as a fair score):
   actual >= 1.25 x free flow, actual >= 1.5 x free flow."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from forecast_common import *

d = load_data(60); train, test = d[d.index.year == 2024], d[d.index.year == 2025]
FF = float(train[train.hod < 6].minutes.median()); print("free flow (night median, 2024):", round(FF, 2), "min; 1.25x =", round(1.25 * FF, 1), " 1.5x =", round(1.5 * FF, 1))
tr, te = add_cats(train, "shared"), add_cats(test, "shared")
P = {"last week (lag 7d)": test.lag7.values,
     "mean per slot x weekday": test[["slot", "dow"]].merge(train.groupby(["slot", "dow"]).minutes.mean().rename("m").reset_index(), on=["slot", "dow"], how="left").m.values}
lp, alpha, ins, oof = ridge_log_residuals(tr, te, CATS, [])
P["Ridge log"] = np.exp(lp)
g_ins = float(np.mean(np.exp(ins))); g_oof = float(np.nanmean(np.exp(oof)))
P["Ridge log + smearing (in-sample)"] = np.exp(lp) * g_ins
P["Ridge log + smearing (out-of-fold)"] = np.exp(lp) * g_oof
# group-wise smearing: the factor depends on the predicted level (high predicted level = peak-like), estimated from out-of-fold residuals of the training year
tr_pred_oof = np.exp(np.log(train.minutes.values) - oof)            # out-of-fold predictions for the training rows
hi_tr = tr_pred_oof >= 1.15 * FF; ok = ~np.isnan(oof)
f_hi, f_lo = float(np.mean(np.exp(oof[ok & hi_tr]))), float(np.mean(np.exp(oof[ok & ~hi_tr])))
hi_te = np.exp(lp) >= 1.15 * FF
P["Ridge log + smearing (by predicted level)"] = np.exp(lp) * np.where(hi_te, f_hi, f_lo)
P["Ridge L1"], _ = ridge_l1(tr, te, CATS, [])
P["GBM log"], _ = gbm_fit(train, test, False, True, "regression")
P["GBM L1"], _ = gbm_fit(train, test, False, False, "regression_l1")
print("smearing factors: in-sample", round(g_ins, 4), "| out-of-fold", round(g_oof, 4), "| OOF high-level", round(f_hi, 4), "low-level", round(f_lo, 4), "| ridge alpha", alpha)

y, dt, hod, dow = test.minutes.values, test.daytype.values, test.hod.values, test.dow.values
lw = np.isin(dt, ["lw_first", "lw_mid", "lw_last"])
groups = {"all slots": np.ones(len(test), bool),
          "long weekend, all hours": lw,
          "[ex ante] long weekend 08-18h": lw & (hod >= 8) & (hod < 18),
          "[ex ante] long weekend morning 07-12h": lw & (hod >= 7) & (hod < 12),
          "[ex ante] lw_first 10-18h": (dt == "lw_first") & (hod >= 10) & (hod < 18),
          "[ex ante] Saturday 08-17h": (dt == "sat") & (hod >= 8) & (hod < 17),
          "[ex ante] ordinary weekday 16-20h": (dt == "weekday") & (hod >= 16) & (hod < 20),
          "[ex post] actual >= 1.25 x free flow": y >= 1.25 * FF,
          "[ex post] actual >= 1.5 x free flow": y >= 1.5 * FF,
          "[ex post] long weekend, actual >= 1.25 x FF": lw & (y >= 1.25 * FF),
          "[ex post] long weekend, actual >= 1.5 x FF": lw & (y >= 1.5 * FF),
          "long weekend, night 00-06h + 20-24h": lw & ((hod < 6) | (hod >= 20))}
rows, brows = [], []
for g, m in groups.items():
    r = {"slice": g, "n": int(m.sum()), "days": int(pd.Series(test.index.normalize()[m]).nunique()), "mean actual": round(float(y[m].mean()), 1)}
    b = {"slice": g, "n": int(m.sum())}
    for k, p in P.items():
        r[k + " | MAE"] = round(float(np.abs(p[m] - y[m]).mean()), 2); b[k] = round(float((y[m] - p[m]).mean()), 2)
    rows.append(r); brows.append(b)
res = pd.DataFrame(rows); res.to_csv("peak_results.csv", index=False, encoding="utf-8-sig")
bias = pd.DataFrame(brows); bias.to_csv("bias_results.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 60)
mae_cols = ["slice", "n", "days", "mean actual"] + [k + " | MAE" for k in ["last week (lag 7d)", "mean per slot x weekday", "Ridge log", "Ridge log + smearing (by predicted level)", "Ridge L1", "GBM log", "GBM L1"]]
print("\nMAE (minutes)"); print(res[mae_cols].to_string(index=False))
print("\nMean residual = actual - predicted (positive = model too LOW)"); print(bias[["slice", "n", "last week (lag 7d)", "mean per slot x weekday", "Ridge log", "Ridge log + smearing (in-sample)", "Ridge log + smearing (out-of-fold)", "Ridge log + smearing (by predicted level)", "Ridge L1", "GBM log", "GBM L1"]].to_string(index=False))

# day-level bootstrap: resample whole days, improvement in total absolute error
rng = np.random.default_rng(0); day_id = test.index.normalize().values; out = []
for g, m in groups.items():
    days = np.unique(day_id[m]); ae = {k: pd.Series(np.abs(p[m] - y[m])).groupby(day_id[m]).sum() for k, p in P.items()}
    for cand in ["Ridge log", "Ridge log + smearing (by predicted level)", "Ridge L1", "GBM L1"]:
        for base in ["last week (lag 7d)", "mean per slot x weekday"]:
            imp = [100 * (1 - ae[cand].loc[pk].sum() / ae[base].loc[pk].sum()) for pk in (rng.choice(days, len(days)) for _ in range(2000))]
            out.append({"slice": g, "days": len(days), "candidate": cand, "baseline": base, "improvement_%": round(100 * (1 - ae[cand].sum() / ae[base].sum()), 1),
                        "ci95_low": round(float(np.percentile(imp, 2.5)), 1), "ci95_high": round(float(np.percentile(imp, 97.5)), 1)})
bt = pd.DataFrame(out); bt.to_csv("peak_bootstrap.csv", index=False, encoding="utf-8-sig")
print("\nDay-level bootstrap (Ridge log vs baselines)"); print(bt[bt.candidate == "Ridge log"].to_string(index=False))
