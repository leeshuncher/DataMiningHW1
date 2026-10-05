"""The single largest hourly errors of the selected model on the test year (from final2_predictions.csv), with context and a per-segment breakdown
from the 5-minute M04A data: which of the four segments (Nangang System -> ... -> 43.9 km) carried the delay.
Output: error_cases.csv and printed text. Usage: python src/error_cases.py (after final_compact.py)"""
import numpy as np, pandas as pd
CHAIN = ["05F0000S", "05F0055S", "05F0287S", "05F0309S"]; NAMES = {"05F0000S": "0.0-5.5km", "05F0055S": "5.5-28.7km (tunnel section)", "05F0287S": "28.7-30.9km", "05F0309S": "30.9-43.9km"}
p = pd.read_csv("results/final2_predictions.csv", parse_dates=["depart"]); cal = pd.read_csv("data/calendar.csv", parse_dates=["date"]).set_index("date")
p["err"] = p["Ours: compact linear (median regression)"] - p.actual; p["abs_err"] = p.err.abs(); p["date"] = p.depart.dt.normalize()
top = p.sort_values("abs_err", ascending=False).drop_duplicates("date").head(3)       # three different days
rows = []
for _, r in top.iterrows():
    f = pd.read_csv(f"data/m04a/M04A_{r.date:%Y%m%d}.csv", header=None, names=["t", "a", "b", "veh", "sec", "n"]); f = f[(f.veh == 31) & f.a.isin(CHAIN) & (f.sec > 0) & (f.n >= 1)]
    f["t"] = pd.to_datetime(f.t, format="%Y/%m/%d %H:%M"); f["hour"] = f.t.dt.hour; seg = f[f.hour == r.depart.hour].groupby("a").sec.mean().div(60).round(1)
    prev = f[f.hour == r.depart.hour].groupby("a").sec.mean()
    day_alt = p[p.date == r.date]
    rows.append({"date": r.date.date(), "weekday": "一二三四五六日"[r.date.dayofweek], "hour": r.depart.hour, "daytype": r.daytype, "holiday": str(cal.loc[r.date, "description"]) if r.date in cal.index else "",
                 "actual": round(r.actual, 1), "predicted": round(r["Ours: compact linear (median regression)"], 1), "error": round(r.err, 1), "last week": round(r["Persistence-type: last week same slot"], 1), "same holiday last time": round(r["Same holiday last time (else last week)"], 1),
                 "day max actual": round(day_alt.actual.max(), 1), "day max actual hour": int(day_alt.loc[day_alt.actual.idxmax(), "depart"].hour),
                 **{f"seg {NAMES[k]} (min)": float(seg.get(k, np.nan)) for k in CHAIN}})
ec = pd.DataFrame(rows); ec.to_csv("results/error_cases.csv", index=False, encoding="utf-8-sig"); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); print(ec.T.to_string())
