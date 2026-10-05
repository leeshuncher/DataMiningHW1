"""Exploration that uses TRAINING YEARS ONLY (2023-2024). The test year 2025 is never looked at here.
Outputs train_exploration_*.csv: share of slots > 40 min and peak hours by day type; eve-of-long-weekend versus ordinary weekdays;
by year (2023 vs 2024) to document the drift. Usage: python explore_train.py"""
import numpy as np, pandas as pd
pd.set_option("display.width", 220)
cal = pd.read_csv("data/calendar.csv", parse_dates=["date"])
t = pd.read_csv("data/travel_60min.csv", parse_dates=["depart"]); t["date"] = t.depart.dt.normalize(); t["hour"] = t.depart.dt.hour
t = t[t.depart.dt.year <= 2024].merge(cal[["date", "daytype", "dow"]], on="date"); t["year"] = t.depart.dt.year
FF = float(t[t.hour < 6].minutes.median()); print("free flow (night median, 2023-24):", round(FF, 1))
ty = ["weekday", "sat", "sun", "lw_first", "lw_mid", "lw_last", "eve_of_long"]
share = t[t.daytype.isin(ty)].assign(c=lambda d: d.minutes > 40).groupby("daytype").agg(days=("date", "nunique"), share_gt40=("c", "mean"), mean=("minutes", "mean")).loc[ty].round(3)
share.to_csv("train_exploration_share.csv", encoding="utf-8-sig"); print("2023-2024 (training years)"); print(share.to_string())
sy = t[t.daytype.isin(["sat", "weekday"])].assign(c=lambda d: d.minutes > 40).groupby(["daytype", "year"]).c.mean().unstack().round(3); print("by year"); print(sy.to_string())
p = t[t.daytype.isin(ty)].pivot_table(index="hour", columns="daytype", values="minutes", aggfunc="mean").round(1)[ty]
p.to_csv("train_exploration_hourly.csv", encoding="utf-8-sig"); print(p.loc[[0, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 23]].to_string())
peak = {k: (int(p[k].idxmax()), float(p[k].max())) for k in ty}; print("peak hour & level:", peak)
fri = t[(t.dow == 4) & (t.daytype == "weekday")].groupby("hour").minutes.mean().rename("ordinary Friday")
ev = t[t.daytype == "eve_of_long"].assign(fri=lambda d: d.dow == 4)
e = pd.concat([fri, ev[ev.fri].groupby("hour").minutes.mean().rename("eve Friday"), ev[~ev.fri].groupby("hour").minutes.mean().rename("eve other weekday"),
               t[(t.dow.isin([1, 2, 3])) & (t.daytype == "weekday")].groupby("hour").minutes.mean().rename("ordinary Tue-Thu")], axis=1).round(1)
e.to_csv("train_exploration_eve.csv", encoding="utf-8-sig"); print("eve days in training:", ev.groupby("fri").date.nunique().to_dict()); print(e.loc[12:23].to_string())
