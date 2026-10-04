"""Flag candidate event days at 台北小巨蛋 from evening entry spikes vs. a local same-hour baseline."""
import pandas as pd, numpy as np

e = pd.read_csv("arena_entries_hourly.csv", encoding="utf-8-sig")
e["dt"] = pd.to_datetime(e["date"]) + pd.to_timedelta(e["hour"], unit="h")
grid = pd.date_range("2017-01-01", "2026-08-31 23:00", freq="h")
s = e.set_index("dt")["entries"].reindex(grid).fillna(0)
cal = pd.read_csv("external/calendar.csv", encoding="utf-8-sig", parse_dates=["date"]).set_index("date")
d = pd.DataFrame({"entries": s})
d["date"] = d.index.normalize(); d["hour"] = d.index.hour
d["offday"] = d["date"].map(cal["is_holiday"]).fillna(False).astype(bool)  # weekend/holiday vs workday

# baseline: median of same hour & same day type within +-28 days, excluding the day itself
tab = d.pivot_table(index="date", columns="hour", values="entries")
off = d.drop_duplicates("date").set_index("date")["offday"]
base = pd.DataFrame(index=tab.index, columns=tab.columns, dtype=float)
for typ in (True, False):
    sub = tab[off.reindex(tab.index) == typ]
    # rolling median over +-28 days (time-based window), then self is included -> use median of neighbours via shift trick
    r = sub.rolling("57D", center=True, min_periods=5).median()
    base.loc[sub.index] = r
exc = tab - base
ev = pd.DataFrame({
    "peak_hour": exc.loc[:, 16:23].idxmax(axis=1),
    "peak_excess": exc.loc[:, 16:23].max(axis=1),
    "evening_excess": exc.loc[:, 16:23].clip(lower=0).sum(axis=1),
})
ev["peak_entries"] = [tab.loc[i, h] for i, h in ev["peak_hour"].items()]
ev["baseline"] = [base.loc[i, h] for i, h in ev["peak_hour"].items()]
ev["ratio"] = ev["peak_entries"] / ev["baseline"].clip(lower=50)
ev["dow"] = ev.index.dayofweek; ev["is_offday"] = off.reindex(ev.index)
ev["holiday_name"] = cal["description"].reindex(ev.index)
ev.round(1).to_csv("events_all_days.csv", encoding="utf-8-sig")
print(ev["peak_excess"].describe(percentiles=[.5,.8,.9,.95,.97,.99]))
