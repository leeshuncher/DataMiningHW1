"""Build the modelling table: hourly entries at 台北小巨蛋 + weather + calendar + lag features."""
import pandas as pd

# full hourly grid; hours with no row in the OD data (no service) become 0
ent = pd.read_csv("arena_entries_hourly.csv", encoding="utf-8-sig")
ent["datetime"] = pd.to_datetime(ent["date"]) + pd.to_timedelta(ent["hour"], unit="h")
ent = ent.set_index("datetime")["entries"]
grid = pd.date_range("2017-01-01 00:00", "2026-08-31 23:00", freq="h", name="datetime")
df = pd.DataFrame({"entries": ent.reindex(grid)})
df["has_record"] = df["entries"].notna()
df["entries"] = df["entries"].fillna(0).astype(int)

# time features
df["date"] = df.index.normalize()
df["hour"] = df.index.hour
df["dow"] = df.index.dayofweek  # 0 = Monday
df["month"] = df.index.month

# calendar
cal = pd.read_csv("external/calendar.csv", encoding="utf-8-sig", parse_dates=["date"]).set_index("date")
cal["is_makeup_workday"] = cal["description"].fillna("").str.contains("補行上班")
cal["is_day_before_holiday"] = cal["is_holiday"].shift(-1, fill_value=False) & ~cal["is_holiday"]
cal["is_day_after_holiday"] = cal["is_holiday"].shift(1, fill_value=False) & ~cal["is_holiday"]
cal = cal.rename(columns={"description": "holiday_name"}).drop(columns="weekday")
df = df.join(cal, on="date")
df["is_holiday"] = df["is_holiday"].astype(bool)

# weather: actual only (ERA5, 2017+); archived forecasts in external/ are deliberately not used
w = pd.read_csv("external/weather_actual.csv", parse_dates=["datetime"]).set_index("datetime")
df = df.join(w.add_suffix("_act"))

# lag features: only values that are known when predicting a day ahead (>= 7 days back)
for label, h in [("lag_7d", 168), ("lag_14d", 336), ("lag_21d", 504), ("lag_364d", 364 * 24)]:
    df[label] = df["entries"].shift(h)
df["lag_mean_4w"] = df[["lag_7d", "lag_14d", "lag_21d"]].join(df["entries"].shift(504 + 168)).mean(axis=1)

# event features from the official schedule (known in advance)
ev = pd.read_csv("events_daily.csv", encoding="utf-8-sig", parse_dates=["date"]).set_index("date")
df = df.join(ev, on="date")
df["event_today"] = df["n_events"].notna()
for c in ["n_events", "n_sessions"]: df[c] = df[c].fillna(0).astype(int)
for c in ["is_concert", "is_sport", "is_other_event", "end_imputed"]: df[c] = df[c].fillna(False).astype(bool)
df["has_event_time"] = df["last_end_h"].notna()
# hours relative to the last scheduled end / first start (0 = the hour the event ends / starts)
df["rel_end_h"] = (df["hour"] - df["last_end_h"].floordiv(1)).clip(-12, 6)
df["rel_start_h"] = (df["hour"] - df["first_start_h"].floordiv(1)).clip(-12, 12)
df["is_end_hour"] = df["rel_end_h"] == 0
df["is_post_end_hour"] = df["rel_end_h"] == 1

# exclude 2017 (lags incomplete) and 2020-2022 (COVID); done after the lags so 2023+ lags still see 2022 values
df = df[~df.index.year.isin([2017, 2020, 2021, 2022])]
# past-event size features: entries in the end window (end hour-1 .. end hour+2) of earlier event days.
# Only event days >= 7 days before the target day are used (same rule as the lags); COVID years are already dropped.
win = {}
for d, g in df[df["has_event_time"]].groupby("date"):
    e = int(g["last_end_h"].iloc[0] // 1)
    win[d] = (g.loc[g["hour"].between(e - 1, e + 2), "entries"].sum(), "concert" if g["is_concert"].iloc[0] else "sport" if g["is_sport"].iloc[0] else "other")
wd = pd.DataFrame(win, index=["w", "t"]).T.sort_index(); wd["w"] = wd["w"].astype(float)
def past_feats(d):
    p = wd[wd.index <= d - pd.Timedelta(days=7)]
    if p.empty: return pd.Series({"prev_event_win": None, "prev3_event_win": None, "prev_type_win": None})
    t = wd.loc[d, "t"] if d in wd.index else None
    pt = p[p.t == t].w.tail(3) if t else p.w.tail(0)
    return pd.Series({"prev_event_win": p.w.iloc[-1], "prev3_event_win": p.w.tail(3).mean(), "prev_type_win": pt.mean() if len(pt) else None})
pf = pd.DataFrame({d: past_feats(d) for d in df.loc[df["event_today"], "date"].unique()}).T
pf.index = pd.to_datetime(pf.index)
df = df.join(pf, on="date")
df = df.drop(columns="date")
df.to_parquet("master.parquet")
df.to_csv("master.csv", encoding="utf-8-sig")
print(df.shape)
print(df.isna().sum()[lambda s: s > 0])
