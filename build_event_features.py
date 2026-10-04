"""Daily event features from the OFFICIAL schedule (not from the flow-based detector, which would leak the target)."""
import pandas as pd
from event_utils import kind

DEFAULT_DURATION_H = 3.0  # used only when a page lists a start time but no end time

def hm(s): h, m = s.split(":"); return int(h) + int(m) / 60

off = pd.read_csv("events_official.csv", parse_dates=["date"])            # cancelled dates already removed
ses = pd.read_csv("events_sessions.csv", parse_dates=["date"]).fillna("")
ses = ses.merge(off[["date", "raw", "title"]], left_on=["date", "title_raw"], right_on=["date", "raw"], how="inner")
ses["type"] = ses["title"].map(kind)
ses["start_h"] = ses["start"].map(lambda s: hm(s) if s else None)
ses["end_h"] = ses["end"].map(lambda s: hm(s) if s else None)
ses.loc[ses.end_h < ses.start_h, "end_h"] += 24                            # ends after midnight
ses["end_imputed"] = ses.end_h.isna() & ses.start_h.notna()
ses.loc[ses.end_imputed, "end_h"] = ses.start_h + DEFAULT_DURATION_H

day = off.groupby("date").agg(n_events=("raw", "nunique")).join(
    ses.groupby("date").agg(
        n_sessions=("start", "size"),
        first_start_h=("start_h", "min"), last_start_h=("start_h", "max"),
        last_end_h=("end_h", "max"),
        end_imputed=("end_imputed", "max"),
    ))
types = off.assign(t=off["title"].map(kind)).groupby("date")["t"].agg(set)
day["is_concert"] = types.map(lambda s: "演唱會/音樂類" in s)
day["is_sport"] = types.map(lambda s: "體育" in s)
day["is_other_event"] = types.map(lambda s: "其他" in s)
day["n_sessions"] = day["n_sessions"].fillna(0).astype(int)
day["end_imputed"] = day["end_imputed"].fillna(False).astype(bool)
day.to_csv("events_daily.csv", encoding="utf-8-sig")
print(day.shape, "with end time:", day.last_end_h.notna().sum(), "imputed:", day.end_imputed.sum())
