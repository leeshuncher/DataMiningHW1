"""Daily event features from the OFFICIAL schedule (not from the flow-based detector, which would leak the target)."""
import re, statistics
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

# ticket prices from the official page (known before the event): per-title stats, aggregated by date below
def prices(raw):
    raw = re.sub(r"\d{4}/\d{1,2}/\d{1,2}|\d{1,2}/\d{1,2}", " ", str(raw))     # drop dates such as 2017/6/30
    raw = re.sub(r"\([^)]*\)|（[^）]*）", " ", raw)                                 # drop notes such as (VIP Package)
    return [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", raw) if int(x.replace(",", "")) >= 100]
tk = pd.read_csv("events_tickets.csv", encoding="utf-8-sig").fillna("")
tk["p"] = tk["price_raw"].map(prices)
tk["price_median"] = tk.p.map(lambda p: statistics.median(p) if p else None)  # robust to VIP/package prices
tk["price_min"] = tk.p.map(lambda p: min(p) if p else None)
tk["price_mean"] = tk.p.map(lambda p: sum(p) / len(p) if p else None)
tk["n_price_tiers"] = tk.p.map(len)
off = off.merge(tk[["title_raw", "price_median", "price_min", "price_mean", "n_price_tiers"]].drop_duplicates("title_raw"),
                left_on="raw", right_on="title_raw", how="left")

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
day = day.join(off.groupby("date").agg(price_median=("price_median", "max"), price_min=("price_min", "min"),
                                       price_mean=("price_mean", "mean"), n_price_tiers=("n_price_tiers", "max")))
day["n_sessions"] = day["n_sessions"].fillna(0).astype(int)
day["end_imputed"] = day["end_imputed"].fillna(False).astype(bool)
day.to_csv("events_daily.csv", encoding="utf-8-sig")
print(day.shape, "with end time:", day.last_end_h.notna().sum(), "imputed:", day.end_imputed.sum())
print("with price:", day.price_median.notna().sum(), "of", len(day))
