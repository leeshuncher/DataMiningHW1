"""Taiwan holidays and make-up workdays (ruyut/TaiwanCalendar) -> data/calendar.csv with the structure of long weekends.
Columns: date, is_holiday (incl. weekends), description, is_makeup_workday, block_len, block_pos, daytype.
A block = run of consecutive non-working days. Long weekend = block of >= 3 days. daytype is one of
weekday, sat, sun, eve_of_long (workday just before a long block), lw_first, lw_mid, lw_last, single_holiday (1-2 day block that is
not a plain weekend), makeup_workday. Usage: python src/fetch_calendar.py"""
import json, subprocess
import pandas as pd

rows = []
for y in (2023, 2024, 2025, 2026):
    url = f"https://raw.githubusercontent.com/ruyut/TaiwanCalendar/master/data/{y}.json"
    rows += json.loads(subprocess.run(["curl", "-sS", "-f", "-m", "60", url], check=True, capture_output=True).stdout)
c = pd.DataFrame(rows)
c["date"] = pd.to_datetime(c["date"], format="%Y%m%d")
c = c.sort_values("date").reset_index(drop=True).rename(columns={"isHoliday": "is_holiday", "description": "description"})
c["dow"] = c.date.dt.dayofweek
c["is_makeup_workday"] = (~c.is_holiday) & c.dow.isin([5, 6])
grp = (c.is_holiday != c.is_holiday.shift()).cumsum()
c["block_len"] = c.groupby(grp).date.transform("size").where(c.is_holiday, 0)
c["block_pos"] = c.groupby(grp).cumcount().where(c.is_holiday, -1)
long_ = c.is_holiday & (c.block_len >= 3)
first = long_ & (c.block_pos == 0); last = long_ & (c.block_pos == c.block_len - 1)
c["daytype"] = "weekday"
c.loc[c.dow == 5, "daytype"] = "sat"; c.loc[c.dow == 6, "daytype"] = "sun"
c.loc[c.is_holiday & (c.block_len < 3) & ~c.dow.isin([5, 6]), "daytype"] = "single_holiday"
c.loc[c.is_holiday & (c.block_len < 3) & c.dow.isin([5, 6]) & c.description.ne(""), "daytype"] = "single_holiday"
c.loc[long_, "daytype"] = "lw_mid"; c.loc[first, "daytype"] = "lw_first"; c.loc[last, "daytype"] = "lw_last"
c.loc[c.is_makeup_workday, "daytype"] = "makeup_workday"
eve = (~c.is_holiday) & first.shift(-1, fill_value=False)
c.loc[eve, "daytype"] = "eve_of_long"
c.to_csv("data/calendar.csv", index=False, encoding="utf-8-sig")
print(c[c.date.between("2024-01-01", "2025-12-31")].daytype.value_counts().to_dict())
