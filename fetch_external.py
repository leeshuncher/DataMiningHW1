"""Fetch hourly weather (Open-Meteo) and Taiwan holiday calendar into external/."""
import csv, json, time, urllib.request, urllib.parse

LAT, LON = 25.0516, 121.5493  # 台北小巨蛋
START, END = "2017-01-01", "2026-08-31"
HOURLY = ["temperature_2m", "apparent_temperature", "relative_humidity_2m",
          "precipitation", "weather_code", "wind_speed_10m"]

def get(url, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as e:
            if i == tries - 1: raise
            time.sleep(5 * (i + 1))

def weather(base, first, out, extra=()):
    cols = HOURLY + list(extra)
    rows = []
    for y in range(int(first[:4]), 2027):
        s, e = max(f"{y}-01-01", first), min(f"{y}-12-31", END)
        if s > e: continue
        q = urllib.parse.urlencode(dict(latitude=LAT, longitude=LON, start_date=s, end_date=e,
                                        hourly=",".join(cols), timezone="Asia/Taipei"))
        h = json.loads(get(f"{base}?{q}"))["hourly"]
        rows += [[h["time"][i]] + [h[c][i] for c in cols] for i in range(len(h["time"]))]
        print(out, y, len(rows)); time.sleep(1)
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["datetime"] + cols); w.writerows(rows)

# observation-like (ERA5 reanalysis), full history
weather("https://archive-api.open-meteo.com/v1/archive", START, "external/weather_actual.csv")
# archived forecasts (what was known beforehand), available from ~2022
weather("https://historical-forecast-api.open-meteo.com/v1/forecast", "2022-01-01",
        "external/weather_forecast.csv", extra=["precipitation_probability"])

# holiday calendar
with open("external/calendar.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["date", "weekday", "is_holiday", "description"])
    for y in range(2017, 2027):
        for d in json.loads(get(f"https://raw.githubusercontent.com/ruyut/TaiwanCalendar/master/data/{y}.json")):
            ds = d["date"]; w.writerow([f"{ds[:4]}-{ds[4:6]}-{ds[6:]}", d["week"], d["isHoliday"], d["description"]])
    print("calendar done")
