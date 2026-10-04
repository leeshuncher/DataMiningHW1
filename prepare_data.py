"""Step 1: FitRec (Endomondo) `endomondoHR_proper.json` -> row-level shards of 10-second time steps from RUNNING workouts.

Per workout: drop non-monotone timestamps, GPS jumps (implied > 9 m/s) and impossible/spiky heart rate, derive speed from
GPS distance (the raw `speed` field is not used), resample to a fixed 10 s grid (grid points inside gaps > 30 s are invalid),
then build features known at time t (only past values) and the target HR(t+60 s).
Output: data/rows/shard_XXXX.parquet (+ data/prepare_stats.json). Run `python prepare_data.py`.
The HR-gap feature and the train/test split need all workouts of a runner, so they are added in build_table.py."""
import json, ast, os, sys, glob, collections
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.signal import medfilt

SRC = "data/endomondoHR_proper.json"
OUT = "data/rows"
STEP = 10            # seconds per grid step
HORIZON = 6          # steps ahead -> 60 s
MAX_GAP = 30         # seconds; bracketing raw points further apart than this make the grid point invalid
STRIDE = 3           # keep every 3rd valid row (30 s) to limit autocorrelated rows
V_MAX = 9.0          # m/s, faster than any recreational runner -> GPS jump
HR_MIN, HR_MAX = 40, 220
SPIKE = 25           # bpm away from a 5-point median -> sensor spike

def haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742000 * np.arcsin(np.sqrt(a))

def parse(line):
    try:
        return json.loads(line.replace("'", '"'))
    except Exception:
        return ast.literal_eval(line)

def lag(a, k):
    out = np.full_like(a, np.nan); out[k:] = a[:-k] if k else a; return out

def process(line):
    """Returns (DataFrame | None, reason)."""
    d = parse(line)
    need = ("timestamp", "heart_rate", "altitude", "latitude", "longitude")
    if any(k not in d for k in need): return None, "missing_field"
    t = np.asarray(d["timestamp"], float); hr = np.asarray(d["heart_rate"], float)
    alt = np.asarray(d["altitude"], float); lat = np.asarray(d["latitude"], float); lon = np.asarray(d["longitude"], float)
    n = len(t)
    if not (len(hr) == len(alt) == len(lat) == len(lon) == n) or n < 30: return None, "short_or_mismatched"
    keep = np.r_[True, np.diff(t) > 0]                                     # strictly increasing time
    t, hr, alt, lat, lon = t[keep], hr[keep], alt[keep], lat[keep], lon[keep]
    for _ in range(5):                                                      # remove GPS jumps (drop the later point)
        if len(t) < 30: return None, "short_or_mismatched"
        v = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:]) / np.diff(t)
        bad = np.r_[False, v > V_MAX]
        if not bad.any(): break
        t, hr, alt, lat, lon = t[~bad], hr[~bad], alt[~bad], lat[~bad], lon[~bad]
    else:
        return None, "gps_jumps"
    ok = (hr >= HR_MIN) & (hr <= HR_MAX) & np.isfinite(hr) & np.isfinite(alt)   # impossible HR (0, >220)
    if ok.sum() < 30: return None, "bad_hr"
    t, hr, alt, lat, lon = t[ok], hr[ok], alt[ok], lat[ok], lon[ok]
    sp_ok = np.abs(hr - medfilt(hr, 5)) <= SPIKE                            # sensor spikes
    t, hr, alt, lat, lon = t[sp_ok], hr[sp_ok], alt[sp_ok], lat[sp_ok], lon[sp_ok]
    if len(t) < 30: return None, "bad_hr"
    if hr.std() < 3: return None, "flat_hr"
    dist = np.r_[0, np.cumsum(haversine(lat[:-1], lon[:-1], lat[1:], lon[1:]))]
    # resample to a fixed grid
    tg = np.arange(t[0], t[-1], STEP, dtype=float)
    idx = np.searchsorted(t, tg)
    valid = (idx > 0) & (idx < len(t))
    valid[valid] = (t[idx[valid]] - t[idx[valid] - 1]) <= MAX_GAP
    if valid.sum() < 3 * HORIZON: return None, "too_few_grid_points"
    f = lambda a: np.where(valid, np.interp(tg, t, a), np.nan)
    hr_g, alt_g, dist_g = f(hr), f(alt), f(dist)
    sp = np.full(len(tg), np.nan); sp[1:] = np.diff(dist_g) / STEP          # m/s, NaN if either end is invalid
    s = pd.Series(sp)
    speed = s.rolling(3, min_periods=3).mean().values                       # backward 30 s mean
    ema = s.ewm(halflife=3, ignore_na=False).mean().values                  # halflife 30 s
    alt_s = pd.Series(alt_g).rolling(3, min_periods=3).mean().values
    dd = dist_g - lag(dist_g, 3)
    grade = np.where(dd >= 20, (alt_s - lag(alt_s, 3)) / np.where(dd >= 20, dd, 1), 0.0)
    grade = np.where(np.isnan(alt_s) | np.isnan(lag(alt_s, 3)) | np.isnan(dd), np.nan, np.clip(grade, -0.3, 0.3))
    elapsed = tg - t[0]
    X = pd.DataFrame({
        "elapsed_s": elapsed,
        "hr": hr_g,
        "hr_slope30": (hr_g - lag(hr_g, 3)) / 30, "hr_slope60": (hr_g - lag(hr_g, 6)) / 60,
        "speed": speed, "dspeed30": speed - lag(speed, 3), "dspeed60": speed - lag(speed, 6), "speed_ema": ema,
        "grade": grade, "dgrade30": grade - lag(grade, 3), "dgrade60": grade - lag(grade, 6),
        "hr_future": np.r_[hr_g[HORIZON:], np.full(HORIZON, np.nan)],
    })
    X["speed_x_grade"] = X.speed * X.grade
    X["elapsed_x_speed"] = X.elapsed_s * X.speed
    X["dhr"] = X.hr_future - X.hr
    X = X.iloc[::STRIDE].dropna()
    if X.empty: return None, "no_valid_rows"
    X.insert(0, "start_ts", int(t[0])); X.insert(0, "workout_id", int(d.get("id", -1)))
    X.insert(0, "gender", d.get("gender", "unknown")); X.insert(0, "userId", int(d["userId"]))
    return X, "ok"

def work(lines):
    out, stats = [], collections.Counter()
    for ln in lines:
        try:
            x, why = process(ln)
        except Exception as e:                                              # never lose a batch to one bad record
            x, why = None, "error:" + type(e).__name__
        stats[why] += 1
        if x is not None: out.append(x)
    return (pd.concat(out) if out else None), stats

def batches(path, size=200):
    buf, seen = [], collections.Counter()
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            sport = ln[ln.find("'sport': '") + 10: ln.find("'", ln.find("'sport': '") + 10)] if "'sport': '" in ln else "?"
            seen[sport] += 1
            if sport == "run": buf.append(ln)
            if len(buf) >= size: yield buf; buf = []
    if buf: yield buf
    json.dump(seen, open("data/sport_counts.json", "w"), ensure_ascii=False, indent=1)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for old in glob.glob(f"{OUT}/shard_*.parquet"): os.remove(old)
    total, rows, k = collections.Counter(), 0, 0
    with Pool(max(1, (os.cpu_count() or 2) - 0)) as pool:
        for df, st in pool.imap(work, batches(SRC)):
            total.update(st)
            if df is not None:
                df = df.astype({c: "float32" for c in df.columns if df[c].dtype == "float64"})
                df.to_parquet(f"{OUT}/shard_{k:04d}.parquet"); rows += len(df); k += 1
            if k % 20 == 0: print(f"shards {k} rows {rows:,} {dict(total)}", flush=True)
    json.dump({"rows": rows, "workouts": dict(total)}, open("data/prepare_stats.json", "w"), indent=1)
    print("done", rows, dict(total))
