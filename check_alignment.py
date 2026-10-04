"""Independent audit of the modelling table against the RAW json (does not reuse prepare_data.process):
 - hr(t), hr_future = HR(t+60 s) and hr_slope60 recomputed by interpolating raw HR at absolute times start_ts + elapsed_s,
 - speed (30 s mean of GPS speed) recomputed from raw lat/lon,
 - no row looks across workouts (elapsed_s >= 60 s for every row).
Differences are expected only where cleaning removed raw points (spikes / GPS jumps)."""
import os, sys, random, json
import numpy as np, pandas as pd
import prepare_data as P

TAG = os.environ.get("TABLE_TAG", "")
d = pd.read_parquet(f"data/modeling_table{TAG}.parquet")
print("min elapsed_s over all rows:", float(d.elapsed_s.min()), "(must be >= 60)")
ids = set(d.workout_id.unique()); random.seed(1)
want = set(random.sample(sorted(ids), 400)); recs = {}
with open(P.SRC, encoding="utf-8") as fh:
    for ln in fh:
        if "'sport': 'run'" not in ln: continue
        i = ln.find("'id': "); wid = int(ln[i + 6: ln.find(",", i)])
        if wid in want: recs[wid] = P.parse(ln)
        if len(recs) == len(want): break
res = []
for wid, r in recs.items():
    g = d[d.workout_id == wid]
    t = np.asarray(r["timestamp"], float); hr = np.asarray(r["heart_rate"], float)
    lat = np.asarray(r["latitude"], float); lon = np.asarray(r["longitude"], float)
    ok = np.isfinite(hr) & (hr >= 40) & (hr <= 220); order = np.argsort(t)
    tt = t[ok]; hh = hr[ok]
    at = g.start_ts.values + g.elapsed_s.values.astype(float)
    dist = np.r_[0, np.cumsum(P.haversine(lat[:-1], lon[:-1], lat[1:], lon[1:]) * (np.diff(t) > 0))]
    sp30 = (np.interp(at, t, dist) - np.interp(at - 30, t, dist)) / 30
    res.append(pd.DataFrame({"wid": wid,
        "d_hr": np.interp(at, tt, hh) - g.hr.values,
        "d_future": np.interp(at + 60, tt, hh) - g.hr_future.values,
        "d_slope60": (np.interp(at, tt, hh) - np.interp(at - 60, tt, hh)) / 60 - g.hr_slope60.values,
        "d_speed": sp30 - g.speed.values}))
r = pd.concat(res); n_w = r.wid.nunique()
out = {"workouts_checked": int(n_w), "rows_checked": len(r)}
for c, tol in [("d_hr", 1), ("d_future", 1), ("d_slope60", 0.02), ("d_speed", 0.3)]:
    out[c] = {"share_within_tol": float((r[c].abs() <= tol).mean()), "tol": tol,
              "median_abs": float(r[c].abs().median()), "p99_abs": float(r[c].abs().quantile(.99))}
json.dump(out, open(f"alignment_check{TAG}.json", "w"), indent=1); print(json.dumps(out, indent=1))
