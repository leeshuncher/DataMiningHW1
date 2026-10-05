"""Build the Freeway 5 southbound travel-time table from the filtered M04A files.

Target  : passenger car (vehicle type 31), Nangang System (05F0000S, km 0.0) ->
          05F0439S (km 43.9), the nearest mainline gantry past the Toucheng exit
          (ramp gantry 05FR113S at km 41.3). There is no mainline gantry at Toucheng
          itself, so the target overshoots by ~2.6 km. Segments summed:
          0000S->0055S, 0055S->0287S, 0287S->0309S, 0309S->0439S.
Method  : for each 5-minute timestamp, sum the four segment medians reported for that
          same timestamp. This is a simplification of the true trip time (the trip
          takes ~30 min, the segments are observed at the same instant).
Validity: a segment row counts only if travel_time_sec > 0 and sample_count >= 1
          (0 means "no vehicle observed"). A 5-min sum needs all four segments valid.
Bins    : 30 and 60 min, labelled by the START of the window (departure time).
          Mean of the valid 5-min sums; a window needs >= 50% of its 5-min slots
          valid, otherwise it is NaN (dropped, not imputed).
Output  : data/travel_5min.csv, data/travel_30min.csv, data/travel_60min.csv
          (columns: depart, minutes, n_valid). No weather, flow or other
          target-day information is used here.
"""
import glob
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
CHAIN = ["05F0000S", "05F0055S", "05F0287S", "05F0309S"]  # from-gantries of the 4 segments


def load():
    frames = [pd.read_csv(f, header=None, names=["t", "a", "b", "veh", "sec", "n"])
              for f in sorted(glob.glob(str(DATA / "m04a" / "*.csv")))]
    df = pd.concat(frames, ignore_index=True)
    df = df[(df.veh == 31) & df.a.isin(CHAIN) & (df.sec > 0) & (df.n >= 1)].copy()
    df["t"] = pd.to_datetime(df.t, format="%Y/%m/%d %H:%M")
    return df


def five_min(df):
    wide = df.pivot_table(index="t", columns="a", values="sec")[CHAIN]
    full = wide.index.min().normalize(), wide.index.max().normalize() + pd.Timedelta("23h55min")
    wide = wide.reindex(pd.date_range(*full, freq="5min"))
    total = wide.sum(axis=1, min_count=len(CHAIN)) / 60.0  # NaN unless all 4 segments present
    return total.rename("minutes").rename_axis("depart")


def binned(s, minutes):
    r = s.resample(f"{minutes}min")
    out = pd.DataFrame({"minutes": r.mean(), "n_valid": r.count()})
    need = minutes // 5 / 2
    out.loc[out.n_valid < need, "minutes"] = float("nan")
    return out


if __name__ == "__main__":
    s = five_min(load())
    s.to_frame().assign(n_valid=s.notna().astype(int)).to_csv(DATA / "travel_5min.csv")
    for m in (30, 60):
        binned(s, m).to_csv(DATA / f"travel_{m}min.csv")
    for m in (30, 60):
        b = binned(s, m)
        print(f"{m}min: {len(b)} windows, {b.minutes.isna().sum()} NaN, "
              f"median {b.minutes.median():.1f} min, p5 {b.minutes.quantile(.05):.1f}, "
              f"p95 {b.minutes.quantile(.95):.1f}")
