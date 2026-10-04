"""Step 2: data/rows/*.parquet -> data/modeling_table.parquet (+ sample_rows.csv, data_report.md).

- Keep runners with >= MIN_WORKOUTS cleaned running workouts (needed for a per-runner time split).
- Split per runner BY TIME: the earliest 80% of a runner's workouts are `train`, the latest 20% are `test`
  (whole workouts, never rows of one workout on both sides). `workout_id` is the group key for GroupKFold inside train.
- HR-gap feature: per runner, fit a steady-state line HR ~ speed on that runner's TRAIN workouts only (steady rows:
  small speed/HR change, flat ground, running pace); hr_gap = expected steady-state HR(speed) - HR(t).
  Runners with too few steady rows fall back to the global train fit (gap_source = "global").
  Note: train rows are in their own fit, so their gap is slightly optimistic; test rows are clean.
Run `python build_table.py` after prepare_data.py."""
import glob, json
import numpy as np, pandas as pd

MIN_WORKOUTS = 10
TRAIN_FRAC = 0.8
MIN_STEADY_ROWS = 100

df = pd.concat((pd.read_parquet(f) for f in sorted(glob.glob("data/rows/shard_*.parquet"))), ignore_index=True)
n_before = (df.userId.nunique(), df.workout_id.nunique(), len(df))
wk = df.groupby(["userId", "workout_id"], as_index=False).start_ts.first()
cnt = wk.groupby("userId").size()
keep_users = cnt[cnt >= MIN_WORKOUTS].index
df = df[df.userId.isin(keep_users)].copy()
wk = wk[wk.userId.isin(keep_users)].sort_values(["userId", "start_ts", "workout_id"])
wk["rank"] = wk.groupby("userId").cumcount(); wk["n"] = wk.userId.map(cnt)
wk["split"] = np.where(wk["rank"] < np.ceil(wk["n"] * TRAIN_FRAC), "train", "test")
wk["workout_idx"] = wk["rank"]
df = df.merge(wk[["workout_id", "split", "workout_idx"]], on="workout_id", how="left")

# ---- HR gap ----
steady = (df.dspeed60.abs() < 0.3) & (df.hr_slope60.abs() < 0.05) & (df.speed > 1.5) & (df.grade.abs() < 0.03)
tr = df[steady & (df.split == "train")]
def ols(g):
    n = len(g); sx, sy = g.speed.sum(), g.hr.sum(); sxx, sxy = (g.speed ** 2).sum(), (g.speed * g.hr).sum()
    den = n * sxx - sx ** 2
    b = (n * sxy - sx * sy) / den if den > 0 else np.nan
    return pd.Series({"n": n, "b": b, "a": (sy - b * sx) / n})
gl = ols(tr)
per = tr.groupby("userId")[["speed", "hr"]].apply(ols)
per["ok"] = (per.n >= MIN_STEADY_ROWS) & (per.b > 0) & (per.b < 40)
coef = df[["userId"]].join(per[["a", "b", "ok"]], on="userId")
use_user = coef.ok.fillna(False).astype(bool)
a = np.where(use_user, coef.a, gl.a); b = np.where(use_user, coef.b, gl.b)
df["hr_ss_expected"] = (a + b * df.speed).astype("float32")
df["hr_gap"] = (df.hr_ss_expected - df.hr).astype("float32")
df["gap_source"] = np.where(use_user, "user", "global")
df["gender"] = df.gender.astype("category")
df = df.sort_values(["userId", "start_ts", "elapsed_s"]).reset_index(drop=True)
df.to_parquet("data/modeling_table.parquet")
df.sample(3000, random_state=0).sort_values(["userId", "start_ts", "elapsed_s"]).to_csv("sample_rows.csv", index=False)

# ---- report ----
te = df[df.split == "test"]; trn = df[df.split == "train"]
mae = lambda e: float(np.abs(e).mean())
rep = {
    "users_before/after": [n_before[0], int(df.userId.nunique())],
    "workouts_before/after": [n_before[1], int(df.workout_id.nunique())],
    "rows_before/after": [n_before[2], len(df)],
    "rows_train/test": [len(trn), len(te)],
    "workouts_train/test": [int(trn.workout_id.nunique()), int(te.workout_id.nunique())],
    "users_with_own_gap_fit": int((per.ok & per.index.isin(keep_users)).sum()),
    "global_fit(a,b)": [float(gl.a), float(gl.b)],
    "target_dhr_std_train/test": [float(trn.dhr.std()), float(te.dhr.std())],
    "test_MAE_persistence(dhr=0)": mae(te.dhr),
    "test_MAE_train_mean_dhr": mae(te.dhr - trn.dhr.mean()),
    "test_MAE_persistence_on_|dspeed60|>0.5": mae(te.dhr[te.dspeed60.abs() > 0.5]),
    "test_MAE_persistence_on_|grade|>0.05": mae(te.dhr[te.grade.abs() > 0.05]),
    "share_test_rows_|dspeed60|>0.5": float((te.dspeed60.abs() > 0.5).mean()),
    "corr_dhr_hr_gap_test": float(te.dhr.corr(te.hr_gap)),
    "corr_dhr_dspeed60_test": float(te.dhr.corr(te.dspeed60)),
    "corr_dhr_hr_slope60_test": float(te.dhr.corr(te.hr_slope60)),
}
json.dump(rep, open("data/table_report.json", "w"), indent=1)
for k, v in rep.items(): print(k, v)
