# Data: Runner Heart-Rate Forecast (FitRec / Endomondo)

Source: FitRec `endomondoHR_proper.json` (filtered version, 167,373 workouts; UCSD McAuley lab).
Rebuild: download the file into `data/`, then `python prepare_data.py` and `python build_table.py`
(about 4.9 GB download, ~3 min processing, outputs are git-ignored; `sample_rows.csv` is a 3,000-row sample).

## Pipeline
1. `prepare_data.py` keeps `sport == "run"` (70,591 workouts). Per workout: drop non-increasing timestamps; drop GPS jumps (implied > 9 m/s);
   drop HR outside 40–220 and spikes (> 25 bpm from a 5-point median); skip workouts with flat HR (std < 3); derive speed from GPS distance
   (the raw `speed` field is not used); resample to a 10 s grid, grid points inside gaps > 30 s are invalid; build features from the past only
   and the target HR(t+60 s); keep every 3rd valid row (30 s). Result: 69,256 workouts, 8,035,452 rows (`prepare_stats.json` lists the dropped ones).
2. `build_table.py` keeps runners with >= 10 workouts (633 runners, 68,670 workouts, 7,973,405 rows), splits per runner by time and adds the HR-gap feature.

## Split
Per runner, the earliest 80% of workouts are `train`, the latest 20% `test` (whole workouts). train: 55,181 workouts / 6,391,356 rows;
test: 13,489 workouts / 1,582,049 rows. Use `workout_id` as the group in GroupKFold inside train; touch test once.

## Columns of `data/modeling_table.parquet`
| column | meaning |
|---|---|
| userId, workout_id, start_ts, gender | ids, workout start (unix s), gender (male / female / unknown) |
| split, workout_idx | train / test; order of the workout among the runner's workouts |
| elapsed_s | seconds since workout start |
| hr | heart rate at t (BPM) |
| hr_slope30 / hr_slope60 | HR change over the last 30 / 60 s (BPM per s) |
| speed | GPS-derived speed, mean over the last 30 s (m/s) |
| dspeed30 / dspeed60 | speed change over the last 30 / 60 s |
| speed_ema | exponential moving average of 10 s speed, half-life 30 s |
| grade | slope from smoothed altitude over the last 30 s (clipped to ±0.3) |
| dgrade30 / dgrade60 | grade change over 30 / 60 s |
| speed_x_grade, elapsed_x_speed | interaction terms |
| hr_ss_expected, hr_gap, gap_source | runner's steady-state HR at this speed (fit on that runner's train workouts), gap = expected − hr; `global` = fallback fit |
| **hr_future** | target: HR(t + 60 s) |
| **dhr** | recommended target: hr_future − hr |

No whole-workout statistics are used. Every feature uses only values at or before t.

## Caveats (see `data_report.json`)
- **Persistence is already strong:** test MAE of "HR stays the same" is 4.36 BPM, so the proposed threshold MAE <= 5 BPM is met by the trivial baseline.
  A first Ridge check on all features gets 4.32 BPM (R² 0.10 on dhr). Gains only show on slices: pace change (|dspeed60| > 0.5, 14% of rows) 7.50 → 7.25, hills (|grade| > 0.05) 5.91 → 5.81.
  Frame the claim around those slices and the decision accuracy, not overall MAE.
- Correlation of dhr with hr_gap is 0.19 and with hr_slope60 is −0.19 (mean reversion); with dspeed60 it is only 0.03 overall.
- `hr_gap` train rows are in their own steady-state fit (slightly optimistic); test rows are clean.
- 94% of rows are male runners; rows within one workout are autocorrelated, so use HAC errors / group splits.
- Rows are every 30 s, not every 10 s; set `STRIDE = 1` in `prepare_data.py` for the full grid (about 3x the rows).
