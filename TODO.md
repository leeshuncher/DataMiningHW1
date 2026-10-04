# TODO

Status: the modelling table `master.parquet` is built (49,656 hourly rows, 2018–2019 and 2023–2026-08). Baselines are trained (`train_baseline.py` → `baseline_results.csv`).

## 1. Baseline models (done; refine next)
Result (MAE, test 2026; split train 2018–19+2023 / valid 2024–25 / test 2026):
all hours: lag_7d 202.8 → LGBM A 123.1 → LGBM B (+events) 111.9; event days 327→223→195;
end hour(+1): 1368→1363→1013. Event features help most at the end hour but the error there is still large
(~1000 people/hour): next try capacity/ticket counts, event-type-specific models, quantile/Poisson loss, tuning.
Notes: early stopping uses the valid set, so valid scores are slightly optimistic; test is clean.
Original plan:
- Time-based split, no shuffling. Example (not decided): train 2018–2019 + 2023, validate 2024–2025, test 2026.
- Baseline A: calendar/time features + weather + lags only. Baseline B: A + event features. Compare, especially on event evenings, since that is what staff care about.
- Candidate model: LightGBM (installed in `metro-forecast`); also try a plain "same hour last week" baseline.
- Evaluate separately on event vs non-event hours and on the end hour / end hour + 1.

## 2. Make the pipeline reproducible
- Save scripts for the steps that were run inline: OD download, station filter (`data/arena/`), hourly entry aggregation (`arena_entries_hourly.csv`).
- Save the 25 arena.taipei listing pages into the repo (e.g. `external/arena_pages/`) and replace the hard-coded scratchpad path in `match_events.py` and `fetch_details.py`.
- Generate the `events/candidates_YYYY.csv` sheets from a script instead of inline code.

## 3. Event data gaps
- 15 flagged days have no match on the official schedule (e.g. 2017-08-19..30 run, 2023-10-19..23, 2018-07-10); find out what they were.
- ~70 matched event days have no estimated end time (e.g. 太陽馬戲《阿凡達前傳》 pages use a different format); improve the parser.
- 225 official event dates were not flagged by the flow detector; decide whether small/daytime events count as event features.
- Review `類型` (keyword heuristic) and fill the `確認` column in `events/`.
- Add venue capacity / ticket counts (not collected yet).
- After-midnight end times do not affect the next day's early-morning hours; handle if it matters.

## 4. Weather check
- Training uses actual weather but serving will use forecasts. Use `external/weather_forecast.csv` (2022+, only 2023+ rows are in the master) to measure how much the error matters.
- Decide whether to use `precipitation_probability` (only available since 2024-04-25).

## 5. Open decisions
- Is the COVID range exactly 2020–2022? Early-2023 lags (`lag_364d` and similar) still read 2022 values; check whether the first months of 2023 behave differently.
- Scope: 台北大巨蛋 is not covered. It needs its own station mapping (nearby stations are not the 小巨蛋 station) and its own event schedule.
- Commit these docs; configure a git identity on this machine.
