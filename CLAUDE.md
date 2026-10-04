# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Forecast **hourly 進站 (entry) counts at 台北小巨蛋 station for the next day**, to help station staff plan staffing/ticket booths/crowd control on event nights. Inputs must be things known at prediction time: date/hour, weather, same-hour last week, and the event schedule (start/end time). Split: train 2018–2025 / test 2026-01..08 (no validation set; tree count picked on an inner 2025 hold-out). Baselines are in `train_baseline.py` (results in `baseline_results.csv`); `python` needs pandas/lightgbm/pyarrow.

## Environment and commands

- Use the conda env `metro-forecast` (pandas, scikit-learn, lightgbm, pyarrow, jupyterlab). The base Python has no pandas.
- Run scripts as `mamba run -n metro-forecast python <script>.py` from the repo root (all paths are relative to it).
- There are no tests or linters.
- Rebuild the modelling table: `python build_event_features.py` then `python build_master.py` (writes `master.parquet` and `master.csv`).

## Data pipeline (order matters)

1. **Raw OD data** → `data/od_YYYYMM.csv` (116 months, ~33 GB, git-ignored). Downloaded with an inline script from the URL list in `臺北捷運每日分時各站OD流量統計資料.csv`; no script is saved in the repo. Columns: `日期,時段,進站,出站,人次`. Never load these whole; process per file.
2. **Station filter** → `data/arena/arena_YYYYMM.csv`: rows where 進站 or 出站 is the arena station (awk). Also not saved as a script.
3. **`arena_entries_hourly.csv`** (`date,hour,entries`): sum of 人次 where 進站 is the arena station, from step 2. Hours with no row (no service) are missing here and are filled with 0 later.
4. `fetch_external.py` → `external/weather_actual.csv` (Open-Meteo ERA5), `external/weather_forecast.csv` (archived forecasts, 2022+; **not used**), `external/calendar.csv` (ruyut/TaiwanCalendar holidays and makeup workdays).
5. **Event schedule** (from the official site www.arena.taipei 「歷年節目查詢」):
   - `match_events.py`, `fetch_details.py`, `fetch_tickets.py` read the 25 saved listing pages from `external/arena_pages/{1..25}.html` (fetched from `News.aspx?PageSize=20&n=4A7BFF61FA074F0F&page=N&sms=F9A95D3F5A5C2C68`, host must be `www.`; the sandbox must allow www.arena.taipei).
   - `fetch_tickets.py` → `events_tickets.csv` (主辦單位/票價/售票系統 per event page; no ticket counts exist on the site). `build_event_features.py` turns prices into `price_median/min/mean`, `n_price_tiers` (median, not max, because VIP/package prices distort the max).
   - `match_events.py` → `events_official.csv` (one row per event date; cancelled dates dropped, postponed dates moved) and fills `活動名稱/類型/備註` in `events/candidates_YYYY.csv`.
   - `fetch_details.py` → `events_detail.csv` (the 「活動日期/時間」 field of every detail page); `fill_times.py` → `events_sessions.csv` and fills 開演時間/預估散場 in the yearly sheets.
   - `event_series.py` links titles of the same artist/series (used by `build_master.py` for `series_*` history features; they bring little gain, see TODO).
   - `build_event_features.py` → `events_daily.csv` (daily features from the official schedule).
6. `build_master.py` joins everything into the hourly table `master.parquet`.

`detect_events.py` (flow-spike detector → `events_all_days.csv`, `events_candidates.csv`) only exists to produce the review sheets in `events/` and to cross-check the official schedule. The yearly sheets were generated inline from its output, then enriched by steps 5.

## Rules that are easy to get wrong

- **Leakage**: event features must come from the official schedule (`events_daily.csv`), never from the flow-based detector or the `events/` sheets, since those are derived from the target. Likewise, only lags ≥ 7 days are allowed (`lag_7d/14d/21d/364d`, `lag_mean_4w`); no 1-day lag.
- **Weather**: only actual (`*_act`) weather is used by decision; forecast data was deliberately dropped from the master table. Note this means train/serve weather is not identical.
- **Row exclusions**: `build_master.py` drops 2017 (lags incomplete) and 2020–2022 (COVID) **after** computing lags, so 2023+ lags still read 2022 values. Do not drop years before the lag step.
- **Station name** is `台北小巨蛋` (台, not 臺) in all OD files. The pasted source list file name uses 臺.
- `entries` is 0-filled on a full hourly grid; `has_record=False` marks hours absent from the OD data (mostly 02–05h, no service).
- Missing end times are imputed as start + 3 h (`end_imputed`); events with no times have NaN relative-time features.
- `type` classification (`event_utils.kind`) is a keyword heuristic, not verified.
- Files are UTF-8 with BOM (read with `utf-8-sig`); the CSVs written by `csv.writer` use CRLF, so strip `\r` when using awk/shell.
- The estimated end time matches the entry spike well: the peak hour is the end hour or the one after in ~89% of checked days.

## Git

Single branch `master`; `data/` and `__pycache__/` are ignored. No git identity is configured on this machine, so commits need `-c user.name=... -c user.email=...` or a local `git config`.
