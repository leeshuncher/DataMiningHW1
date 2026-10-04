# TODO

Status: the modelling table `master.parquet` is built (49,656 hourly rows, 2018–2019 and 2023–2026-08). Baselines are trained (`train_baseline.py` → `baseline_results.csv`).

## 1. Baseline models (done; refine next)
Result (MAE, test 2026; split train 2018–19+2023 / valid 2024–25 / test 2026):
all hours: lag_7d 202.8 → LGBM A 123.1 → LGBM B (+events) 111.9; event days 327→223→195;
end hour(+1): 1368→1363→1013. Event features help most at the end hour but the error there is still large
(~1000 people/hour): next try capacity/ticket counts, event-type-specific models, quantile/Poisson loss, tuning.
Notes: early stopping uses the valid set, so valid scores are slightly optimistic; test is clean.

Variants (`train_variants.py` → `variant_results.csv`, features = C, test MAE):
| subset | C L1 | C Poisson | avg(L1,Poisson) | per-type L1 | per-type Poisson |
|---|---|---|---|---|---|
| all | 111.3 | 112.1 | **109.1** | 123.7 | 124.5 |
| concert days | 208.5 | 202.0 | **200.9** | 227.2 | 221.3 |
| end hour(+1) | 982.5 | **906.9** | 930.6 | 995.5 | 914.8 |
- Per-event-type models are worse everywhere (event segments have only ~1.3–3.3k training rows); keep one global model with the type flags.
- Poisson loss cuts the end-hour error ~8% (valid 936→911, test 983→907) but is slightly worse on ordinary hours (L1 targets the median);
  averaging L1 and Poisson gives the best overall MAE. Candidate: use Poisson (or the blend) as the main model since end hours matter most.
- Next ideas: Tweedie/log-target, tune num_leaves/min_child_samples, sample weights for event hours, peak-hour specific evaluation (peak error is still ~900).
Original plan:
- Time-based split, no shuffling. Example (not decided): train 2018–2019 + 2023, validate 2024–2025, test 2026.
- Baseline A: calendar/time features + weather + lags only. Baseline B: A + event features. Compare, especially on event evenings, since that is what staff care about.
- Candidate model: LightGBM (installed in `metro-forecast`); also try a plain "same hour last week" baseline.
- Evaluate separately on event vs non-event hours and on the end hour / end hour + 1.

## 2. Make the pipeline reproducible
- Save scripts for the steps that were run inline: OD download, station filter (`data/arena/`), hourly entry aggregation (`arena_entries_hourly.csv`).
- ~~Save the 25 arena.taipei listing pages~~ done: `external/arena_pages/`, scripts now read it.
- Generate the `events/candidates_YYYY.csv` sheets from a script instead of inline code.

## 3. Event data gaps
- 15 flagged days have no match on the official schedule (e.g. 2017-08-19..30 run, 2023-10-19..23, 2018-07-10); find out what they were.
- ~70 matched event days have no estimated end time (e.g. 太陽馬戲《阿凡達前傳》 pages use a different format); improve the parser.
- 225 official event dates were not flagged by the flow detector; decide whether small/daytime events count as event features.
- Review `類型` (keyword heuristic) and fill the `確認` column in `events/`.
- Capacity / ticket counts: DONE as far as the official site allows. Pages give 票價/主辦/售票系統, **no ticket counts**;
  venue capacity (www.arena.taipei 小巨蛋簡介): sports ~15,000, concerts ~11,000–13,000 depending on stage (fixed seats 12,477). Price features
  (`price_median/min/mean`, `n_price_tiers`) added; they help only slightly (LGBM C vs B, test end-hour(+1) MAE 1012.7 → 982.5, all 111.9 → 111.3).
  Real sales counts would need the ticket platforms (KKTIX/拓元/寬宏/年代), not tried.
- ~25% of event days still have no price (awards shows, free events, competitions); tree models treat them as NaN.
- After-midnight end times do not affect the next day's early-morning hours; handle if it matters.

## 4. Weather check
- Training uses actual weather but serving will use forecasts. Use `external/weather_forecast.csv` (2022+, only 2023+ rows are in the master) to measure how much the error matters.
- Decide whether to use `precipitation_probability` (only available since 2024-04-25).

## 5. Open decisions
- Is the COVID range exactly 2020–2022? Early-2023 lags (`lag_364d` and similar) still read 2022 values; check whether the first months of 2023 behave differently.
- Scope: 台北大巨蛋 is not covered. It needs its own station mapping (nearby stations are not the 小巨蛋 station) and its own event schedule.
- Commit these docs; configure a git identity on this machine.
