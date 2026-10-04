# TODO

**Split changed (latest): train 2018–2025 (2020–22 excluded in master), test 2026-01..08, no validation set.** `train_baseline.py` and `eval_mape.py`
pick the number of trees on an inner hold-out (2025, fitted on ≤2024) and refit on all of train. Latest test MAE (all / event day / end hour(+1)):
lag_7d 202.8/327.2/1367.6, A 116.8/220.0/1390.6, B 104.5/186.2/980.0, C 102.9/182.2/938.3, D 103.6/184.6/968.9. With this split the Poisson
advantage at end hours disappeared (end-hour WAPE: L1 40.7, Poisson 41.3, blend 40.1), consistent with the earlier "differences are noise" note.
`train_variants.py`, `tune_models.py`, `diag_end.py` still use the legacy 3-way split (train 2018-19+2023 / valid 2024-25 / test 2026); the numbers below this note come from that split.

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
- Tweedie / log target / tuning (`tune_models.py` → `loss_results.csv`, `tuning_trials.csv`, `tuned_results.csv`; test MAE all / event / end hour(+1)):
  L1 111.3/193.0/982.5; Poisson 112.1/192.3/906.9; Tweedie p=1.1 114.2/196.6/909.0 (p≥1.3 gets worse, p=1.8 is bad); log1p+L2 114.3/194.7/926.2;
  log1p+L1 110.0/191.0/1011.2 (best on all hours, worst at end hour). Tweedie does not beat Poisson.
- Random search (24 trials, selected on valid) did not help: tuned Tweedie test end hour 909→936 (worse), tuned L1 983→941 (better), blend(tuned Tweedie, tuned L1) 108.9/191.0/919.6.
  Valid end-hour MAE across all 24 trials spans 900–953 with no clear pattern: with only ~360 valid / ~117 test end-hour rows, differences of a few % are noise.
  Conclusion: stay with default params; Poisson (or the Poisson+L1 blend) is the main candidate. Further loss/param tuning is not worth it.
- Next ideas: sample weights for event hours, repeated/rolling-origin evaluation for stable end-hour estimates, better event features (see section 3), then a day-ahead pipeline with forecast weather (section 4).

End-hour diagnosis (`diag_end.py` → `end_hour_diag.csv`; 59 test event days, window = end hour-1..+2, L1 model C):
- Median |error| of the window total is 27%, corr(pred, actual) only 0.49; the actual window total itself varies only ±32% (CV) around its mean,
  so the model is barely better than "predict the average event". The model cannot tell a big show from a small one.
- Concerts are under-predicted by ~26% on average (pred 4,945 vs actual 6,702); the worst days are -40..-55%.
- Predicted peak hour equals the actual one on only 54% of days (41% are off by one hour; end-time estimate error splits the spike).
- Outlier: 2026-07-11 actual 420 vs pred 5,677 (probably cancelled/postponed; not reflected in the official page).
- Tried past-event size features (`prev_event_win`, `prev3_event_win`, `prev_type_win`: end-window entries of event days >=7 days earlier, in `build_master.py`):
  no gain (model D end-hour(+1) test MAE 994.8 vs 982.5 for C); their correlation with the target event's size is only 0.05–0.2. Kept in master but unused by the best model.
- What would actually help: information about THIS event's size/artist — same-artist/series history (needs artist name matching), ticket sales/sold-out status from
  platforms, or announced session count/seat map. Also a better end-time estimate.
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
