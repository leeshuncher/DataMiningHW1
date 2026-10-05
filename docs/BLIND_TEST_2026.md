# Blind test on 2026 (pre-registered)

Written and committed **before** any 2026 prediction was computed or any 2026 error was looked at.
The 2026 rows (2026-01-01 to 2026-08-31, M04A) were added to the repo in commit ff89c0c; only the
travel-time tables were rebuilt, no model was evaluated on them.

## Why
2025 was looked at several times while the design was being developed (go/no-go, eve variants,
30- vs 60-min slots, the first compact design). 2025 is therefore a *development test*. 2026 is the
first data that played no role in any decision.

## Frozen design (no changes allowed after this commit)
- Model: compact median (L1) linear regression, design `compact8h` in `src/compact_model.py`
  (23 hourly effects + 9 day types x (main + 8 two-hour blocks 06-22h) + 4 Friday terms + 1 >=4-day term).
- Loss L1 by 8-step IRLS ridge; alpha chosen from {0.01, 0.1, 1, 3, 10} by 4-fold TimeSeriesSplit inside the training data.
- Weights: 2023 ordinary days (weekday/Sat/Sun) 0.1, all other rows 1 (`drift_weights(train, 0.1)`).
- Baselines exactly as in `src/final_compact.py`: training mean, last week same hour, same holiday last time
  (else last week), seasonal mean hour x weekday, OLS on last week, GBM (L1), GBM (log), large one-hot ridge
  with the CV-selected configuration `results/cv_selected.json`.
- Rows: identical for every method (target, last week and 4-week mean present).

## Evaluations (all reported, whatever the outcome)
- **Primary (P):** refit every method on 2023-01-01 .. 2025-12-31 with the rules above, predict 2026-01-01 .. 2026-08-31.
- **Secondary (S):** the models fitted on 2023-2024 exactly as in the paper, applied to 2026 without refitting.
- Metrics: MAE, RMSE, R^2; MAE on ordinary days, long-weekend days, holiday days (long weekend + eve),
  first day 10-18h, eve 16-23h, congested hours (actual > 40 min).
- Decision regret per 4-hour window (06-10, 10-14, 14-18, 18-22h); threshold 1.0 min on long-weekend days.
- Day-block bootstrap (2,000 resamples, seed 0): improvement of ours over each baseline, and ours - GBM (L1) in minutes.

## Known limits, stated in advance
2026-01..08 holds 5 long weekends (22 days: 5 first, 12 middle, 5 last) and 5 eves, so holiday intervals will be wide.
Script: `src/blind_2026.py`; outputs `results/blind2026_*.csv`.
