# Day-ahead travel time forecasting on Freeway 5 southbound

NYCU Data Mining HW1. We forecast the hourly small-car travel time (Nangang System 05F0000S to 05F0439S, southbound, through the Hsuehshan Tunnel) one day ahead, using calendar and holiday-structure features only, with a 109-term median linear regression. Train 2023-2024, test 2025.

Paper: [`paper/paper.pdf`](paper/paper.pdf) (ACL format, LaTeX source in `paper/`). Assignment slides: [`docs/2026_NYCU_Data_Mining_HW1_slides.pdf`](docs/2026_NYCU_Data_Mining_HW1_slides.pdf).

## Headline results (2025 test year)
- MAE 1.42 min; 25% below last week's same hour (95% CI 16-34%), 50% below it on long-weekend days (3.40 to 1.69 min).
- Departure-hour regret on long weekends 0.82 min per 4-hour window (threshold 1.0).
- Statistically tied with gradient boosting in cross-validation (p = 0.72) and with a 1,000+ term one-hot ridge, while every coefficient reads as minutes of delay.
- The 2025 test set was not fully blind; see "Protocol note" in the paper.

## Layout
| Path | Content |
|---|---|
| `src/` | Final pipeline (run from the repo root, see below) |
| `results/` | CSV/JSON outputs of the final pipeline (`final2_*` = final model tables) |
| `data/` | Raw M04A archives, hourly travel-time tables, holiday calendar |
| `paper/` | LaTeX paper, ACL style files, compiled PDF |
| `docs/` | Data notes (`DATA.md`), original proposal, assignment slides |
| `archive/` | Earlier exploratory scripts, results and notes, superseded by `src/` (run with `PYTHONPATH=src`) |

## Reproduce
Run everything from the repository root; scripts read `data/` and write to `results/`.
```
python src/download_m04a.py          # rate-limited download of M04A (needs access to tisvcloud.freeway.gov.tw)
python src/build_travel_time.py      # hourly travel-time table
python src/fetch_calendar.py         # holidays and make-up workdays (ruyut/TaiwanCalendar)
python src/explore_train.py          # training-years-only exploration
python src/eda_associations.py       # Pearson / Spearman / MI, redundancy
python src/cv_select.py              # rolling-origin CV selection for the large one-hot model
python src/compact_model.py; python src/compact_cv2.py   # compact design selection
python src/final_compact.py          # final model, baselines, bootstrap, decision metric
python src/final_ablation_coef.py    # ablation Sets A/B/C, coefficient table
python src/alpha_curve.py            # ridge-penalty tuning curve of the final model
python src/interpretable_ols.py      # H1-H4 tests on log minutes
python src/error_cases.py            # largest errors
cd paper && latexmk -pdf paper.tex
```
Module guide: `forecast_common.py` (data loading, ridge/GBM fits, decision metric), `protocol.py` (folds, weights), `compact_model.py` (109-term design). Data details and limits are in `docs/DATA.md`.
