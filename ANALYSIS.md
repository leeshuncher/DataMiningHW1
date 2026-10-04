# Checks on the Runner Heart-Rate Forecast data (test = latest 20% of each runner's workouts, 1,569,940 rows)

Scripts: `analyze.py` (slices + decisions), `check_alignment.py` (feature audit). Results: `slice_results.csv`, `decision_results.csv`,
`decision_3class.csv`, `alignment_check.json`. `_med` files = same with a rolling-MEDIAN speed instead of the 30 s mean.
Target dHR = HR(t+60) − HR(t). Models: persistence (dHR = 0), linear extrapolation (c · hr_slope60 · 60), Ridge on raw columns
(hr, speed, grade, elapsed_s), Ridge on all 14 features, LightGBM on all (reference ceiling only, not the final model).

## 1. Slices (MAE in BPM)
| slice | share | persistence | Ridge all | LightGBM | Ridge vs persistence |
|---|---|---|---|---|---|
| all | 100% | 4.33 | 4.29 | 4.04 | −0.9% |
| steady speed and grade | 25% | 3.26 | 3.30 | 3.08 | +1.1% (worse) |
| \|Δspeed60\| ≥ 1.0 m/s | 4.8% | 10.02 | 9.59 | 8.72 | −4.3% |
| decelerating (Δspeed60 < −0.5) | 6.8% | 8.39 | 8.04 | 7.58 | −4.2% |
| accelerating (Δspeed60 > +0.5) | 7.0% | 6.57 | 6.42 | 5.85 | −2.4% |
| transition (speed or grade change) | 25% | 6.23 | 6.08 | 5.69 | −2.5% |
| downhill (grade < −0.05) | 5.4% | 6.10 | 5.90 | 5.61 | −3.3% |
| uphill (grade > 0.05) | 6.3% | 5.67 | 5.67 | 5.29 | 0.1% |

The larger the speed change, the larger the persistence error (3.5 → 10.0 BPM), but the linear model recovers only 2–4% of it.
The hoped-for "10 → 6 BPM" does not appear: the best slice goes 10.02 → 9.59 (Ridge) or 8.72 (LightGBM, −13%).
Ridge is slightly worse than persistence on steady rows. Extrapolation alone matches Ridge on most slices.

## 2. Feature audit
- "All features" = 4 raw + 10 engineered (slopes, speed/grade changes, EMA, interactions, HR gap). Ridge on raw only: 4.39 (worse than persistence);
  engineered: 4.29. The engineered features carry the (small) gain.
- Rolling median instead of mean for speed: same result (Ridge 4.29 both; slices differ by ≤ 0.25 BPM, a bit better on large speed changes).
- All rolling/shift operations are inside one workout (arrays are per workout). `check_alignment.py` recomputes HR(t), HR(t+60), HR slope and
  speed from the raw json for 400 workouts (47,699 rows): 99.97% agree within tolerance, median difference ~1e-6; minimum `elapsed_s` is 120 s.
- Found and fixed: a row used to need only t and t+60 s to be valid; now the whole [t−60 s, t+60 s] window must be gap-free.

## 3. Decision level (per-runner zone from the runner's TRAIN heart rate; zone q25–q75, median width 15.6 BPM)
Events among rows that start inside the zone: exits above (9.1% of rows) and below (11.2%) within 60 s.
| model | up-exit F1 | up AUC | down-exit F1 | down AUC |
|---|---|---|---|---|
| persistence, strict (margin 0) | 0.000 (recall 0) | 0.786 | 0.000 | 0.753 |
| persistence + margin (3 BPM) | 0.376 | 0.786 | 0.369 | 0.753 |
| extrapolation + margin | 0.375 | 0.784 | 0.371 | 0.752 |
| Ridge all + margin | 0.372 | 0.787 | 0.372 | 0.760 |
| LightGBM + margin | 0.400 | 0.802 | 0.392 | 0.776 |

Strict persistence has recall 0 by construction, but that is a strawman: a persistence rule with a safety margin ("warn when HR is within 3 BPM
of the bound") reaches F1 0.37 and AUC 0.79, and Ridge does not beat it. 3-class accuracy (reduce / hold / increase): persistence 0.798, Ridge 0.796, LightGBM 0.807;
on rows whose zone state actually changes, Ridge is right 17.6% of the time, LightGBM 18.0%, persistence 0% (zones q35–q65 and q15–q85 behave the same).

## Reading
- Heart rate is mostly determined by its current value and distance to the bound. The linear model adds little beyond a margin rule.
- A non-linear model gains ~5–13% on the hard slices, so there is some non-linear signal. The proposal needs a linear final model, so claims must be modest.
- Honest framing options: (a) report the small but consistent gain on transitions with a margin-rule baseline, (b) add interaction/threshold features
  (e.g. hr_gap × dspeed, piecewise terms) to the linear model and see whether they close the gap to LightGBM, (c) change the target to something persistence cannot predict.
