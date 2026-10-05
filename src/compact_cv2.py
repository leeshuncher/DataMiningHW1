"""Second round of the CV-based design choice, now with the decision cost included (pre-specified): among compact designs whose CV MAE score
(0.5*ordinary + 0.5*holiday) is within 5% of the best large model, choose the one with the lowest CV decision regret (mean minutes lost per 4-hour window).
Candidates: compact8, compact8h (adds hourly main effects) x loss {log, l1} x 2023 ordinary weight {1, 0.1}. Outputs compact_cv2.csv."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from protocol import data, fold_split, QUARTERS, ORD, HOL
from forecast_common import decision_regret
import compact_model as cm
d = data(); rows = []
for name in ["compact8", "compact8h"]:
    for loss in ["log", "l1"]:
        for w in [1.0, 0.1]:
            eo, eh, regs, regs_h = [], [], [], []
            for s, e in QUARTERS:
                tr, va = fold_split(d, s, e); p = cm.fit_predict(tr, va, name, loss, w)[0]; ae = np.abs(p - va.minutes.values)
                eo += list(ae[va.daytype.isin(ORD).values]); eh += list(ae[va.daytype.isin(HOL).values])
                dr = decision_regret(va, {"m": p}, 60); regs += list(dr.m); regs_h += list(dr[dr.daytype.isin(HOL)].m)
            rows.append({"model": name, "loss": loss, "w2023": w, "mae_ordinary": np.mean(eo), "mae_holiday": np.mean(eh), "score": 0.5 * np.mean(eo) + 0.5 * np.mean(eh), "regret_all": np.mean(regs), "regret_holiday": np.mean(regs_h) if regs_h else np.nan, "n_windows": len(regs)})
r = pd.DataFrame(rows).sort_values("regret_all"); r.to_csv("results/compact_cv2.csv", index=False, encoding="utf-8-sig"); pd.set_option("display.width", 200); print(r.round(3).to_string(index=False))
print("5% rule threshold on score:", round(pd.read_csv("results/cv_grid.csv").score.min() * 1.05, 3))
