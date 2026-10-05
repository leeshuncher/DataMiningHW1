"""Rolling-origin CV on the four quarters of 2024 to select the configuration. Grid (pre-specified, 48 configs):
loss {log-Ridge, L1} x eve handling {ignored, shared evening-bin effect, bin x Friday/other} x last-year feature {no, yes} x 2023 ordinary weight {1, 0.1} x
long-weekend effect {all hours, daytime only}. Score = 0.5*MAE(ordinary days) + 0.5*MAE(holiday days), pooled over the four validation quarters.
Outputs cv_grid.csv (all configs) and cv_selected.json. Usage: python src/cv_select.py"""
import itertools, json, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from protocol import *

d = data()
GRID = [dict(loss=l, eve=e, ly=ly, w=w, night=n) for l, e, ly, w, n in itertools.product(["log", "l1"], ["none", "shared", "by_dow"], [0, 1], [1.0, 0.1], [0, 1])]

def run(cfg):
    errs = {"ord": [], "hol": [], "all": []}; per_fold = []
    for s, e in QUARTERS:
        tr, va = fold_split(d, s, e); p = predict_cfg(tr, va, cfg); ae = np.abs(p - va.minutes.values)
        o, h = va.daytype.isin(ORD).values, va.daytype.isin(HOL).values
        errs["ord"] += list(ae[o]); errs["hol"] += list(ae[h]); errs["all"] += list(ae); per_fold.append(float(ae.mean()))
    r = {**cfg, "mae_ordinary": np.mean(errs["ord"]), "mae_holiday": np.mean(errs["hol"]), "n_holiday_rows": len(errs["hol"]), "mae_all": np.mean(errs["all"]),
         "fold_mae_mean": np.mean(per_fold), "fold_mae_sd": np.std(per_fold, ddof=1)}
    r["score"] = 0.5 * r["mae_ordinary"] + 0.5 * r["mae_holiday"]; return r

if __name__ == "__main__":
    with ProcessPoolExecutor(4) as ex: res = list(ex.map(run, GRID))
    g = pd.DataFrame(res).sort_values("score"); g.to_csv("results/cv_grid.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20); print(g.head(15).round(3).to_string(index=False))
    best = g.iloc[0]; sel = {k: (best[k].item() if hasattr(best[k], "item") else best[k]) for k in ["loss", "eve", "ly", "w", "night"]}
    json.dump(sel, open("results/cv_selected.json", "w")); print("selected:", sel)
    # marginal effects of each choice (mean score over the configs sharing that choice)
    for col in ["loss", "eve", "ly", "w", "night"]: print(col, g.groupby(col).score.mean().round(3).to_dict())
