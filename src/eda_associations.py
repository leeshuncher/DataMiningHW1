"""Exploratory association statistics on TRAINING YEARS ONLY (2023-2024): Pearson r, Spearman rho and mutual information between travel time and candidate features,
plus redundancy among them. Outputs eda_associations.csv, eda_redundancy.csv. Usage: python src/eda_associations.py"""
import numpy as np, pandas as pd
from sklearn.feature_selection import mutual_info_regression
from protocol import data
d = data(); d = d[d.index.year <= 2024].copy(); y = d.minutes.values
F = pd.DataFrame({"hour of day (0-23)": d.hod, "daytime 08-18h": ((d.hod >= 8) & (d.hod < 18)).astype(float), "weekday number (Mon=0)": d.dow, "Saturday": (d.daytype == "sat").astype(float),
                  "Sunday": (d.daytype == "sun").astype(float), "long-weekend day": d.daytype.str.startswith("lw_").astype(float), "long weekend, first day": (d.daytype == "lw_first").astype(float),
                  "long weekend, last day": (d.daytype == "lw_last").astype(float), "eve of long weekend": (d.daytype == "eve_of_long").astype(float), "Friday (ordinary)": ((d.daytype == "weekday") & (d.dow == 4)).astype(float),
                  "holiday length (days, 0 if none)": np.where(d.daytype.str.startswith("lw_"), d.block_len, 0), "same slot last week": d.lag7, "mean of same slot, last 4 weeks": d.lag4mean})
F["Saturday x daytime"] = F["Saturday"] * F["daytime 08-18h"]; F["long weekend x daytime"] = F["long-weekend day"] * F["daytime 08-18h"]; F["first day x afternoon 14-18h"] = F["long weekend, first day"] * ((d.hod >= 14) & (d.hod < 18)).astype(float)
rows = []
for c in F.columns:
    x = F[c].values.astype(float); ok = ~np.isnan(x)
    rows.append({"feature": c, "pearson_r": np.corrcoef(x[ok], y[ok])[0, 1], "spearman_rho": pd.Series(x[ok]).corr(pd.Series(y[ok]), method="spearman"),
                 "mutual_info": mutual_info_regression(x[ok].reshape(-1, 1), y[ok], random_state=0, discrete_features=bool(len(np.unique(x[ok])) <= 12))[0], "n": int(ok.sum())})
a = pd.DataFrame(rows).sort_values("mutual_info", ascending=False).round(3); a.to_csv("results/eda_associations.csv", index=False, encoding="utf-8-sig"); pd.set_option("display.width", 200); print(a.to_string(index=False))
R = F[["Saturday", "long-weekend day", "daytime 08-18h", "same slot last week", "mean of same slot, last 4 weeks", "Saturday x daytime", "long weekend x daytime"]].corr().round(2); R.to_csv("results/eda_redundancy.csv", encoding="utf-8-sig"); print(R.to_string())
print("rows", len(d), "days", d.index.normalize().nunique())
