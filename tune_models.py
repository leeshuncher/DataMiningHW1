"""Tweedie / log1p-target variants, then a small random search on the best loss.
Selection uses the VALID set only (2024-25); test (2026) is reported once at the end."""
import warnings; warnings.filterwarnings("ignore")
import time, numpy as np, pandas as pd, lightgbm as lgb
exec(open("train_baseline.py", encoding="utf-8").read().split("def fit_predict")[0])

X = mk(BASE + EVENT + PRICE)
y = df.entries.values
tr, va = (df.split == "train").values, (df.split == "valid").values
end_mask = (df.is_end_hour | df.is_post_end_hour).values
ev_mask = df.event_today.values
BASEP = dict(learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, verbose=-1, n_estimators=4000, n_jobs=-1)

def run(obj, log=False, **kw):
    p = {**BASEP, "objective": obj, **kw}
    if obj == "regression_l1": p["metric"] = "l1"
    m = lgb.LGBMRegressor(**p)
    t = np.log1p(y) if log else y
    m.fit(X[tr], t[tr], eval_set=[(X[va], t[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    pr = m.predict(X)
    pr = np.expm1(pr) if log else pr
    return np.clip(pr, 0, None), m

def score(pr, mask):
    return np.abs(pr[mask] - y[mask]).mean()

def report(name, pr, rows):
    r = {"model": name}
    for s in ["valid", "test"]:
        sp = (df.split == s).values
        r[f"{s}_all"] = score(pr, sp); r[f"{s}_event"] = score(pr, sp & ev_mask); r[f"{s}_end"] = score(pr, sp & end_mask)
    rows.append(r); return r

rows, preds = [], {}
t0 = time.time()
for name, args in [("l1", ("regression_l1", False)), ("poisson", ("poisson", False)),
                   ("tweedie1.1", ("tweedie", False, dict(tweedie_variance_power=1.1))),
                   ("tweedie1.3", ("tweedie", False, dict(tweedie_variance_power=1.3))),
                   ("tweedie1.5", ("tweedie", False, dict(tweedie_variance_power=1.5))),
                   ("tweedie1.8", ("tweedie", False, dict(tweedie_variance_power=1.8))),
                   ("log1p_l2", ("regression", True)), ("log1p_l1", ("regression_l1", True))]:
    obj, log = args[0], args[1]; kw = args[2] if len(args) > 2 else {}
    preds[name], _ = run(obj, log, **kw); report(name, preds[name], rows)
    print(name, f"{time.time()-t0:.0f}s", flush=True)
loss = pd.DataFrame(rows).round(1); print(loss.to_string(index=False))
loss.to_csv("loss_results.csv", index=False, encoding="utf-8-sig")

# random search for the loss with best valid end-hour MAE among the Poisson-family / log choices
cand = loss.set_index("model")
best = cand.loc[["poisson", "tweedie1.1", "tweedie1.3", "tweedie1.5", "tweedie1.8"], "valid_end"].idxmin()
print("tuning loss:", best)
obj, kw0 = ("poisson", {}) if best == "poisson" else ("tweedie", dict(tweedie_variance_power=float(best[7:])))
rng = np.random.default_rng(0)
trials = []
for i in range(24):
    kw = dict(num_leaves=int(rng.choice([15, 31, 63, 127])), min_child_samples=int(rng.choice([5, 10, 20, 40, 80])),
              learning_rate=float(rng.choice([0.02, 0.03, 0.05])), colsample_bytree=float(rng.choice([0.5, 0.7, 0.9])),
              reg_lambda=float(rng.choice([0, 1, 5, 20])), max_bin=int(rng.choice([127, 255])))
    pr, m = run(obj, False, **kw0, **kw)
    sp = va
    trials.append({**kw, "iters": m.best_iteration_, "valid_all": score(pr, sp), "valid_event": score(pr, sp & ev_mask),
                   "valid_end": score(pr, sp & end_mask)})
    print(i, trials[-1], flush=True)
tr_df = pd.DataFrame(trials).round(3)
tr_df.to_csv("tuning_trials.csv", index=False)
tr_df["score"] = tr_df.valid_end / tr_df.valid_end.min() + tr_df.valid_all / tr_df.valid_all.min()
b = tr_df.sort_values("score").iloc[0]; print("best trial\n", b)
bk = dict(num_leaves=int(b.num_leaves), min_child_samples=int(b.min_child_samples), learning_rate=float(b.learning_rate),
          colsample_bytree=float(b.colsample_bytree), reg_lambda=float(b.reg_lambda), max_bin=int(b.max_bin))
final, _ = run(obj, False, **kw0, **bk)
l1_tuned, _ = run("regression_l1", False, **bk)
rows2 = []
report(f"{best}_default", preds[best], rows2); report(f"{best}_tuned", final, rows2)
report("l1_default", preds["l1"], rows2); report("l1_tuned_params", l1_tuned, rows2)
report(f"blend({best}_tuned,l1_tuned)", (final + l1_tuned) / 2, rows2)
out = pd.DataFrame(rows2).round(1); print(out.to_string(index=False))
out.to_csv("tuned_results.csv", index=False, encoding="utf-8-sig")
