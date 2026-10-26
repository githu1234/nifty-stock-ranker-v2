"""
selftest.py - proves the pipeline is leak-free, handles outliers, and can
actually detect signal.

  A. unit checks: bad-tick cleaning, winsorizer (train-only limits), lag
     alignment for global / FII-DII features, earnings-date features
  B. pure random prices + random sector/global/FII/earnings data, ALL feature
     groups on: accuracy / AUC must stay ~50% and IC ~0. If not, there is leakage.
  C. prices with a small planted signal: AUC must rise, and the importance
     table must credit the Technical group.
"""
import sys

import numpy as np
import pandas as pd

import engine as E

ok = True


def check(cond, msg):
    global ok
    print(("PASS  " if cond else "FAIL  ") + msg)
    ok = ok and bool(cond)


# ---------------------------------------------------------------- A. units
prices, market = E.make_synthetic(n_stocks=12, n_days=900, phi=0.0, seed=3)
df = next(iter(prices.values())).copy()

bad = df.copy()
i = 400
for col in ("Open", "High", "Low", "Close"):
    bad.iloc[i, bad.columns.get_loc(col)] *= 1.6          # one-day glitch spike
cleaned, n = E.clean_prices(bad)
check(n == 1 and bad.index[i] not in cleaned.index, "bad one-day spike that reverses is removed")

jump = df.copy()
for col in ("Open", "High", "Low", "Close"):
    jump.iloc[600:, jump.columns.get_loc(col)] *= 1.5     # genuine, persistent +50% jump
cleaned, n = E.clean_prices(jump)
check(n == 0, "genuine persistent jump is kept")

rng = np.random.default_rng(0)
train = rng.normal(size=(1000, 3))
w = E.Winsorizer().fit(train)
test = np.array([[50.0, -50.0, 0.0]])
out = w.transform(test)
check(out[0, 0] <= np.quantile(train[:, 0], 0.99) + 1e-9 and out[0, 1] >= np.quantile(train[:, 1], 0.01) - 1e-9,
      "winsorizer clips test outliers to TRAIN-set limits")

india = pd.bdate_range("2024-01-01", periods=30)
us = pd.DataFrame({"x": np.arange(len(india), dtype=float)}, index=india + pd.Timedelta(hours=0))
al = E._align(us, india, 1)
check(al["x"].iloc[5] == us["x"].iloc[4], "global cue at day t equals last value from day t-1 (lag 1)")

fii = pd.DataFrame({"fii_net": 0.0, "dii_net": 0.0}, index=india)
fii.iloc[10, 0] = 1000.0
ff = E._fii_features(fii, india)
check(ff["fii_net"].iloc[10] == 0.0 and ff["fii_net"].iloc[11] == 1000.0,
      "FII flow of day t only becomes visible on day t+1 (no look-ahead)")
check(ff["fii_5d"].iloc[10] == 0.0 and ff["fii_5d"].iloc[11] == 1000.0,
      "FII rolling sums are lagged too")

ed = pd.DatetimeIndex(["2024-01-15", "2024-04-15"])
ef = E._earn_feats(ed, pd.DatetimeIndex(["2024-01-10", "2024-01-20"]), 5)
check(ef["days_to_earnings"].iloc[0] == 5 and ef["days_since_earnings"].iloc[1] == 5,
      "earnings proximity features computed correctly")

# -------------------------------------------------- B. noise, all groups on
def pipeline(phi, horizon, model, label, retrain_every=126):
    px, mkt = E.make_synthetic(n_stocks=30, n_days=1800, phi=phi, seed=1)
    extras, smap = E.make_synthetic_extras(px, seed=5)
    res = E.run_pipeline(px, market=mkt, horizon=horizon, top_k=5, cost_pct=0.0,
                         model_name=model, min_train_days=750, retrain_every=retrain_every,
                         extras=extras, sector_map=smap)
    m = res["metrics"]
    print(f"      {label:<34} acc={m['accuracy']:.3f} AUC={m['auc']:.3f} "
          f"IC={m['ic_mean']:+.4f} (t={m['ic_tstat']:+.1f}) groups={len(res['feature_groups'])} "
          f"features={res['n_features']}")
    return res


for horizon, model, rt in ((1, "Gradient boosting", 126), (5, "Logistic regression", 126),
                           (1, "Random forest", 252)):
    r = pipeline(0.0, horizon, model, f"noise | {model} | {horizon}d", rt)
    m = r["metrics"]
    check(len(r["feature_groups"]) == 6, f"all 6 feature groups active ({model}, {horizon}d)")
    check(abs(m["auc"] - 0.5) < 0.015 and abs(m["ic_mean"]) < 0.02,
          f"noise stays ~50% with every group on ({model}, {horizon}d)")

# ---------------------------------------------------------- C. planted signal
r = pipeline(0.15, 1, "Gradient boosting", "planted signal | boosting | 1d")
check(r["metrics"]["auc"] > 0.515, "planted signal is detected with all groups on")
check(r["importance"].index[0] == "Technical", "importance table credits the Technical group")
print("\n      importance (AUC drop):", {k: round(v, 4) for k, v in r["importance"].items()})

print("\nSELFTEST", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
