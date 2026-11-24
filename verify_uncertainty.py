#!/usr/bin/env python3
"""
Recomputes every uncertainty / shift / error-analysis number of Report 3 from saved test predictions,
so the report can be corrected from ONE source of truth.

1) Export the test predictions of your champion LSTM (one row per test engine) from your own code:
       import pandas as pd
       pd.DataFrame({"true_rul": y_true, "mean": mc_mean, "sigma": mc_std}).to_csv("lstm_fd001_mc.csv", index=False)
   (use the Monte-Carlo-dropout mean/std; export a second file with the ensemble mean/std if you want that row too)
2) Run:   python3 verify_uncertainty.py --csv lstm_fd001_mc.csv
Needs: pip3 install numpy pandas   (scipy optional, for p-values)
"""
import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True); ap.add_argument("--cap", type=float, default=125)
ap.add_argument("--z", type=float, default=1.645)
a = ap.parse_args()
df = pd.read_csv(a.csv); df.columns = [c.lower() for c in df.columns]
for c in ("true_rul", "mean", "sigma"):
    if c not in df: raise SystemExit(f"Column '{c}' missing. Found: {list(df.columns)}")
true, mean, sig = df.true_rul.values.astype(float), np.clip(df["mean"].values.astype(float), 0, a.cap), df.sigma.values.astype(float)


def nasa(d): return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))
def metrics(p):
    d = p - true
    return dict(rmse=np.sqrt(np.mean(d ** 2)), mae=np.mean(np.abs(d)), nasa=nasa(d), late=100 * np.mean(d > 0), early=100 * np.mean(d < 0))


print(f"N engines: {len(df)}   (prediction clipped to [0, {a.cap:g}])")
m = metrics(mean); print(f"\nBASE: RMSE {m['rmse']:.2f}  MAE {m['mae']:.2f}  NASA {m['nasa']:.1f}  late {m['late']:.1f}%  early {m['early']:.1f}%")

print(f"\nUNCERTAINTY (interval = mean +/- {a.z} sigma)")
lo, hi = mean - a.z * sig, mean + a.z * sig
cov_cap = 100 * np.mean((np.minimum(true, a.cap) >= lo) & (np.minimum(true, a.cap) <= hi)); cov_raw = 100 * np.mean((true >= lo) & (true <= hi))
print(f"  mean sigma {sig.mean():.2f}   mean interval width {np.mean(hi - lo):.1f}")
print(f"  90% coverage vs capped true RUL: {cov_cap:.1f}%   vs raw true RUL: {cov_raw:.1f}%   (nominal 90%)")
err = np.abs(mean - true); pear = float(np.corrcoef(sig, err)[0, 1]); spear = float(pd.Series(sig).rank().corr(pd.Series(err).rank()))
try:
    from scipy.stats import pearsonr, spearmanr
    print(f"  Pearson r {pear:.3f} (p={pearsonr(sig, err)[1]:.2g})   Spearman rho {spear:.3f} (p={spearmanr(sig, err)[1]:.2g})")
except Exception:
    print(f"  Pearson r {pear:.3f}   Spearman rho {spear:.3f}")
print("  -> " + ("close to nominal" if 85 <= cov_cap <= 95 else "OVER-CONFIDENT (intervals too narrow)" if cov_cap < 85 else "intervals wider than needed"))

print("\nSHIFT  y_shifted = y_hat - alpha * sigma")
print("  alpha    NASA   reduction   late%   early%    RMSE     MAE")
base_n = metrics(mean)["nasa"]
for al in (0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0):
    s = metrics(np.clip(mean - al * sig, 0, a.cap))
    print(f"  {al:4.1f}  {s['nasa']:7.1f}  {100 * (s['nasa'] - base_n) / base_n:+7.1f}%  {s['late']:6.1f}  {s['early']:7.1f}  {s['rmse']:7.2f}  {s['mae']:6.2f}")

print("\nERROR BY TRUE-RUL RANGE (raw true RUL)")
for lo_, hi_, name in [(0, 30, "[0,30]"), (30, 60, "(30,60]"), (60, a.cap, f"(60,{a.cap:g}]"), (a.cap, 1e9, f">{a.cap:g}")]:
    mk = (true >= lo_) & (true <= hi_) if lo_ == 0 else (true > lo_) & (true <= hi_)
    if mk.sum():
        d = mean[mk] - true[mk]
        print(f"  {name:10s} n={mk.sum():3d}  RMSE {np.sqrt(np.mean(d**2)):6.2f}  MAE {np.mean(np.abs(d)):6.2f}  NASA(sum) {nasa(d):8.1f}  late {100*np.mean(d>0):5.1f}%")

print("\nTOP-10 WORST ENGINES (index = row in the CSV, starting at 1)")
idx = np.argsort(-np.abs(mean - true))[:10]
print("  engine  true_RUL   pred    error   sigma   zone")
for i in idx:
    z = "critical" if true[i] <= 30 else "mid" if true[i] <= 60 else "early-life"
    print(f"  {i+1:5d}  {true[i]:8.0f}  {mean[i]:6.1f}  {mean[i]-true[i]:+7.1f}  {sig[i]:6.2f}   {z}")
print(f"  mean sigma of these 10: {sig[idx].mean():.2f}   fleet mean sigma: {sig.mean():.2f}   in early-life zone (true>60): {(true[idx] > 60).sum()} of 10")
