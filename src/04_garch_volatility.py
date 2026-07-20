"""
STEP 4: Volatility Modeling using GARCH (Task 4)

Reads:  data/brent_oil_processed.csv
Writes: outputs/plot_9_returns_volatility_clustering.png
        outputs/plot_10_acf_squared_returns.png
        outputs/plot_11_garch_volatility_forecast.png
        outputs/garch_summary.txt
        outputs/garch_comparison_results.csv

Run this after 02_eda_preprocessing.py (does not depend on Step 3)

WHAT THIS SCRIPT DOES, in plain terms:
- Oil prices don't move by a constant amount every day. Some periods are calm,
  some periods are wild (e.g. when OPEC makes an announcement or a war starts).
  This "calm periods followed by wild periods" pattern is called volatility
  clustering, and it's caused by something called conditional heteroscedasticity
  (a fancy way of saying "how much the price jumps around changes over time,
  and depends on how much it was jumping around recently").
- A GARCH model is built specifically to forecast HOW MUCH volatility to expect
  next, not the price itself. That's a different, complementary forecast to
  the ARIMA/LSTM/RF models in Step 3.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

from arch import arch_model
from statsmodels.stats.diagnostic import het_arch
from statsmodels.graphics.tsaplots import plot_acf

os.makedirs("../outputs", exist_ok=True)

TEST_SIZE = 60

df = pd.read_csv("../data/brent_oil_processed.csv", parse_dates=["date"])

# Use log returns, the standard input for GARCH models (more statistically
# well-behaved than raw percentage returns)
df["log_return"] = np.log(df["close"] / df["close"].shift(1))
df = df.dropna(subset=["log_return"]).reset_index(drop=True)

# arch package works more reliably in percentage units, not decimals
returns_pct = df["log_return"] * 100

# ---------------------------------------------------------
# STEP 1: Visual evidence of volatility clustering
# ---------------------------------------------------------
plt.figure(figsize=(14, 5))
plt.plot(df["date"], returns_pct, linewidth=0.6)
plt.title("Brent Oil Daily Returns — Visual Evidence of Volatility Clustering")
plt.xlabel("Date")
plt.ylabel("Daily Return (%)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../outputs/plot_9_returns_volatility_clustering.png", dpi=150)
plt.close()
print("Saved outputs/plot_9_returns_volatility_clustering.png")
print("Look at this plot: calm, tight bands of movement alternate with wide,")
print("noisy bands. That pattern IS volatility clustering.")

# ---------------------------------------------------------
# STEP 2: Statistical evidence — ACF of squared returns
# ---------------------------------------------------------
# If returns were truly random with no clustering, squared returns would show
# no autocorrelation. If there IS clustering, squared returns will be
# correlated with their own past values.
fig, ax = plt.subplots(figsize=(10, 5))
plot_acf(returns_pct**2, lags=40, ax=ax)
ax.set_title("Autocorrelation of Squared Returns (evidence of volatility clustering)")
plt.tight_layout()
plt.savefig("../outputs/plot_10_acf_squared_returns.png", dpi=150)
plt.close()
print("Saved outputs/plot_10_acf_squared_returns.png")

# ---------------------------------------------------------
# STEP 3: Engle's ARCH test — formal statistical confirmation
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("ENGLE'S ARCH TEST")
print("=" * 60)
arch_test_stat, arch_test_pvalue, _, _ = het_arch(returns_pct, nlags=10)
print(f"LM statistic: {arch_test_stat:.4f}")
print(f"p-value: {arch_test_pvalue:.6f}")
if arch_test_pvalue < 0.05:
    print("Result: ARCH effects ARE present (p < 0.05) — GARCH modelling is justified.")
else:
    print("Result: No strong evidence of ARCH effects.")

# ---------------------------------------------------------
# STEP 4: Train/test split (same test window as Step 3, for consistency)
# ---------------------------------------------------------
train_returns = returns_pct.iloc[:-TEST_SIZE].reset_index(drop=True)
test_returns = returns_pct.iloc[-TEST_SIZE:].reset_index(drop=True)
test_dates = df["date"].iloc[-TEST_SIZE:].reset_index(drop=True)

# ---------------------------------------------------------
# STEP 5: Fit GARCH(1,1) — the standard baseline
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("GARCH(1,1) MODEL")
print("=" * 60)

garch_model = arch_model(train_returns, vol="Garch", p=1, q=1, dist="normal")
garch_fit = garch_model.fit(disp="off")
print(garch_fit.summary())

omega = garch_fit.params["omega"]
alpha = garch_fit.params["alpha[1]"]
beta = garch_fit.params["beta[1]"]
persistence = alpha + beta

print(f"\nomega (baseline variance level): {omega:.4f}")
print(f"alpha (reaction to recent shocks): {alpha:.4f}")
print(f"beta (persistence of past volatility): {beta:.4f}")
print(f"alpha + beta (total persistence): {persistence:.4f}")
if persistence < 1:
    print("Persistence is below 1: volatility shocks fade out over time (mean-reverting),")
    print("which is the expected, stable behaviour for a well-specified GARCH model.")
else:
    print("Persistence is at or above 1: volatility shocks do not fade — check the model.")

# ---------------------------------------------------------
# STEP 6: Try GJR-GARCH too (captures asymmetric shocks — oil prices often
# react more strongly to bad news / supply shocks than good news, which
# plain GARCH(1,1) cannot capture)
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("GJR-GARCH(1,1) MODEL (captures asymmetric/leverage effects)")
print("=" * 60)

gjr_model = arch_model(train_returns, vol="Garch", p=1, o=1, q=1, dist="normal")
gjr_fit = gjr_model.fit(disp="off")
print(gjr_fit.summary())

print(f"\nGARCH(1,1)      AIC: {garch_fit.aic:.2f} | BIC: {garch_fit.bic:.2f}")
print(f"GJR-GARCH(1,1)  AIC: {gjr_fit.aic:.2f} | BIC: {gjr_fit.bic:.2f}")

if gjr_fit.aic < garch_fit.aic:
    print("GJR-GARCH fits better (lower AIC) — asymmetric shocks matter for this series.")
    final_model_fit = gjr_fit
    final_model_name = "GJR-GARCH(1,1)"
else:
    print("Plain GARCH(1,1) fits at least as well — no strong evidence of asymmetry.")
    final_model_fit = garch_fit
    final_model_name = "GARCH(1,1)"

# ---------------------------------------------------------
# STEP 7: Forecast volatility over the test period and compare to realized volatility
# ---------------------------------------------------------
print("\n" + "=" * 60)
print(f"VOLATILITY FORECAST vs REALIZED — using {final_model_name}")
print("=" * 60)

# Rolling one-step-ahead volatility forecast: uses the GARCH parameters
# estimated ONLY on the training period, but recursively updates using each
# day's true observed return as we move through the test period — so the
# forecast can actually react to real shocks, the same principle used to fix
# the ARIMA forecast earlier. A single 60-day-ahead forecast from one origin
# point (the old approach) just settles toward the long-run average variance
# and can never react to anything that happens during the test window.
final_params = final_model_fit.params
omega_f = final_params["omega"]
alpha_f = final_params["alpha[1]"]
gamma_f = final_params["gamma[1]"] if "gamma[1]" in final_params else 0.0
beta_f = final_params["beta[1]"]
mu_f = final_params["mu"]

h_prev = final_model_fit.conditional_volatility.iloc[-1] ** 2
e_prev = train_returns.iloc[-1] - mu_f

rolling_predicted_vol = []
for actual_return in test_returns:
    indicator = 1.0 if e_prev < 0 else 0.0
    h_next = omega_f + alpha_f * e_prev**2 + gamma_f * (e_prev**2) * indicator + beta_f * h_prev
    rolling_predicted_vol.append(np.sqrt(h_next))
    h_prev = h_next
    e_prev = actual_return - mu_f

predicted_vol = np.array(rolling_predicted_vol)

# Realized volatility proxy: absolute value of actual returns (simple, standard proxy)
realized_vol = test_returns.abs().values

comparison_df = pd.DataFrame({
    "date": test_dates,
    "predicted_volatility": predicted_vol,
    "realized_volatility": realized_vol
})
comparison_df.to_csv("../outputs/garch_comparison_results.csv", index=False)
print("Saved outputs/garch_comparison_results.csv")

vol_rmse = np.sqrt(np.mean((predicted_vol - realized_vol) ** 2))
print(f"\nVolatility forecast RMSE (predicted vs realized): {vol_rmse:.4f}")

plt.figure(figsize=(12, 5))
plt.plot(test_dates, realized_vol, label="Realized Volatility (|actual return|)")
plt.plot(test_dates, predicted_vol, label=f"{final_model_name} Forecast Volatility")
plt.title("GARCH Volatility Forecast vs Realized Volatility (Test Set)")
plt.xlabel("Date")
plt.ylabel("Volatility (%)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../outputs/plot_11_garch_volatility_forecast.png", dpi=150)
plt.close()
print("Saved outputs/plot_11_garch_volatility_forecast.png")

# ---------------------------------------------------------
# STEP 8: Save a plain-text summary for the report
# ---------------------------------------------------------
with open("../outputs/garch_summary.txt", "w") as f:
    f.write(f"Best model: {final_model_name}\n")
    f.write(f"ARCH test p-value: {arch_test_pvalue:.6f}\n")
    f.write(f"GARCH(1,1) omega: {omega:.4f}, alpha: {alpha:.4f}, beta: {beta:.4f}, persistence: {persistence:.4f}\n")
    f.write(f"GARCH(1,1) AIC: {garch_fit.aic:.2f}, BIC: {garch_fit.bic:.2f}\n")
    f.write(f"GJR-GARCH(1,1) AIC: {gjr_fit.aic:.2f}, BIC: {gjr_fit.bic:.2f}\n")
    f.write(f"Volatility forecast RMSE: {vol_rmse:.4f}\n")

print("\nSaved outputs/garch_summary.txt — open this file, you'll need these")
print("exact numbers for the Task 4 write-up.")
print("\nDONE.")
