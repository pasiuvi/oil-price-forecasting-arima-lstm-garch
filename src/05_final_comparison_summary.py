"""
STEP 5: Model Comparison Summary (supports Task 5 write-up)

Reads:  outputs/model_comparison_results.csv (from Step 3)
        outputs/garch_summary.txt (from Step 4)
Writes: outputs/plot_12_final_summary.png
        outputs/final_summary.txt

Run this after 03_model_development.py and 04_garch_volatility.py
This does not create new models — it just pulls together everything you
already produced so you have one clean reference file for the Task 5
discussion, instead of digging through 4 different terminal outputs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("../outputs", exist_ok=True)

results_df = pd.read_csv("../outputs/model_comparison_results.csv")

print("=" * 60)
print("PRICE FORECASTING MODEL COMPARISON (Task 3 models)")
print("=" * 60)
print(results_df.to_string(index=False))

best_model_row = results_df.loc[results_df["RMSE"].idxmin()]
print(f"\nBest performing model by RMSE: {best_model_row['model']} (RMSE = {best_model_row['RMSE']:.4f})")

with open("../outputs/garch_summary.txt") as f:
    garch_text = f.read()

print("\n" + "=" * 60)
print("GARCH VOLATILITY MODEL SUMMARY (Task 4)")
print("=" * 60)
print(garch_text)

# Combined visual: price model RMSE next to GARCH volatility RMSE, side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(results_df["model"], results_df["RMSE"], color=["#4C72B0", "#DD8452", "#55A868"])
axes[0].set_title("Price Forecasting Models — RMSE")
axes[0].tick_params(axis="x", rotation=15)

garch_comparison = pd.read_csv("../outputs/garch_comparison_results.csv", parse_dates=["date"])
axes[1].plot(garch_comparison["date"], garch_comparison["realized_volatility"], label="Realized")
axes[1].plot(garch_comparison["date"], garch_comparison["predicted_volatility"], label="GARCH Forecast")
axes[1].set_title("GARCH — Predicted vs Realized Volatility")
axes[1].legend()
fig.autofmt_xdate()

plt.tight_layout()
plt.savefig("../outputs/plot_12_final_summary.png", dpi=150)
plt.close()
print("\nSaved outputs/plot_12_final_summary.png")

with open("../outputs/final_summary.txt", "w") as f:
    f.write("PRICE FORECASTING MODELS\n")
    f.write(results_df.to_string(index=False))
    f.write(f"\n\nBest model by RMSE: {best_model_row['model']}\n")
    f.write("\n\nGARCH VOLATILITY MODEL\n")
    f.write(garch_text)

print("Saved outputs/final_summary.txt — use these numbers directly in the")
print("Task 5 write-up. Send me this file's contents and I'll write Task 5 around it.")
