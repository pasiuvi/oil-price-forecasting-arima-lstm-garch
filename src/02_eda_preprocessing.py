"""
STEP 2: Time Series Exploration and Preprocessing (Task 2)

Reads:  data/brent_oil_prices.csv
Writes: data/brent_oil_processed.csv
        outputs/plot_1_raw_series.png
        outputs/plot_2_decomposition.png
        outputs/plot_3_outliers.png

Run this after 01_download_data.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

os.makedirs("../outputs", exist_ok=True)

# ---------------------------------------------------------
# STEP 1: Load data and basic overview
# ---------------------------------------------------------
df = pd.read_csv("../data/brent_oil_prices.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

print("=" * 60)
print("BASIC OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(df.head())
print(df.tail())
print("\nMissing values per column:")
print(df.isnull().sum())
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# ---------------------------------------------------------
# STEP 2: Handle missing values and duplicates
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("CLEANING")
print("=" * 60)

before = len(df)
df = df.drop_duplicates()
print(f"Removed {before - len(df)} duplicate rows")

df = df.set_index("date")
full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
df = df.reindex(full_range)
df.index.name = "date"

missing_before = df["close"].isnull().sum()
df = df.ffill()
print(f"Filled {missing_before} non-trading-day gaps using forward fill")
print(f"Remaining missing values: {df.isnull().sum().sum()}")

df = df.reset_index()

# ---------------------------------------------------------
# STEP 3: Visualize the raw series
# ---------------------------------------------------------
plt.figure(figsize=(14, 5))
plt.plot(df["date"], df["close"], linewidth=0.8)
plt.title("Brent Crude Oil Daily Closing Price")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../outputs/plot_1_raw_series.png", dpi=150)
plt.close()
print("\nSaved outputs/plot_1_raw_series.png")

# ---------------------------------------------------------
# STEP 4: Decomposition
# ---------------------------------------------------------
series = df.set_index("date")["close"]
decomposition = seasonal_decompose(series, model="additive", period=365, extrapolate_trend="freq")

fig = decomposition.plot()
fig.set_size_inches(14, 8)
plt.tight_layout()
plt.savefig("../outputs/plot_2_decomposition.png", dpi=150)
plt.close()
print("Saved outputs/plot_2_decomposition.png")

# ---------------------------------------------------------
# STEP 5: Stationarity test
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STATIONARITY TEST (ADF)")
print("=" * 60)

def run_adf_test(series, label):
    result = adfuller(series.dropna())
    print(f"\n{label}")
    print(f"  ADF Statistic: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    if result[1] <= 0.05:
        print("  Result: Series IS stationary")
    else:
        print("  Result: Series is NOT stationary")
    return result[1]

run_adf_test(df["close"], "Raw closing price")
df["close_diff"] = df["close"].diff()
run_adf_test(df["close_diff"], "First-differenced closing price")

# ---------------------------------------------------------
# STEP 6: Outlier detection
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("OUTLIER DETECTION")
print("=" * 60)

df["daily_return"] = df["close"].pct_change()
Q1 = df["daily_return"].quantile(0.25)
Q3 = df["daily_return"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 3 * IQR
upper_bound = Q3 + 3 * IQR

outliers = df[(df["daily_return"] < lower_bound) | (df["daily_return"] > upper_bound)]
print(f"Number of outlier days detected: {len(outliers)}")
print(outliers[["date", "close", "daily_return"]].head(10))

plt.figure(figsize=(14, 4))
plt.plot(df["date"], df["daily_return"], linewidth=0.6)
plt.scatter(outliers["date"], outliers["daily_return"], color="red", s=15, label="Outlier days")
plt.title("Daily Returns with Outliers Highlighted")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../outputs/plot_3_outliers.png", dpi=150)
plt.close()
print("Saved outputs/plot_3_outliers.png")

# ---------------------------------------------------------
# STEP 7: Feature engineering
# ---------------------------------------------------------
df["lag_1"] = df["close"].shift(1)
df["lag_7"] = df["close"].shift(7)
df["lag_30"] = df["close"].shift(30)
df["rolling_mean_7"] = df["close"].rolling(window=7).mean()
df["rolling_mean_30"] = df["close"].rolling(window=30).mean()
df["rolling_std_7"] = df["close"].rolling(window=7).std()
df["rolling_std_30"] = df["close"].rolling(window=30).std()
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter
df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

df_clean = df.dropna().reset_index(drop=True)
print(f"\nRows before dropping NaN: {len(df)} | after: {len(df_clean)}")

df_clean.to_csv("../data/brent_oil_processed.csv", index=False)
print("\nDONE — saved data/brent_oil_processed.csv")
