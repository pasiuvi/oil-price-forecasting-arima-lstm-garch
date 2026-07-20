"""
STEP 1: Download Brent Crude Oil price data

This downloads daily Brent Crude Oil futures prices from Yahoo Finance
and saves them into the data/ folder.

Run this first, before any other script.
"""

import yfinance as yf
import os

ticker = "BZ=F"

df = yf.download(ticker, start="2015-01-01", end="2025-12-31", auto_adjust=True)
df.reset_index(inplace=True)
df = df[["Date", "Close", "Open", "High", "Low", "Volume"]]
df.columns = ["date", "close", "open", "high", "low", "volume"]

os.makedirs("../data", exist_ok=True)
df.to_csv("../data/brent_oil_prices.csv", index=False)

print("Saved data/brent_oil_prices.csv")
print(f"Shape: {df.shape}")
print(df.head())
print(df.tail())
