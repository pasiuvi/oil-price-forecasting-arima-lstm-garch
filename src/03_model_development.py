"""
STEP 3: Model Development — ARIMA, LSTM, Random Forest (Task 3)

Reads:  data/brent_oil_processed.csv
Writes: outputs/plot_4_arima_forecast.png
        outputs/plot_5_lstm_forecast.png
        outputs/plot_6_rf_forecast.png
        outputs/plot_7_feature_importance.png
        outputs/plot_8_model_comparison.png
        outputs/model_comparison_results.csv

Run this after 02_eda_preprocessing.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.makedirs("../outputs", exist_ok=True)

df = pd.read_csv("../data/brent_oil_processed.csv", parse_dates=["date"])

TEST_SIZE = 60
VAL_SIZE = 60

train_df = df.iloc[:-(TEST_SIZE + VAL_SIZE)].reset_index(drop=True)
val_df = df.iloc[-(TEST_SIZE + VAL_SIZE):-TEST_SIZE].reset_index(drop=True)
test_df = df.iloc[-TEST_SIZE:].reset_index(drop=True)

print(f"Train: {len(train_df)} rows | Validation: {len(val_df)} rows | Test: {len(test_df)} rows")

def evaluate(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100
    print(f"\n{model_name} — Test Set Performance")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    return {"model": model_name, "MAE": mae, "RMSE": rmse, "MAPE": mape}

results = []

# ===========================================================
# MODEL 1: ARIMA
# ===========================================================
print("\n" + "=" * 60)
print("MODEL 1: ARIMA")
print("=" * 60)

from statsmodels.tsa.arima.model import ARIMA
from itertools import product

history = pd.concat([train_df, val_df])["close"].reset_index(drop=True)
train_series = train_df["close"].reset_index(drop=True)

p_values = [0, 1, 2, 3]
d_values = [1]
q_values = [0, 1, 2]

best_score, best_order = float("inf"), None

for p, d, q in product(p_values, d_values, q_values):
    try:
        model = ARIMA(train_series, order=(p, d, q))
        fitted = model.fit()
        forecast = fitted.forecast(steps=len(val_df))
        rmse = np.sqrt(mean_squared_error(val_df["close"], forecast))
        if rmse < best_score:
            best_score = rmse
            best_order = (p, d, q)
    except Exception:
        continue

print(f"Best ARIMA order: {best_order} (val RMSE: {best_score:.4f})")

final_arima = ARIMA(history, order=best_order).fit()

# Rolling one-step-ahead forecast: at each test day, forecast only the NEXT
# day using everything known so far, then reveal the true value and move
# forward one step. This matches how LSTM and Random Forest are evaluated
# below (they are given the true previous day's price as a feature at every
# test point), so all three models end up compared on a fair, like-for-like
# basis instead of ARIMA doing a much harder 60-day-blind forecast.
rolling_model = final_arima
arima_forecast = []
for true_value in test_df["close"]:
    next_pred = rolling_model.forecast(steps=1)
    arima_forecast.append(next_pred.iloc[0])
    rolling_model = rolling_model.append([true_value], refit=False)

arima_forecast = pd.Series(arima_forecast, index=test_df.index)
results.append(evaluate(test_df["close"], arima_forecast, "ARIMA"))

plt.figure(figsize=(12, 5))
plt.plot(test_df["date"], test_df["close"], label="Actual")
plt.plot(test_df["date"], arima_forecast.values, label="ARIMA Forecast")
plt.title("ARIMA: Actual vs Forecast (Test Set)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../outputs/plot_4_arima_forecast.png", dpi=150)
plt.close()
print("Saved outputs/plot_4_arima_forecast.png")

# ===========================================================
# MODEL 2: LSTM
# ===========================================================
print("\n" + "=" * 60)
print("MODEL 2: LSTM")
print("=" * 60)

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

LOOKBACK = 30

scaler = MinMaxScaler()
full_series = df["close"].values.reshape(-1, 1)
scaled_full = scaler.fit_transform(full_series)

def make_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_all, y_all = make_sequences(scaled_full, LOOKBACK)

n_test = len(test_df)
n_val = len(val_df)

X_train, y_train = X_all[:-(n_test + n_val)], y_all[:-(n_test + n_val)]
X_val, y_val = X_all[-(n_test + n_val):-n_test], y_all[-(n_test + n_val):-n_test]
X_test, y_test = X_all[-n_test:], y_all[-n_test:]

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

configs = [
    {"units": 50, "dropout": 0.2, "layers": 1},
    {"units": 64, "dropout": 0.3, "layers": 2},
]

best_val_loss, best_model, best_config = float("inf"), None, None

for cfg in configs:
    model = Sequential()
    if cfg["layers"] == 1:
        model.add(LSTM(cfg["units"], input_shape=(LOOKBACK, 1)))
        model.add(Dropout(cfg["dropout"]))
    else:
        model.add(LSTM(cfg["units"], return_sequences=True, input_shape=(LOOKBACK, 1)))
        model.add(Dropout(cfg["dropout"]))
        model.add(LSTM(cfg["units"] // 2))
        model.add(Dropout(cfg["dropout"]))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history_fit = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50, batch_size=32,
        callbacks=[early_stop], verbose=0
    )

    val_loss = min(history_fit.history["val_loss"])
    print(f"Config {cfg} -> validation loss: {val_loss:.6f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        best_config = cfg

print(f"\nBest LSTM configuration: {best_config}")

lstm_pred_scaled = best_model.predict(X_test, verbose=0)
lstm_pred = scaler.inverse_transform(lstm_pred_scaled).flatten()
lstm_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

results.append(evaluate(lstm_actual, lstm_pred, "LSTM"))

plt.figure(figsize=(12, 5))
plt.plot(test_df["date"].values, lstm_actual, label="Actual")
plt.plot(test_df["date"].values, lstm_pred, label="LSTM Forecast")
plt.title("LSTM: Actual vs Forecast (Test Set)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../outputs/plot_5_lstm_forecast.png", dpi=150)
plt.close()
print("Saved outputs/plot_5_lstm_forecast.png")

# ===========================================================
# MODEL 3: RANDOM FOREST
# ===========================================================
print("\n" + "=" * 60)
print("MODEL 3: RANDOM FOREST")
print("=" * 60)

feature_cols = [
    "lag_1", "lag_7", "lag_30",
    "rolling_mean_7", "rolling_mean_30",
    "rolling_std_7", "rolling_std_30",
    "day_of_week", "month", "quarter", "is_month_end"
]

X = df[feature_cols]
y = df["close"]

X_train_rf = X.iloc[:-(TEST_SIZE + VAL_SIZE)]
y_train_rf = y.iloc[:-(TEST_SIZE + VAL_SIZE)]
X_test_rf = X.iloc[-TEST_SIZE:]
y_test_rf = y.iloc[-TEST_SIZE:]

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
}

tscv = TimeSeriesSplit(n_splits=5)
rf = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(rf, param_grid, cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1)
grid_search.fit(X_train_rf, y_train_rf)

print(f"Best Random Forest parameters: {grid_search.best_params_}")

best_rf = grid_search.best_estimator_
rf_pred = best_rf.predict(X_test_rf)
results.append(evaluate(y_test_rf, rf_pred, "Random Forest"))

plt.figure(figsize=(12, 5))
plt.plot(test_df["date"], y_test_rf.values, label="Actual")
plt.plot(test_df["date"], rf_pred, label="Random Forest Forecast")
plt.title("Random Forest: Actual vs Forecast (Test Set)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../outputs/plot_6_rf_forecast.png", dpi=150)
plt.close()
print("Saved outputs/plot_6_rf_forecast.png")

importances = pd.Series(best_rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importances:")
print(importances)

plt.figure(figsize=(8, 5))
importances.plot(kind="barh")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("../outputs/plot_7_feature_importance.png", dpi=150)
plt.close()
print("Saved outputs/plot_7_feature_importance.png")

# ===========================================================
# BASELINE: NAIVE PERSISTENCE FORECAST
# ===========================================================
print("\n" + "=" * 60)
print("BASELINE: NAIVE FORECAST (tomorrow's price = today's price)")
print("=" * 60)
print("This checks whether the models above are genuinely learning something,")
print("or just approximating the simplest possible guess.")

naive_forecast = test_df["lag_1"]  # yesterday's actual price, already a column
results.append(evaluate(test_df["close"], naive_forecast, "Naive Baseline"))

# ===========================================================
# FINAL COMPARISON
# ===========================================================
print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv("../outputs/model_comparison_results.csv", index=False)
print("\nSaved outputs/model_comparison_results.csv")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, metric in enumerate(["MAE", "RMSE", "MAPE"]):
    axes[i].bar(results_df["model"], results_df[metric], color=["#4C72B0", "#DD8452", "#55A868"])
    axes[i].set_title(metric)
    axes[i].tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig("../outputs/plot_8_model_comparison.png", dpi=150)
plt.close()
print("Saved outputs/plot_8_model_comparison.png")

print("\nDONE.")
