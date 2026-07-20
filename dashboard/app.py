"""
Interactive dashboard for the Brent Crude Oil Forecasting project.

Run locally with:  streamlit run app.py   (from inside the dashboard/ folder)
Deploy for free at: https://share.streamlit.io  (see INSTRUCTIONS.md)

This dashboard reads the CSV/PNG outputs already produced by the src/ scripts
in Steps 1-5. Run those scripts first, or the dashboard will show a message
telling you what's missing.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Brent Crude Analytics",
    page_icon="🛢️",
    layout="wide",
)

DATA_DIR = "../data"
OUT_DIR = "../outputs"

# ---------------------------------------------------------
# Professional styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    [data-testid="stMetric"] {
        background-color: #f8f9fb;
        border: 1px solid #e6e8eb;
        border-radius: 10px;
        padding: 16px 18px;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: #6b7280;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }
    h1 {
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .dashboard-tagline {
        color: #6b7280;
        font-size: 1.05rem;
        margin-top: -8px;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛢️ Brent Crude Analytics")
st.markdown('<p class="dashboard-tagline">Price forecasting and volatility intelligence for Brent Crude Oil</p>', unsafe_allow_html=True)

def file_exists(path):
    return os.path.exists(path)

# ---------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Price Forecasts", "Volatility (GARCH)", "Model Comparison", "About this project"]
)

# ---------------------------------------------------------
# PAGE: Overview
# ---------------------------------------------------------
if page == "Overview":
    processed_path = f"{DATA_DIR}/brent_oil_processed.csv"
    if file_exists(processed_path):
        df = pd.read_csv(processed_path, parse_dates=["date"])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latest Price", f"${df['close'].iloc[-1]:.2f}")
        col2.metric("30-Day Change", f"{((df['close'].iloc[-1] / df['close'].iloc[-30]) - 1) * 100:.2f}%")
        col3.metric("Data Points", f"{len(df):,}")
        years_covered = (df["date"].max() - df["date"].min()).days / 365.25
        col4.metric("History Covered", f"{years_covered:.1f} years")

        st.caption(f"Data from {df['date'].min().date()} to {df['date'].max().date()}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=df["close"], mode="lines", name="Close Price",
                                  line=dict(color="#1f77b4", width=1.5)))
        fig.add_trace(go.Scatter(x=df["date"], y=df["rolling_mean_30"], mode="lines", name="30-Day Moving Avg",
                                  line=dict(color="orange", width=1, dash="dash")))
        fig.update_layout(title="Brent Crude Oil Price History", xaxis_title="Date",
                           yaxis_title="Price (USD)", height=500, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Daily Returns Volatility")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df["date"], y=df["daily_return"] * 100, mode="lines",
                                   line=dict(color="crimson", width=0.6)))
        fig2.update_layout(xaxis_title="Date", yaxis_title="Daily Return (%)", height=300)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Run the src/ scripts first (Steps 1 and 2) to generate the data this page needs.")

# ---------------------------------------------------------
# PAGE: Price Forecasts
# ---------------------------------------------------------
elif page == "Price Forecasts":
    st.header("Model Forecasts vs Actual Price")
    forecast_plots = {
        "ARIMA": f"{OUT_DIR}/plot_4_arima_forecast.png",
        "LSTM": f"{OUT_DIR}/plot_5_lstm_forecast.png",
        "Random Forest": f"{OUT_DIR}/plot_6_rf_forecast.png",
    }
    tabs = st.tabs(list(forecast_plots.keys()))
    for tab, (name, path) in zip(tabs, forecast_plots.items()):
        with tab:
            if file_exists(path):
                st.image(path, use_container_width=True)
            else:
                st.info(f"Run 03_model_development.py to generate the {name} forecast plot.")

    st.header("Random Forest — Feature Importance")
    fi_path = f"{OUT_DIR}/plot_7_feature_importance.png"
    if file_exists(fi_path):
        st.image(fi_path, use_container_width=True)

# ---------------------------------------------------------
# PAGE: Volatility (GARCH)
# ---------------------------------------------------------
elif page == "Volatility (GARCH)":
    st.header("Volatility Clustering — Visual Evidence")
    col1, col2 = st.columns(2)
    with col1:
        if file_exists(f"{OUT_DIR}/plot_9_returns_volatility_clustering.png"):
            st.image(f"{OUT_DIR}/plot_9_returns_volatility_clustering.png", use_container_width=True)
    with col2:
        if file_exists(f"{OUT_DIR}/plot_10_acf_squared_returns.png"):
            st.image(f"{OUT_DIR}/plot_10_acf_squared_returns.png", use_container_width=True)

    st.header("GARCH Forecast vs Realized Volatility")
    garch_csv = f"{OUT_DIR}/garch_comparison_results.csv"
    if file_exists(garch_csv):
        gdf = pd.read_csv(garch_csv, parse_dates=["date"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gdf["date"], y=gdf["realized_volatility"], name="Realized Volatility"))
        fig.add_trace(go.Scatter(x=gdf["date"], y=gdf["predicted_volatility"], name="GARCH Forecast"))
        fig.update_layout(height=450, xaxis_title="Date", yaxis_title="Volatility (%)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run 04_garch_volatility.py to generate this data.")

    summary_path = f"{OUT_DIR}/garch_summary.txt"
    if file_exists(summary_path):
        with open(summary_path) as f:
            st.text(f.read())

# ---------------------------------------------------------
# PAGE: Model Comparison
# ---------------------------------------------------------
elif page == "Model Comparison":
    st.header("Price Forecasting Models — Metric Comparison")
    results_path = f"{OUT_DIR}/model_comparison_results.csv"
    if file_exists(results_path):
        results_df = pd.read_csv(results_path)
        st.dataframe(results_df, use_container_width=True)

        fig = go.Figure()
        for metric in ["MAE", "RMSE", "MAPE"]:
            fig.add_trace(go.Bar(x=results_df["model"], y=results_df[metric], name=metric))
        fig.update_layout(barmode="group", height=450, yaxis_title="Error")
        st.plotly_chart(fig, use_container_width=True)

        best = results_df.loc[results_df["RMSE"].idxmin()]
        st.success(f"Best performing model by RMSE: **{best['model']}** ({best['RMSE']:.4f})")
    else:
        st.info("Run 03_model_development.py to generate this data.")

# ---------------------------------------------------------
# PAGE: About
# ---------------------------------------------------------
else:
    st.header("About this project")
    st.markdown("""
    This platform forecasts Brent Crude Oil prices and models market volatility,
    comparing traditional statistical, machine learning, and deep learning
    approaches on a single, consistent dataset.

    **Dataset:** Brent Crude Oil daily futures prices (ticker `BZ=F`)

    **Models compared:**
    - ARIMA — traditional statistical forecasting
    - LSTM — deep learning sequence model
    - Random Forest — machine learning ensemble
    - GARCH(1,1) / GJR-GARCH — volatility and risk modelling

    **Why oil, not a stock index:** oil prices feed directly into global
    logistics and freight cost planning, giving this analysis a genuine
    operational angle rather than a purely academic one.
    """)
