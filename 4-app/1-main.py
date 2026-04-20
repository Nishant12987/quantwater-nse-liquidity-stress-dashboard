import streamlit as st
import pandas as pd
import time
from pathlib import Path
import plotly.express as px
import sys

# =========================
# ROOT PATH (CLOUD SAFE)
# =========================

BASE = Path(__file__).resolve().parent.parent

DATA_PATH = BASE / "models" / "checkpoints"
SRC_PATH = BASE / "3-src" / "4-ensemble"

# Add backend path if exists
if SRC_PATH.exists():
    sys.path.append(str(SRC_PATH))

# =========================
# TRY BACKEND IMPORT
# =========================

backend_available = False

try:
    from dashboard_backend import backend_pipeline
    backend_available = True
except Exception as e:
    backend_error = str(e)

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="QuantWater Liquidity Risk Engine",
    layout="wide"
)

# =========================
# SIDEBAR
# =========================

st.sidebar.markdown("### QuantWater v1.0")

refresh = st.sidebar.slider("Auto Refresh (sec)", 0, 60, 0)

if refresh > 0:
    time.sleep(refresh)
    st.rerun()

# =========================
# HEADER
# =========================

st.title("QuantWater Liquidity Risk Engine")
st.markdown("### Institutional Portfolio Risk Monitoring System")
st.caption(f"Last updated: {pd.Timestamp.now()}")

# =========================
# LOAD DATA (SMART LOADER)
# =========================

def load_data():
    # PRIORITY 1: backend
    if backend_available:
        try:
            data = backend_pipeline()
            return data["results"], data["status"], data["value"]
        except Exception as e:
            st.warning(f"Backend failed, switching to fallback: {e}")

    # PRIORITY 2: CSV fallback
    file_path = DATA_PATH / "xgboost_full_predictions.csv.gz"

    if file_path.exists():
        df = pd.read_csv(file_path, compression="gzip")

        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.dropna(subset=["DATE"])

        status = "Fallback Mode"
        value = df["LSTM_SCORE"].mean()

        return df, status, value

    # FAILURE
    st.error("❌ No data source available (backend + CSV missing)")
    st.stop()

results, status, value = load_data()

# =========================
# SAFETY CHECK
# =========================

if results is None or results.empty:
    st.error("No data available.")
    st.stop()

# =========================
# PREP DATA
# =========================

latest = results.sort_values("DATE").groupby("SYMBOL").tail(1)

# =========================
# MARKET REGIME
# =========================

if value < 0.05:
    regime = "🟢 Liquidity Abundant"
elif value < 0.15:
    regime = "🟡 Tightening Liquidity"
else:
    regime = "🔴 Stress Regime"

st.markdown(f"### Market Regime: **{regime}**")

# =========================
# METRICS
# =========================

col1, col2, col3, col4 = st.columns(4)

col1.metric("Market Status", status)
col2.metric("Market Stress", round(float(value), 4))
col3.metric("Total Alerts", int(results.get("CRITICAL_ALERT", pd.Series([0])).sum()))
col4.metric("Active Stocks", results["SYMBOL"].nunique())

# =========================
# COLOR FUNCTION
# =========================

def risk_color(val):
    if val < 0.05:
        return "background-color: #0f5132"
    elif val < 0.15:
        return "background-color: #664d03"
    else:
        return "background-color: #842029"

# =========================
# TOP RISK TABLE
# =========================

st.subheader("Top Risk Stocks")

top_risk = latest.sort_values("LSTM_SCORE", ascending=False).head(10)

st.dataframe(
    top_risk.style.applymap(risk_color, subset=["LSTM_SCORE"]),
    width="stretch"
)

# =========================
# ALERTS
# =========================

st.subheader("Critical Alerts")

if "CRITICAL_ALERT" in latest.columns:
    alerts = latest[latest["CRITICAL_ALERT"] == 1]
else:
    alerts = pd.DataFrame()

if alerts.empty:
    st.success("No critical stress signals")
else:
    st.dataframe(alerts[["SYMBOL", "LSTM_SCORE"]], width="stretch")

# =========================
# MARKET TREND
# =========================

st.subheader("Market Stress Trend")

market_trend = results.groupby("DATE")["LSTM_SCORE"].mean().reset_index()

fig_trend = px.line(
    market_trend,
    x="DATE",
    y="LSTM_SCORE",
    title="Market Stress Over Time"
)

st.plotly_chart(fig_trend, width="stretch")

# =========================
# BAR CHART
# =========================

st.subheader("Top Risk Distribution")

fig_bar = px.bar(
    top_risk,
    x="SYMBOL",
    y="LSTM_SCORE",
    color="LSTM_SCORE"
)

st.plotly_chart(fig_bar, width="stretch")

# =========================
# SECTOR ANALYSIS
# =========================

sector_map = {
    "RELIANCE": "Energy",
    "HDFCBANK": "Banking",
    "ICICIBANK": "Banking",
    "INFY": "IT",
    "TCS": "IT",
}

results["SECTOR"] = results["SYMBOL"].map(sector_map).fillna("Other")

st.subheader("Sector Risk Distribution")

sector_data = results.groupby("SECTOR")["LSTM_SCORE"].mean().reset_index()

fig_sector = px.bar(
    sector_data,
    x="SECTOR",
    y="LSTM_SCORE",
    color="LSTM_SCORE"
)

st.plotly_chart(fig_sector, width="stretch")

# =========================
# PORTFOLIO BUILDER
# =========================

st.subheader("Portfolio Builder")

stocks = latest["SYMBOL"].unique()[:8]

weights = {}
cols = st.columns(len(stocks))

for i, stock in enumerate(stocks):
    weights[stock] = cols[i].slider(stock, 0.0, 1.0, 0.1)

total_weight = sum(weights.values())
portfolio_stress = 0

if total_weight > 0:
    weights = {k: v / total_weight for k, v in weights.items()}

    for stock, w in weights.items():
        val = latest[latest["SYMBOL"] == stock]["LSTM_SCORE"]
        if len(val) > 0:
            portfolio_stress += w * val.values[0]
else:
    st.warning("Total weight is zero")

# =========================
# RISK CLASSIFICATION
# =========================

def classify_risk(x):
    if x < 0.05:
        return "LOW"
    elif x < 0.15:
        return "MEDIUM"
    else:
        return "HIGH"

risk_label = classify_risk(portfolio_stress)

st.markdown(f"### Portfolio Stress: **{round(portfolio_stress, 4)}**")
st.markdown(f"### Risk Level: **{risk_label}**")

# =========================
# SIMULATION
# =========================

st.subheader("Stress Simulation")

shock = st.slider("Market Shock (%)", -50, 50, 0)

simulated = portfolio_stress * (1 + shock / 100)

st.metric("Simulated Stress", round(simulated, 4))

# =========================
# DOWNLOAD
# =========================

csv = results.to_csv(index=False)

st.download_button(
    "Download Risk Report",
    csv,
    "risk_report.csv",
    "text/csv"
)
