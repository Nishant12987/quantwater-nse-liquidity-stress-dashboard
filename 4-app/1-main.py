import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import time
import plotly.express as px

# =========================
# PATH SETUP (FIXED)
# =========================

ROOT = Path(__file__).resolve().parents[1]

# Only correct path
sys.path.append(str(ROOT / "3-src/4-ensemble"))

from dashboard_backend import backend_pipeline


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
# LOAD DATA (ONLY BACKEND)
# =========================

data = backend_pipeline()

results = data["results"]
status = data["status"]
value = data["value"]


# =========================
# HEADER
# =========================

st.title("QuantWater Liquidity Risk Engine")
st.markdown("### Institutional Portfolio Risk Monitoring System")

st.caption(f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")


# =========================
# SAFETY CHECK
# =========================

if results.empty:
    st.error("No data available. Check backend or files.")
    st.stop()


# =========================
# LATEST DATA
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
col2.metric("Market Stress", round(value, 4))
col3.metric("Total Alerts", int(results["CRITICAL_ALERT"].sum()))
col4.metric("Active Stocks", results["SYMBOL"].nunique())


# =========================
# TOP RISK TABLE
# =========================

def risk_color(val):
    if val < 0.05:
        return "background-color: #0f5132"
    elif val < 0.15:
        return "background-color: #664d03"
    else:
        return "background-color: #842029"

st.subheader("Top Risk Stocks")

top_risk = latest.sort_values("LSTM_SCORE", ascending=False).head(10)

st.dataframe(
    top_risk.style.applymap(risk_color, subset=["LSTM_SCORE"]),
    width="stretch"
)


# =========================
# CRITICAL ALERTS
# =========================

alerts = latest[latest["CRITICAL_ALERT"] == 1]

st.subheader("Critical Alerts")

if alerts.empty:
    st.success("No critical stress signals")
else:
    st.dataframe(
        alerts[["SYMBOL", "LSTM_SCORE"]],
        width="stretch"
    )


# =========================
# MARKET TREND
# =========================

st.subheader("Market Stress Trend")

market_trend = (
    results.groupby("DATE")["LSTM_SCORE"]
    .mean()
    .reset_index()
)

fig_trend = px.line(
    market_trend,
    x="DATE",
    y="LSTM_SCORE",
    title="Market Stress Over Time"
)

st.plotly_chart(fig_trend, width="stretch")


# =========================
# TOP RISK BAR
# =========================

st.subheader("Top Risk Distribution")

fig_bar = px.bar(
    top_risk,
    x="SYMBOL",
    y="LSTM_SCORE",
    color="LSTM_SCORE",
    title="Top Risk Stocks"
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

sector_data = (
    results.groupby("SECTOR")["LSTM_SCORE"]
    .mean()
    .reset_index()
)

fig_sector = px.bar(
    sector_data,
    x="SECTOR",
    y="LSTM_SCORE",
    color="LSTM_SCORE",
    title="Sector-wise Risk"
)

st.plotly_chart(fig_sector, width="stretch")


# =========================
# PORTFOLIO BUILDER
# =========================

st.subheader("Portfolio Builder")

stocks = latest["SYMBOL"].unique()[:10]

weights = {}
cols = st.columns(len(stocks))

for i, stock in enumerate(stocks):
    weights[stock] = cols[i].slider(stock, 0.0, 1.0, 0.1)

total_weight = sum(weights.values())

portfolio_stress = 0

if total_weight == 0:
    st.error("Total weight cannot be zero")
else:
    weights = {k: v / total_weight for k, v in weights.items()}

    for stock, w in weights.items():
        val = latest[latest["SYMBOL"] == stock]["LSTM_SCORE"]
        if len(val) > 0:
            portfolio_stress += w * val.values[0]


# =========================
# PORTFOLIO RISK
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
st.markdown(f"### Portfolio Risk Level: **{risk_label}**")


# =========================
# SCENARIO SIMULATION
# =========================

st.subheader("Stress Simulation")

shock = st.slider("Market Shock (%)", -50, 50, 0)

simulated = portfolio_stress * (1 + shock / 100)

st.metric("Simulated Stress", round(simulated, 4))


# =========================
# DOWNLOAD REPORT
# =========================

csv = results.to_csv(index=False)

st.download_button(
    "Download Risk Report",
    csv,
    "risk_report.csv",
    "text/csv"
)
