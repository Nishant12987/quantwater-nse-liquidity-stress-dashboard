import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import time

# =========================
# PATH SETUP
# =========================

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "3-src"))
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
# AUTO REFRESH
# =========================

refresh = st.sidebar.slider("Auto Refresh (sec)", 0, 60, 0)

if refresh > 0:
    time.sleep(refresh)
    st.rerun()


# =========================
# LOAD DATA
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
# TOP METRICS
# =========================

col1, col2, col3, col4 = st.columns(4)

col1.metric("Market Status", status)
col2.metric("Market Stress", round(value, 4) if value else 0)
col3.metric("Total Alerts", int(results["CRITICAL_ALERT"].sum()) if not results.empty else 0)
col4.metric("Active Stocks", results["SYMBOL"].nunique() if not results.empty else 0)


# =========================
# SAFETY CHECK
# =========================

if results.empty:
    st.error("No data available. Check backend or file paths.")
    st.stop()


# =========================
# LATEST SNAPSHOT
# =========================

latest = results.sort_values("DATE").groupby("SYMBOL").tail(1)


# =========================
# TOP RISK STOCKS TABLE
# =========================

st.subheader("Top Risk Stocks")

top_risk = latest.sort_values("LSTM_SCORE", ascending=False).head(5)

st.dataframe(
    top_risk[["SYMBOL", "LSTM_SCORE", "CRITICAL_ALERT"]],
    use_container_width=True
)


# =========================
# 📈 MARKET STRESS TREND
# =========================

st.subheader("Market Stress Trend")

market_trend = (
    results.groupby("DATE")["LSTM_SCORE"]
    .mean()
    .reset_index()
)

st.line_chart(market_trend.set_index("DATE"))


# =========================
# 📊 TOP RISK BAR CHART
# =========================

st.subheader("Top Risk Distribution")

top10 = latest.sort_values("LSTM_SCORE", ascending=False).head(10)

st.bar_chart(top10.set_index("SYMBOL")["LSTM_SCORE"])


# =========================
# 🧠 SECTOR HEATMAP
# =========================

sector_map = {
    "RELIANCE": "Energy",
    "HDFCBANK": "Banking",
    "INFY": "IT",
    "TCS": "IT",
    "ICICIBANK": "Banking",
}

results["SECTOR"] = results["SYMBOL"].map(sector_map).fillna("Other")

st.subheader("Sector Risk Heatmap")

sector_data = (
    results.groupby("SECTOR")["LSTM_SCORE"]
    .mean()
    .sort_values(ascending=False)
)

st.bar_chart(sector_data)


# =========================
# PORTFOLIO BUILDER
# =========================

st.subheader("Portfolio Builder")

stocks = latest["SYMBOL"].unique()[:10]

weights = {}

cols = st.columns(len(stocks))

for i, stock in enumerate(stocks):
    weights[stock] = cols[i].slider(stock, 0.0, 1.0, 0.1)


# Normalize weights
total_weight = sum(weights.values())

if total_weight == 0:
    st.error("Total weight cannot be zero")
    portfolio_stress = 0
else:
    weights = {k: v / total_weight for k, v in weights.items()}

    portfolio_stress = 0

    for stock, w in weights.items():
        stock_val = latest[latest["SYMBOL"] == stock]["LSTM_SCORE"]
        if len(stock_val) > 0:
            portfolio_stress += w * stock_val.values[0]


# =========================
# PORTFOLIO METRICS
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

simulated_stress = portfolio_stress * (1 + shock / 100)

st.metric("Simulated Stress", round(simulated_stress, 4))


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
