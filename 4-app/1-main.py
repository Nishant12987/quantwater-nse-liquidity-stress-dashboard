import streamlit as st
import pandas as pd
import time
import sys
from pathlib import Path
import importlib.util

# =========================
# LOAD BACKEND (SAFE IMPORT)
# =========================

BASE = Path(__file__).resolve().parents[2]

backend_path = BASE / "3-src/4-ensemble/dashboard_backend.py"

spec = importlib.util.spec_from_file_location("dashboard_backend", backend_path)
backend_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend_module)

backend_pipeline = backend_module.backend_pipeline


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(layout="wide")


# =========================
# AUTO REFRESH
# =========================

refresh = st.sidebar.slider("Auto Refresh (sec)", 0, 60, 0)

if refresh > 0:
    time.sleep(refresh)
    st.rerun()


# =========================
# TITLE
# =========================

st.title("QuantWater Liquidity Risk Engine")
st.markdown("### Institutional Portfolio Risk Monitoring System")


# =========================
# LOAD DATA
# =========================

data = backend_pipeline()

results = data["results"]


# =========================
# HEADER METRICS
# =========================

col1, col2, col3, col4 = st.columns(4)

col1.metric("Market Status", data["status"])
col2.metric("Market Stress", round(data["value"], 4))
col3.metric("Total Alerts", int(results["CRITICAL_ALERT"].sum()))
col4.metric("Active Stocks", results["SYMBOL"].nunique())

st.caption(f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")


# =========================
# SECTOR MAPPING (REALISTIC)
# =========================

sector_map = {
    "AAREYDRUGS": "Pharma",
    "20MICRONS": "Materials",
    "3MINDIA": "Industrial",
    "360ONE": "Finance",
    "AAKASH": "Healthcare"
}

results["SECTOR"] = results["SYMBOL"].map(sector_map).fillna("Other")


# =========================
# TOP RISK STOCKS
# =========================

st.subheader("Top Risk Stocks")

top_risk = results.sort_values("LSTM_SCORE", ascending=False).head(5)

st.dataframe(top_risk[["SYMBOL", "LSTM_SCORE", "CRITICAL_ALERT"]])


# =========================
# PORTFOLIO BUILDER
# =========================

st.subheader("Portfolio Risk Simulator")

symbols = results["SYMBOL"].unique()

selected = st.multiselect("Select Stocks", symbols[:20])

weights = {}

for stock in selected:
    weights[stock] = st.slider(f"{stock} weight", 0.0, 1.0, 0.1)


# =========================
# NORMALIZE WEIGHTS
# =========================

if len(weights) > 0:

    total_weight = sum(weights.values())

    if total_weight == 0:
        st.error("Total weight cannot be zero")

    else:
        weights = {k: v / total_weight for k, v in weights.items()}

        latest = results.sort_values("DATE").groupby("SYMBOL").tail(1)

        portfolio_stress = 0

        for stock, w in weights.items():
            stock_val = latest[latest["SYMBOL"] == stock]["LSTM_SCORE"]

            if len(stock_val) > 0:
                portfolio_stress += w * stock_val.values[0]

        st.metric("Portfolio Stress", round(portfolio_stress, 4))


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

st.subheader("Export")

csv = results.to_csv(index=False)

st.download_button(
    "Download Risk Report",
    csv,
    "risk_report.csv",
    "text/csv"
)
