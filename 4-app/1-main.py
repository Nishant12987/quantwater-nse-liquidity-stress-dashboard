import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

# =========================
# PATH SETUP
# =========================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "3-src"))
sys.path.insert(0, str(ROOT / "3-src/4-ensemble"))

from dashboard_backend import backend_pipeline

# =========================
# CONFIG
# =========================

st.set_page_config(page_title="Liquidity Risk Terminal", layout="wide")

st.title("📊 Liquidity Risk Terminal — Live Portfolio Mode")

# =========================
# AUTO REFRESH (LIVE FEEL)
# =========================

st.sidebar.write("⏱ Auto Refresh (simulate live)")
refresh = st.sidebar.slider("Seconds", 0, 60, 0)

if refresh > 0:
    time.sleep(refresh)
    st.rerun()

# =========================
# LOAD DATA
# =========================

data = backend_pipeline()
df = pd.read_csv("models/checkpoints/xgboost_full_predictions.csv")

df["DATE"] = pd.to_datetime(df["DATE"])

latest_df = df.sort_values("DATE").groupby("SYMBOL").tail(1)

# =========================
# SECTOR MAPPING (MANUAL)
# =========================

sector_map = {
    "HDFCBANK": "Banking",
    "ICICIBANK": "Banking",
    "SBIN": "Banking",
    "TCS": "IT",
    "INFY": "IT",
    "WIPRO": "IT",
    "RELIANCE": "Energy",
    "ONGC": "Energy",
    "ADANIENT": "Energy",
    "BHARTIARTL": "Telecom",
    "IDEA": "Telecom",
}

latest_df["SECTOR"] = latest_df["SYMBOL"].map(sector_map).fillna("Other")

# =========================
# PORTFOLIO BUILDER
# =========================

st.sidebar.header("📂 Build Portfolio")

symbols = sorted(df["SYMBOL"].unique())

selected_stocks = st.sidebar.multiselect(
    "Select Stocks",
    symbols,
    default=symbols[:5]
)

portfolio = []

for stock in selected_stocks:
    weight = st.sidebar.slider(f"{stock} weight", 0.0, 1.0, 0.1)
    portfolio.append((stock, weight))

portfolio_df = pd.DataFrame(portfolio, columns=["SYMBOL", "WEIGHT"])

if portfolio_df["WEIGHT"].sum() > 0:
    portfolio_df["WEIGHT"] /= portfolio_df["WEIGHT"].sum()

portfolio_df = portfolio_df.merge(
    latest_df,
    on="SYMBOL",
    how="left"
)

# =========================
# PORTFOLIO METRICS
# =========================

portfolio_stress = (portfolio_df["WEIGHT"] * portfolio_df["PRED_PROBA"]).sum()

st.metric("📊 Portfolio Stress Score", round(portfolio_stress, 4))

# =========================
# SECTOR RISK
# =========================

st.subheader("🏦 Sector Risk View")

sector_risk = (
    portfolio_df
    .groupby("SECTOR")
    .apply(lambda x: (x["WEIGHT"] * x["PRED_PROBA"]).sum())
    .reset_index(name="SECTOR_RISK")
)

fig = px.bar(sector_risk, x="SECTOR", y="SECTOR_RISK", color="SECTOR")
st.plotly_chart(fig, use_container_width=True)

# =========================
# PORTFOLIO TREND
# =========================

st.subheader("📈 Portfolio Trend")

trend_list = []

for symbol, weight in zip(portfolio_df["SYMBOL"], portfolio_df["WEIGHT"]):
    temp = df[df["SYMBOL"] == symbol].copy()
    temp["WEIGHTED"] = temp["PRED_PROBA"] * weight
    trend_list.append(temp[["DATE", "WEIGHTED"]])

portfolio_trend = pd.concat(trend_list)
portfolio_trend = portfolio_trend.groupby("DATE")["WEIGHTED"].sum().reset_index()

fig = px.line(portfolio_trend, x="DATE", y="WEIGHTED")
st.plotly_chart(fig, use_container_width=True)

# =========================
# LIVE TOP STRESS
# =========================

st.subheader("🔥 Live Market Stress Leaders")

top = latest_df.sort_values("PRED_PROBA", ascending=False).head(10)

st.dataframe(top[["SYMBOL", "PRED_PROBA", "SECTOR"]])

# =========================
# SIGNAL ENGINE
# =========================

st.subheader("🧠 Signal Engine")

if portfolio_stress > 0.7:
    st.error("🔴 High Portfolio Risk")
elif portfolio_stress > 0.4:
    st.warning("🟡 Moderate Risk")
else:
    st.success("🟢 Healthy Portfolio")

# =========================
# INSIGHT
# =========================

top_sector = sector_risk.sort_values("SECTOR_RISK", ascending=False).iloc[0]

st.write(f"⚠️ Highest Risk Sector: {top_sector['SECTOR']}")
