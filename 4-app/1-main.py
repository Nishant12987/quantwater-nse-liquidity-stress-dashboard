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

# =========================
# PREMIUM CSS
# =========================

st.markdown("""
<style>
body { background-color: #0e1117; color: white; }

.block-container {
    padding-top: 1.5rem;
}

.card {
    background-color: #1c1f26;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 15px;
}

.metric-title {
    color: #9aa4b2;
    font-size: 13px;
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
}

.section-title {
    font-size: 18px;
    margin-bottom: 10px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================

st.markdown("## 📊 Liquidity Risk Terminal")
st.markdown("#### Hedge Fund Portfolio Risk System")
st.markdown("---")

# =========================
# AUTO REFRESH
# =========================

refresh = st.sidebar.slider("⏱ Auto Refresh (sec)", 0, 60, 0)

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
# SECTOR MAP
# =========================

sector_map = {
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking",
    "AXISBANK": "Banking",

    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT",

    "RELIANCE": "Energy", "ONGC": "Energy",
    "ADANIENT": "Energy", "ADANIPORTS": "Energy",

    "BHARTIARTL": "Telecom", "IDEA": "Telecom",

    "TATAMOTORS": "Auto", "MARUTI": "Auto"
}

latest_df["SECTOR"] = latest_df["SYMBOL"].map(sector_map).fillna("Other")

# =========================
# PORTFOLIO BUILDER
# =========================

st.sidebar.header("📂 Portfolio")

symbols = sorted(df["SYMBOL"].unique())

selected = st.sidebar.multiselect(
    "Select Stocks",
    symbols,
    default=symbols[:5]
)

portfolio = []
for s in selected:
    w = st.sidebar.slider(f"{s}", 0.0, 1.0, 0.1)
    portfolio.append((s, w))

portfolio_df = pd.DataFrame(portfolio, columns=["SYMBOL", "WEIGHT"])

if portfolio_df["WEIGHT"].sum() > 0:
    portfolio_df["WEIGHT"] /= portfolio_df["WEIGHT"].sum()

portfolio_df = portfolio_df.merge(latest_df, on="SYMBOL", how="left")

# =========================
# KPI CARDS
# =========================

status = data.get("status", "UNKNOWN")
value = data.get("value", 0)
alerts = int(portfolio_df["PRED_STRESS"].sum())

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">Market Status</div>
        <div class="metric-value">{status}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">Portfolio Stress</div>
        <div class="metric-value">{value:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">Stocks in Stress</div>
        <div class="metric-value">{alerts}</div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# GAUGE
# =========================

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=value,
    title={'text': "Market Stress"},
    gauge={
        'axis': {'range': [0, 3]},
        'bar': {'color': "red"},
        'steps': [
            {'range': [0, 0.5], 'color': "green"},
            {'range': [0.5, 1.5], 'color': "yellow"},
            {'range': [1.5, 3], 'color': "red"}
        ]
    }
))

st.plotly_chart(fig, use_container_width=True)

# =========================
# TREND
# =========================

st.markdown('<div class="section-title">📈 Portfolio Trend</div>', unsafe_allow_html=True)

trend_list = []

for s, w in zip(portfolio_df["SYMBOL"], portfolio_df["WEIGHT"]):
    temp = df[df["SYMBOL"] == s].copy()
    temp["W"] = temp["PRED_PROBA"] * w
    trend_list.append(temp[["DATE", "W"]])

trend = pd.concat(trend_list).groupby("DATE")["W"].sum().reset_index()

fig = px.line(trend, x="DATE", y="W")
st.plotly_chart(fig, use_container_width=True)

# =========================
# SECTOR VIEW
# =========================

st.markdown('<div class="section-title">🏦 Sector Risk</div>', unsafe_allow_html=True)

sector = (
    portfolio_df
    .groupby("SECTOR")
    .apply(lambda x: (x["WEIGHT"] * x["PRED_PROBA"]).sum())
    .reset_index(name="RISK")
    .sort_values("RISK", ascending=False)
)

fig = px.bar(sector, x="SECTOR", y="RISK", color="RISK")
st.plotly_chart(fig, use_container_width=True)

# =========================
# TOP RISK DRIVER
# =========================

top = portfolio_df.sort_values("PRED_PROBA", ascending=False).iloc[0]

st.markdown("### 🎯 Top Risk Driver")
st.error(f"⚠️ {top['SYMBOL']} is contributing highest risk")

# =========================
# ALERTS
# =========================

st.markdown("### 🚨 Alerts")

alerts_df = portfolio_df[portfolio_df["PRED_STRESS"] == 1]

if not alerts_df.empty:
    st.dataframe(alerts_df[["SYMBOL", "PRED_PROBA", "WEIGHT"]])
else:
    st.success("No stress in portfolio")

# =========================
# SIGNAL ENGINE
# =========================

st.markdown("### 🧠 Decision Signal")

if value < 0.5:
    st.success("🟢 Healthy Portfolio")
elif value < 1.5:
    st.warning("🟡 Monitor Closely")
else:
    st.error("🔴 Reduce Exposure")
