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
# CSS (IMPROVED)
# =========================

st.markdown("""
<style>
body { background-color: #0e1117; color: white; }

.block-container {
    padding-top: 1.5rem;
}

.card {
    padding: 18px;
    border-radius: 12px;
    text-align: center;
}

.metric-title {
    color: #9aa4b2;
    font-size: 13px;
}

.metric-value {
    font-size: 26px;
    font-weight: bold;
}

.section-title {
    font-size: 18px;
    margin-top: 20px;
    margin-bottom: 10px;
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
# EXPANDED SECTOR MAP
# =========================

sector_map = {
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking",
    "AXISBANK": "Banking", "KOTAKBANK": "Banking",

    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT",

    "RELIANCE": "Energy", "ONGC": "Energy",
    "ADANIENT": "Energy", "ADANIPORTS": "Energy",

    "BHARTIARTL": "Telecom", "IDEA": "Telecom",

    "TATAMOTORS": "Auto", "MARUTI": "Auto",

    "ITC": "FMCG", "HINDUNILVR": "FMCG",
    "LT": "Infra", "ULTRACEMCO": "Infra",
    "BAJFINANCE": "NBFC"
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
# KPI CARDS (UPGRADED)
# =========================

def kpi_card(title, value, color):
    st.markdown(f"""
    <div class="card" style="background-color:{color};">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

status = data.get("status", "UNKNOWN")
value = data.get("value", 0)
alerts = int(portfolio_df["PRED_STRESS"].sum())

col1, col2, col3 = st.columns(3)

kpi_card("Market Status", status,
         "#1f7a3e" if status=="GREEN" else "#a67c00" if status=="AMBER" else "#8b0000")

kpi_card("Portfolio Stress", round(value,4), "#2c2f38")
kpi_card("Stress Alerts", alerts, "#2c2f38")

# =========================
# HERO GAUGE (CENTERED)
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

col_g1, col_g2, col_g3 = st.columns([1,2,1])
with col_g2:
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
# RISK BREAKDOWN (NEW 🔥)
# =========================

st.markdown("### 🔍 Risk Breakdown")

top5 = portfolio_df.sort_values("PRED_PROBA", ascending=False).head(5)

fig = px.bar(top5, x="SYMBOL", y="PRED_PROBA", color="PRED_PROBA")
st.plotly_chart(fig, use_container_width=True)

# =========================
# PORTFOLIO ALLOCATION (NEW 🔥)
# =========================

st.markdown("### 📦 Portfolio Allocation")

fig = px.pie(portfolio_df, names="SYMBOL", values="WEIGHT")
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
