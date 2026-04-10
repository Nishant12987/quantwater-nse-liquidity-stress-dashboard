import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
# DARK THEME (TERMINAL STYLE)
# =========================

st.markdown("""
<style>
body { background-color: #0e1117; color: white; }

.metric-card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}

.title {
    font-size: 28px;
    font-weight: bold;
}

.subtitle {
    color: #9aa4b2;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================

st.markdown('<div class="title">📊 Liquidity Risk Terminal</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Hedge Fund Monitoring System</div>', unsafe_allow_html=True)
st.markdown("---")

# =========================
# LOAD DATA
# =========================

data = backend_pipeline()
df = pd.read_csv("models/checkpoints/xgboost_full_predictions.csv")

df["DATE"] = pd.to_datetime(df["DATE"])

# =========================
# SIDEBAR FILTERS
# =========================

st.sidebar.header("⚙️ Controls")

stock = st.sidebar.selectbox(
    "Select Stock",
    ["ALL"] + sorted(df["SYMBOL"].unique())
)

date_range = st.sidebar.date_input(
    "Date Range",
    [df["DATE"].min(), df["DATE"].max()]
)

if stock != "ALL":
    df = df[df["SYMBOL"] == stock]

df = df[(df["DATE"] >= pd.to_datetime(date_range[0])) &
        (df["DATE"] <= pd.to_datetime(date_range[1]))]

# =========================
# KPI SECTION
# =========================

status = data.get("status", "UNKNOWN")
value = data.get("value", 0)
alerts = int(df["PRED_STRESS"].sum())

col1, col2, col3 = st.columns(3)

col1.metric("Market Status", status)
col2.metric("Stress Score", round(value, 4))
col3.metric("Critical Alerts", alerts)

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

st.subheader("📈 Stress Trend")

trend = df.groupby("DATE")["PRED_STRESS"].mean().reset_index()

fig = px.line(trend, x="DATE", y="PRED_STRESS")
st.plotly_chart(fig, use_container_width=True)

# =========================
# LOWER PANELS
# =========================

col4, col5 = st.columns([1.2, 1])

with col4:
    st.subheader("🔥 Top Risky Stocks")

    top = df.sort_values("PRED_PROBA", ascending=False).head(10)
    st.dataframe(top[["SYMBOL", "PRED_PROBA"]], use_container_width=True)

with col5:
    st.subheader("🔥 Stress Concentration")

    heatmap = df.groupby("SYMBOL")["PRED_STRESS"].mean().sort_values(ascending=False).head(15)
    fig = px.bar(heatmap)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# SIGNAL ENGINE
# =========================

st.markdown("---")
st.subheader("🧠 Signal Engine")

if value < 0.5:
    st.success("🟢 LOW RISK — Market stable")
elif value < 1.5:
    st.warning("🟡 CAUTION — Stress building")
else:
    st.error("🔴 HIGH RISK — Liquidity crisis possible")

# =========================
# ADVANCED ANALYTICS
# =========================

st.subheader("📊 Advanced Analytics")

volatility = df["PRED_STRESS"].std()
max_stress = df["PRED_STRESS"].max()

col6, col7 = st.columns(2)

col6.metric("Stress Volatility", round(volatility, 4))
col7.metric("Peak Stress", round(max_stress, 4))

if volatility > 0.5:
    st.warning("⚠️ Market showing unstable liquidity patterns")

# =========================
# ALERT PANEL
# =========================

st.subheader("🚨 Alerts")

alerts_df = df[df["PRED_STRESS"] == 1].sort_values("DATE", ascending=False).head(5)

if not alerts_df.empty:
    st.dataframe(alerts_df[["DATE", "SYMBOL", "PRED_PROBA"]])
else:
    st.success("No recent alerts")
