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
# DARK THEME
# =========================

st.markdown("""
<style>
body { background-color: #0e1117; color: white; }
.title { font-size: 30px; font-weight: bold; }
.subtitle { color: #9aa4b2; }
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
# SIDEBAR
# =========================

st.sidebar.header("⚙️ Controls")

stock = st.sidebar.selectbox(
    "Select Stock",
    ["ALL"] + sorted(df["SYMBOL"].unique())
)

if stock != "ALL":
    df = df[df["SYMBOL"] == stock]

# =========================
# KPIs
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
# TREND + MOMENTUM
# =========================

col4, col5 = st.columns([2, 1])

trend = df.groupby("DATE")["PRED_STRESS"].mean().reset_index()

with col4:
    st.subheader("📈 Stress Trend")
    fig = px.line(trend, x="DATE", y="PRED_STRESS")
    st.plotly_chart(fig, use_container_width=True)

with col5:
    st.subheader("⚡ Momentum")

    recent = trend.tail(5)["PRED_STRESS"].mean()
    prev = trend.iloc[-10:-5]["PRED_STRESS"].mean()

    if recent > prev:
        st.error("📈 Stress Increasing")
    elif recent < prev:
        st.success("📉 Stress Decreasing")
    else:
        st.warning("➖ Stable")

# =========================
# TOP STRESSED STOCKS (AUTO)
# =========================

st.subheader("🔥 Top Stressed Stocks (Auto Signal)")

top_today = (
    df.sort_values("DATE")
    .groupby("SYMBOL")
    .tail(1)
    .sort_values("PRED_PROBA", ascending=False)
    .head(10)
)

st.dataframe(top_today[["SYMBOL", "PRED_PROBA"]], use_container_width=True)

# =========================
# SECTOR PROXY (SYMBOL CLUSTER)
# =========================

st.subheader("📊 Stress Distribution (Market Breadth)")

breadth = df.groupby("DATE")["PRED_STRESS"].mean()

fig = px.area(breadth)
st.plotly_chart(fig, use_container_width=True)

# =========================
# SMART ALERTS
# =========================

st.subheader("🚨 Smart Alerts (Recent Spikes)")

recent_alerts = df.sort_values("DATE", ascending=False).head(200)
recent_alerts = recent_alerts[recent_alerts["PRED_STRESS"] == 1]

st.dataframe(
    recent_alerts[["DATE", "SYMBOL", "PRED_PROBA"]].head(10),
    use_container_width=True
)

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
