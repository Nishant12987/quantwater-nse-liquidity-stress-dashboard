# UI elements (Heatmaps, Gauges)
import streamlit as st
import pandas as pd


# =========================
# LIQUIDITY GAUGE
# =========================

def liquidity_gauge(status, stress_value):

    if status == "GREEN":
        color = "#28a745"
    elif status == "AMBER":
        color = "#ffc107"
    else:
        color = "#dc3545"

    st.markdown(f"""
        <div style="
            padding: 20px;
            border-radius: 12px;
            background-color: {color};
            color: white;
            text-align: center;
            font-size: 26px;
            font-weight: bold;
        ">
            {status} MARKET<br>
            Stress Score: {stress_value:.2f}
        </div>
    """, unsafe_allow_html=True)


# =========================
# STRESS TREND
# =========================

def stress_trend_chart(df):
    st.subheader("📈 Market Stress Trend")

    trend = df.groupby("DATE")["PRED_STRESS"].mean()
    st.line_chart(trend)


# =========================
# TOP RISKY STOCKS
# =========================

def top_risky_stocks(df):
    st.subheader("🔥 Top Risky Stocks")

    top = df.sort_values("PRED_PROBA", ascending=False).head(10)

    st.dataframe(top[["SYMBOL", "PRED_PROBA"]])


# =========================
# SECTOR HEATMAP (SIMPLE VERSION)
# =========================

def sector_heatmap(df):
    st.subheader("🔥 Sector Heatmap (Proxy)")

    heatmap = df.groupby("SYMBOL")["PRED_STRESS"].mean().sort_values(ascending=False).head(20)

    st.bar_chart(heatmap)
