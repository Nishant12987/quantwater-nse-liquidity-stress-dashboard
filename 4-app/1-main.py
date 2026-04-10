import streamlit as st
import sys
from pathlib import Path
import pandas as pd

# =========================
# FORCE ADD PROJECT PATHS
# =========================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "3-src"))
sys.path.insert(0, str(ROOT / "3-src/4-ensemble"))

# =========================
# IMPORTS
# =========================

from dashboard_backend import backend_pipeline
from components import liquidity_gauge, stress_trend_chart, top_risky_stocks, sector_heatmap

# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="Liquidity Risk Dashboard",
    layout="wide"
)

# =========================
# TITLE
# =========================

st.title("📊 Liquidity Risk Dashboard")
st.markdown("Backend Connected Successfully")

# =========================
# LOAD BACKEND DATA
# =========================

try:
    data = backend_pipeline()
    st.write("DEBUG backend:", data)
except Exception as e:
    st.error(f"Backend Error: {e}")
    data = None

# =========================
# LOAD MODEL DATA
# =========================

try:
    df = pd.read_csv("models/checkpoints/xgboost_full_predictions.csv")
    st.write("DEBUG df shape:", df.shape)
except Exception as e:
    st.error(f"CSV Load Error: {e}")
    df = None

# =========================
# LAYOUT
# =========================

col1, col2 = st.columns([1, 2])

# -------------------------
# LEFT → GAUGE
# -------------------------

with col1:
    st.subheader("Market Status")

    try:
        if data:
            liquidity_gauge(
                data.get("status", "UNKNOWN"),
                data.get("value", 0)
            )
        else:
            st.error("No backend data available")
    except Exception as e:
        st.error(f"Gauge Error: {e}")

# -------------------------
# RIGHT → TREND
# -------------------------

with col2:
    try:
        if df is not None:
            stress_trend_chart(df)
        else:
            st.error("No dataframe available")
    except Exception as e:
        st.error(f"Trend Error: {e}")

# =========================
# LOWER SECTION
# =========================

col3, col4 = st.columns(2)

with col3:
    try:
        if df is not None:
            top_risky_stocks(df)
    except Exception as e:
        st.error(f"Top Risk Error: {e}")

with col4:
    try:
        if df is not None:
            sector_heatmap(df)
    except Exception as e:
        st.error(f"Heatmap Error: {e}")
