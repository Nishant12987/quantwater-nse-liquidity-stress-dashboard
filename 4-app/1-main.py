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
# IMPORT AFTER PATH FIX
# =========================

from dashboard_backend import backend_pipeline
from components import liquidity_gauge, stress_trend_chart, top_risky_stocks, sector_heatmap

# =========================
# APP CONFIG
# =========================

st.set_page_config(
    page_title="Liquidity Risk Dashboard",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================

data = backend_pipeline()

# Load predictions for charts
df = pd.read_csv("models/checkpoints/xgboost_full_predictions.csv")

# =========================
# HEADER
# =========================

st.title("📊 Liquidity Risk Dashboard")
st.markdown("Backend Connected Successfully")

# =========================
# LAYOUT
# =========================

col1, col2 = st.columns([1, 2])

# -------------------------
# LEFT → GAUGE
# -------------------------

with col1:
    liquidity_gauge(data["status"], data["value"])

# -------------------------
# RIGHT → TREND
# -------------------------

with col2:
    stress_trend_chart(df)

# =========================
# LOWER SECTION
# =========================

col3, col4 = st.columns(2)

with col3:
    top_risky_stocks(df)

with col4:
    sector_heatmap(df)
