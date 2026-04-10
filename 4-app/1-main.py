import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import importlib.util

# =========================
# FORCE ADD PROJECT PATHS
# =========================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "3-src"))
sys.path.insert(0, str(ROOT / "3-src/4-ensemble"))

# =========================
# IMPORT BACKEND
# =========================

from dashboard_backend import backend_pipeline

# =========================
# DYNAMIC IMPORT FOR 2-components.py
# =========================

components_path = ROOT / "4-app" / "2-components.py"

spec = importlib.util.spec_from_file_location("components", components_path)
components = importlib.util.module_from_spec(spec)
spec.loader.exec_module(components)

# Extract functions
liquidity_gauge = components.liquidity_gauge
stress_trend_chart = components.stress_trend_chart
top_risky_stocks = components.top_risky_stocks
sector_heatmap = components.sector_heatmap

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
# LOAD DATA
# =========================

data = backend_pipeline()

df = pd.read_csv("models/checkpoints/xgboost_full_predictions.csv")

# =========================
# LAYOUT
# =========================

col1, col2 = st.columns([1, 2])

# LEFT → GAUGE
with col1:
    liquidity_gauge(data.get("status", "UNKNOWN"), data.get("value", 0))

# RIGHT → TREND
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
