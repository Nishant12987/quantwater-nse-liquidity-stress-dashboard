import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import importlib.util

# =========================
# PATH SETUP
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
# DYNAMIC IMPORT (NO RENAME)
# =========================

components_path = ROOT / "4-app" / "2-components.py"

spec = importlib.util.spec_from_file_location("components", components_path)
components = importlib.util.module_from_spec(spec)
spec.loader.exec_module(components)

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

# =========================
# LOAD DATA (SAFE)
# =========================

try:
    data = backend_pipeline()
except Exception as e:
    st.error(f"Backend error: {e}")
    data = {}

try:
    df = pd.read_csv("models/checkpoints/xgboost_full_predictions.csv")
except Exception as e:
    st.error(f"CSV error: {e}")
    df = None

# =========================
# UI RENDER
# =========================

if data and df is not None:

    col1, col2 = st.columns([1, 2])

    with col1:
        liquidity_gauge(
            data.get("status", "UNKNOWN"),
            data.get("value", 0)
        )

    with col2:
        stress_trend_chart(df)

    col3, col4 = st.columns(2)

    with col3:
        top_risky_stocks(df)

    with col4:
        sector_heatmap(df)

else:
    st.warning("⚠️ Data not available. Check backend or model outputs.")
