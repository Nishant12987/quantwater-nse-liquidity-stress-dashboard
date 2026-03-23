import streamlit as st
import sys
from pathlib import Path


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


# =========================
# APP
# =========================

st.set_page_config(
    page_title="Liquidity Risk Dashboard",
    layout="wide"
)

data = backend_pipeline()

st.title("Liquidity Risk Dashboard")

st.write("Backend Connected Successfully")

st.subheader("Current Market Status")

st.write("Status:", data["status"])
st.write("Stress Value:", round(data["value"], 4))
