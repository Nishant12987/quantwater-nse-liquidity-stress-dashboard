import streamlit as st
import sys
from pathlib import Path


# =========================
# ADD ENSEMBLE PATH
# =========================

BASE = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE / "3-src/4-ensemble"))


# =========================
# IMPORT
# =========================

from dashboard_backend import backend_pipeline


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Liquidity Risk Dashboard",
    layout="wide"
)


# =========================
# RUN BACKEND
# =========================

data = backend_pipeline()


# =========================
# UI (TEST)
# =========================

st.title("Liquidity Risk Dashboard")

st.write("Backend Connected Successfully")

st.subheader("Current Market Status")

st.write("Status:", data["status"])
st.write("Stress Value:", round(data["value"], 4))
