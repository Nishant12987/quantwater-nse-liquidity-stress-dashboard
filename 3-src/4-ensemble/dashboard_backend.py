import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import importlib.util


# =========================
# FORCE LOAD stress_detector (FINAL FIX)
# =========================

BASE = Path(__file__).resolve().parents[2]

stress_path = BASE / "3-src/4-ensemble/stress_detector.py"

spec = importlib.util.spec_from_file_location("stress_detector", stress_path)
stress_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stress_module)

# 🔥 IMPORTANT LINE (this fixes your error)
sys.modules["stress_detector"] = stress_module

StressDetector = stress_module.StressDetector


# =========================
# DATA LOADING
# =========================

@st.cache_data
def load_data():
    df = pd.read_csv(
        BASE / "models/checkpoints/xgboost_full_predictions.csv",
        parse_dates=["DATE"]
    )
    return df


# =========================
# MODEL INFERENCE
# =========================

@st.cache_data
def run_inference(df):
    detector = StressDetector(BASE)
    results = detector.detect(df)
    return results


# =========================
# MARKET SIGNAL
# =========================

def compute_market_signal(results):

    daily = (
        results.groupby("DATE")["CRITICAL_ALERT"]
        .mean()
        .reset_index()
    )

    daily.rename(
        columns={"CRITICAL_ALERT": "MARKET_STRESS"},
        inplace=True
    )

    return daily


# =========================
# HEATMAP
# =========================

def compute_sector_heatmap(results):

    results["SECTOR"] = results["SYMBOL"].str[:2]

    heatmap = (
        results.groupby(["DATE", "SECTOR"])["CRITICAL_ALERT"]
        .mean()
        .reset_index()
    )

    return heatmap


# =========================
# STATUS LOGIC
# =========================

def get_latest_status(market_signal):

    latest = market_signal.iloc[-1]["MARKET_STRESS"]

    if latest < 0.05:
        return "GREEN", latest
    elif latest < 0.15:
        return "AMBER", latest
    else:
        return "RED", latest


# =========================
# PIPELINE
# =========================

def backend_pipeline():

    df = load_data()

    results = run_inference(df)

    market_signal = compute_market_signal(results)

    heatmap = compute_sector_heatmap(results)

    status, value = get_latest_status(market_signal)

    return {
        "results": results,
        "market_signal": market_signal,
        "heatmap": heatmap,
        "status": status,
        "value": value
    }
