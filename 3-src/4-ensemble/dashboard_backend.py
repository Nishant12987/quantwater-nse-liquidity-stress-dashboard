import sys
from pathlib import Path

# =========================
# PATH FIX (FINAL)
# =========================

BASE = Path(__file__).resolve().parents[2]

ENSEMBLE_PATH = BASE / "3-src/4-ensemble"
sys.path.insert(0, str(ENSEMBLE_PATH))


# =========================
# IMPORTS
# =========================

import pandas as pd
import streamlit as st

from stress_detector import StressDetector


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
# SECTOR HEATMAP
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
