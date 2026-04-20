import pandas as pd
import numpy as np
from pathlib import Path


# =========================
# PATH SETUP
# =========================

BASE = Path(__file__).resolve().parents[2]


# =========================
# LOAD DATA (FIXED)
# =========================

def load_data():
    xgb_path = BASE / "models/checkpoints/xgboost_full_predictions.csv.gz"
    lstm_path = BASE / "models/checkpoints/lstm_predictions.csv.gz"

    if not xgb_path.exists():
        raise FileNotFoundError(f"{xgb_path} not found")

    if not lstm_path.exists():
        raise FileNotFoundError(f"{lstm_path} not found")

    # ✅ Load XGBoost predictions
    df = pd.read_csv(
        xgb_path,
        parse_dates=["DATE"],
        compression="gzip"
    )

    # ✅ Load LSTM predictions
    lstm_df = pd.read_csv(
        lstm_path,
        parse_dates=["DATE"],
        compression="gzip"
    )

    return df, lstm_df


# =========================
# MERGE DATA
# =========================

def merge_models(df, lstm_df):

    merged = df.merge(
        lstm_df,
        on=["DATE", "SYMBOL"],
        how="left"
    )

    # Fill missing safely
    merged["LSTM_SCORE"] = merged["LSTM_SCORE"].fillna(0)

    return merged


# =========================
# COMPUTE ALERTS
# =========================

def compute_alerts(df):

    # Dynamic threshold
    lstm_threshold = np.mean(df["LSTM_SCORE"]) + 2 * np.std(df["LSTM_SCORE"])

    df["CRITICAL_ALERT"] = (
        (df["PRED_STRESS"] == 1) &
        (df["LSTM_SCORE"] > lstm_threshold)
    ).astype(int)

    return df


# =========================
# MARKET STATUS
# =========================

def get_market_status(df):

    latest = df.sort_values("DATE").groupby("SYMBOL").tail(1)

    market_stress = latest["LSTM_SCORE"].mean()

    if market_stress < 0.05:
        status = "Normal"
    elif market_stress < 0.15:
        status = "Caution"
    else:
        status = "High Risk"

    return status, market_stress


# =========================
# MAIN PIPELINE
# =========================

def backend_pipeline():

    df, lstm_df = load_data()

    df = merge_models(df, lstm_df)

    df = compute_alerts(df)

    status, value = get_market_status(df)

    return {
        "status": status,
        "value": value,
        "results": df
    }
