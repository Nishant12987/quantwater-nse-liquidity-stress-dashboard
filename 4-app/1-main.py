import pandas as pd
import numpy as np
from pathlib import Path


# =========================
# PATH SETUP (ROBUST)
# =========================

BASE = Path(__file__).resolve().parents[2]


# =========================
# LOAD DATA (SAFE + DEBUG)
# =========================

def load_data():
    xgb_path = BASE / "models" / "checkpoints" / "xgboost_full_predictions.csv.gz"
    lstm_path = BASE / "models" / "checkpoints" / "lstm_predictions.csv.gz"

    print(f"Looking for XGBoost file at: {xgb_path}")
    print(f"Looking for LSTM file at: {lstm_path}")

    if not xgb_path.exists():
        raise FileNotFoundError(f"❌ XGBoost file not found at {xgb_path}")

    if not lstm_path.exists():
        raise FileNotFoundError(f"❌ LSTM file not found at {lstm_path}")

    try:
        df = pd.read_csv(
            xgb_path,
            parse_dates=["DATE"],
            compression="gzip"
        )
    except Exception as e:
        raise RuntimeError(f"Error loading XGBoost file: {e}")

    try:
        lstm_df = pd.read_csv(
            lstm_path,
            parse_dates=["DATE"],
            compression="gzip"
        )
    except Exception as e:
        raise RuntimeError(f"Error loading LSTM file: {e}")

    return df, lstm_df


# =========================
# MERGE MODELS (SAFE)
# =========================

def merge_models(df, lstm_df):

    if "LSTM_SCORE" not in lstm_df.columns:
        raise ValueError("❌ LSTM file must contain 'LSTM_SCORE' column")

    merged = df.merge(
        lstm_df,
        on=["DATE", "SYMBOL"],
        how="left"
    )

    # Fill missing safely
    merged["LSTM_SCORE"] = merged["LSTM_SCORE"].fillna(0)

    return merged


# =========================
# COMPUTE ALERTS (STABLE)
# =========================

def compute_alerts(df):

    if df.empty:
        raise ValueError("❌ Dataframe is empty after merge")

    mean = df["LSTM_SCORE"].mean()
    std = df["LSTM_SCORE"].std()

    # fallback if std is NaN
    if pd.isna(std):
        std = 0

    lstm_threshold = mean + 2 * std

    df["CRITICAL_ALERT"] = (
        (df.get("PRED_STRESS", 0) == 1) &
        (df["LSTM_SCORE"] > lstm_threshold)
    ).astype(int)

    return df


# =========================
# MARKET STATUS (SAFE)
# =========================

def get_market_status(df):

    if df.empty:
        return "No Data", 0

    latest = df.sort_values("DATE").groupby("SYMBOL").tail(1)

    if latest.empty:
        return "No Data", 0

    market_stress = latest["LSTM_SCORE"].mean()

    if market_stress < 0.05:
        status = "Normal"
    elif market_stress < 0.15:
        status = "Caution"
    else:
        status = "High Risk"

    return status, market_stress


# =========================
# MAIN PIPELINE (ERROR SAFE)
# =========================

def backend_pipeline():

    try:
        df, lstm_df = load_data()

        df = merge_models(df, lstm_df)

        df = compute_alerts(df)

        status, value = get_market_status(df)

        return {
            "status": status,
            "value": value,
            "results": df
        }

    except Exception as e:
        print(f"❌ Backend failed: {e}")

        # Return safe fallback (prevents blank screen)
        return {
            "status": "Error",
            "value": 0,
            "results": pd.DataFrame()
        }
