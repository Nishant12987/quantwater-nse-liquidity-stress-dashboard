import joblib
import numpy as np
import pandas as pd
from pathlib import Path


class StressDetector:

    def __init__(self, base_path):

        self.base = Path(base_path)

        self.xgb_model = joblib.load(
            self.base / "models/checkpoints/xgboost_stress_gatekeeper.pkl"
        )

        self.xgb_threshold = joblib.load(
            self.base / "models/checkpoints/xgboost_threshold.pkl"
        )

        # LOAD PRECOMPUTED LSTM CSV (GZ FIX APPLIED)
        self.lstm_data = pd.read_csv(
            self.base / "models/checkpoints/lstm_predictions.csv.gz",
            parse_dates=["DATE"],
            compression="gzip"
        )

        self.features = [
            "AMIHUD",
            "ROLL_SPREAD",
            "LIQUIDITY_COMMONALITY",
            "MARKET_AMIHUD_Z"
        ]

        self.lookback = 10


    def run_xgboost(self, df):

        X = df[self.features]

        prob = self.xgb_model.predict_proba(X)[:, 1]

        pred = (prob > self.xgb_threshold).astype(int)

        return prob, pred


    def run_lstm(self, df):

        merged = df.merge(
            self.lstm_data,
            on=["DATE", "SYMBOL"],
            how="left"
        )

        lstm_pred = merged["LSTM_SCORE"].fillna(0).values

        return lstm_pred[self.lookback:]


    def detect(self, df):

        df = df.copy()

        xgb_prob, xgb_pred = self.run_xgboost(df)

        lstm_pred = self.run_lstm(df)

        xgb_pred_aligned = xgb_pred[self.lookback:]

        lstm_threshold = np.mean(lstm_pred) + 2 * np.std(lstm_pred)

        critical_alert = (
            (xgb_pred_aligned == 1) &
            (lstm_pred > lstm_threshold)
        ).astype(int)

        results = df.iloc[self.lookback:].copy()

        results["XGB_PROB"] = xgb_prob[self.lookback:]
        results["XGB_STRESS"] = xgb_pred[self.lookback:]
        results["LSTM_SCORE"] = lstm_pred
        results["CRITICAL_ALERT"] = critical_alert

        return results


if __name__ == "__main__":

    BASE = "."

    data = pd.read_csv(
        Path(BASE) / "models/checkpoints/xgboost_full_predictions.csv.gz",
        parse_dates=["DATE"],
        compression="gzip"
    )

    detector = StressDetector(BASE)

    results = detector.detect(data)

    print("Total Critical Alerts:", results["CRITICAL_ALERT"].sum())
    print(results.head())
