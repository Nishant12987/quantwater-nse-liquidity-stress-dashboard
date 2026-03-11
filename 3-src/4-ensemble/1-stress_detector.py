# (Combining XGB + LSTM outputs)
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
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

        self.lstm_model = tf.keras.models.load_model(
            self.base / "models/checkpoints/lstm_stress_forecaster.keras"
        )

        self.features = [
            "AMIHUD",
            "ROLL_SPREAD",
            "LIQUIDITY_COMMONALITY",
            "MARKET_AMIHUD_Z"
        ]

        self.lookback = 10


    def prepare_lstm_sequences(self, df):

        values = df[self.features].values

        sequences = []

        for i in range(self.lookback, len(values)):
            sequences.append(values[i - self.lookback:i])

        return np.array(sequences)


    def run_xgboost(self, df):

        X = df[self.features]

        prob = self.xgb_model.predict_proba(X)[:, 1]

        pred = (prob > self.xgb_threshold).astype(int)

        return prob, pred


    def run_lstm(self, df):

        X_lstm = self.prepare_lstm_sequences(df)

        prob = self.lstm_model.predict(X_lstm).flatten()

        return prob


    def detect(self, df):

        df = df.copy()

        xgb_prob, xgb_pred = self.run_xgboost(df)

        lstm_prob = self.run_lstm(df)

        xgb_pred_aligned = xgb_pred[self.lookback:]

        critical_alert = (
            (xgb_pred_aligned == 1) &
            (lstm_prob > 0.5)
        ).astype(int)

        results = df.iloc[self.lookback:].copy()

        results["XGB_PROB"] = xgb_prob[self.lookback:]
        results["XGB_STRESS"] = xgb_pred[self.lookback:]
        results["LSTM_PROB"] = lstm_prob
        results["CRITICAL_ALERT"] = critical_alert

        return results


if __name__ == "__main__":

    BASE = "/content/data/NSE_Liquidity_Project"

    data = pd.read_csv(
        Path(BASE) / "models/checkpoints/xgboost_full_predictions.csv",
        parse_dates=["DATE"]
    )

    detector = StressDetector(BASE)

    results = detector.detect(data)

    print("Total Critical Alerts:", results["CRITICAL_ALERT"].sum())

    print(results.head())
