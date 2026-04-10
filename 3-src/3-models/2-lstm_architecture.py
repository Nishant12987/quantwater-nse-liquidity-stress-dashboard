import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import os

from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# -------------------------------------------------
# Paths (FIXED — no config dependency)
# -------------------------------------------------

BASE = Path(os.getcwd())
FEATURES_DIR = BASE / "data/processed"
MODEL_DIR = BASE / "models/checkpoints"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# Load Feature Data
# -------------------------------------------------

amihud = pd.read_csv(FEATURES_DIR / "amihud.csv")
roll = pd.read_csv(FEATURES_DIR / "roll_spread.csv")
commonality = pd.read_csv(FEATURES_DIR / "liquidity_commonality.csv")
market = pd.read_csv(FEATURES_DIR / "market_amihud.csv", parse_dates=["DATE"])
labels = pd.read_csv(FEATURES_DIR / "stress_labels.csv", parse_dates=["DATE"])


df = (
    amihud
    .merge(roll, on="SYMBOL", how="left")
    .merge(commonality, on="SYMBOL", how="left")
)

df["DATE"] = pd.to_datetime(df["DATE"])
df = df.merge(market, on="DATE", how="left")
df = df.merge(labels, on="DATE", how="inner")


# -------------------------------------------------
# Feature Engineering (PIT SAFE)
# -------------------------------------------------

rolling_mean = df["MARKET_AMIHUD"].rolling(60).mean().shift(1)
rolling_std = df["MARKET_AMIHUD"].rolling(60).std().shift(1)

df["MARKET_AMIHUD_Z"] = (
    (df["MARKET_AMIHUD"] - rolling_mean) / rolling_std
)

FEATURES = [
    "AMIHUD",
    "ROLL_SPREAD",
    "LIQUIDITY_COMMONALITY",
    "MARKET_AMIHUD_Z"
]

df = df.dropna(subset=FEATURES + ["STRESS_LABEL"]).copy()

df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=FEATURES)

df = df.sort_values("DATE")


# -------------------------------------------------
# TRAIN / TEST SPLIT FIRST (NO LEAKAGE)
# -------------------------------------------------

split_index = int(len(df) * 0.75)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()


# -------------------------------------------------
# SCALING (FIT ONLY ON TRAIN)
# -------------------------------------------------

scaler = StandardScaler()

train_df.loc[:, FEATURES] = scaler.fit_transform(train_df[FEATURES])
test_df.loc[:, FEATURES] = scaler.transform(test_df[FEATURES])

# Save scaler
joblib.dump(scaler, MODEL_DIR / "lstm_scaler.pkl")


# -------------------------------------------------
# BUILD SEQUENCES
# -------------------------------------------------

LOOKBACK = 10

def build_sequences(df, features, lookback):
    X, y = [], []
    values = df[features].values
    targets = df["STRESS_LABEL"].values

    for i in range(lookback, len(df)):
        X.append(values[i - lookback:i])
        y.append(targets[i])

    return np.array(X), np.array(y)


X_train, y_train = build_sequences(train_df, FEATURES, LOOKBACK)
X_test, y_test = build_sequences(test_df, FEATURES, LOOKBACK)


print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


# -------------------------------------------------
# LSTM MODEL (REGRESSION)
# -------------------------------------------------

model = Sequential([
    LSTM(32, input_shape=(LOOKBACK, len(FEATURES))),
    Dropout(0.2),
    Dense(1, activation="linear")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

model.fit(
    X_train,
    y_train,
    epochs=3,
    batch_size=512,
    validation_data=(X_test, y_test),
    verbose=1
)


# -------------------------------------------------
# SAVE MODEL
# -------------------------------------------------

model.save(
    MODEL_DIR / "lstm_stress_forecaster.keras"
)

print("LSTM model + scaler saved successfully.")
