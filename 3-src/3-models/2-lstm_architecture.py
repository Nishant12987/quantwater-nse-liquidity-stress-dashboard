import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# -------------------------------------------------
# Paths
# -------------------------------------------------

BASE = Path("/content/data/NSE_Liquidity_Project")
FEATURES_DIR = BASE / "features"
MODEL_DIR = BASE / "models" / "checkpoints"
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
# Feature Engineering (PIT SAFE FIX APPLIED)
# -------------------------------------------------

rolling_mean = df["MARKET_AMIHUD"].rolling(60).mean().shift(1)
rolling_std = df["MARKET_AMIHUD"].rolling(60).std().shift(1)

df["MARKET_AMIHUD_Z"] = (
    (df["MARKET_AMIHUD"] - rolling_mean)
    / rolling_std
)

FEATURES = [
    "AMIHUD",
    "ROLL_SPREAD",
    "LIQUIDITY_COMMONALITY",
    "MARKET_AMIHUD_Z"
]

df = df.dropna(subset=FEATURES + ["STRESS_LABEL"]).copy()

# Remove infinities
df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=FEATURES)

# Scale features
scaler = StandardScaler()
df[FEATURES] = scaler.fit_transform(df[FEATURES])

df = df.sort_values("DATE")


# -------------------------------------------------
# Build Sequences (10-day lookback)
# -------------------------------------------------

LOOKBACK = 10

X_sequences = []
y_sequences = []

values = df[FEATURES].values
targets = df["STRESS_LABEL"].values

for i in range(LOOKBACK, len(df)):
    X_sequences.append(values[i-LOOKBACK:i])
    y_sequences.append(targets[i])

X = np.array(X_sequences)
y = np.array(y_sequences)

print("X shape:", X.shape)
print("y shape:", y.shape)


# -------------------------------------------------
# Train / Test Split (Time Series Safe)
# -------------------------------------------------

split_index = int(len(X) * 0.75)

X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]


# -------------------------------------------------
# LSTM Model
# -------------------------------------------------

model = Sequential([
    LSTM(32, input_shape=(LOOKBACK, len(FEATURES))),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
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
# Evaluation
# -------------------------------------------------

y_prob = model.predict(X_test).flatten()
y_pred = (y_prob > 0.5).astype(int)

print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# -------------------------------------------------
# Save Model (Modern Keras Format)
# -------------------------------------------------

model.save(
    MODEL_DIR / "lstm_stress_forecaster.keras"
)
